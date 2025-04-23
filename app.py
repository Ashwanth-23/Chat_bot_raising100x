from flask import Flask, request, jsonify, render_template, session
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_groq import ChatGroq  # Add import for Groq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_core.messages import HumanMessage
from langchain.chains import create_history_aware_retriever
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from pymongo import MongoClient
import secrets
import json
import time
import requests
from requests.auth import HTTPDigestAuth
from datetime import datetime
import uuid

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", secrets.token_urlsafe(16))

# Environment variables
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")  # Add Groq API key
DOCUMENT_PATH = os.getenv("DOCUMENT_PATH", "./data/Raising100x-FAQs.pdf")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-70b-8192")  # Add Groq model

# MongoDB Atlas Search Index configuration
ATLAS_PUBLIC_KEY = os.getenv("ATLAS_PUBLIC_KEY")
ATLAS_PRIVATE_KEY = os.getenv("ATLAS_PRIVATE_KEY")
ATLAS_GROUP_ID = os.getenv("ATLAS_GROUP_ID")
ATLAS_CLUSTER_NAME = os.getenv("ATLAS_CLUSTER_NAME")
DATABASE_NAME = "ChatbotDB"
INDEX_NAME = "vector_index"

# MongoDB setup
client = MongoClient(MONGODB_URI)
db = client.ChatbotDB
chat_collection = db.chat_history
lead_collection = db.lead_data
collection_name = "customer_data"

# Set OpenAI API Key
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
# Set Groq API Key
os.environ['GROQ_API_KEY'] = GROQ_API_KEY

# Initialize LLMs
llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
groq_llm = ChatGroq(model=GROQ_MODEL, temperature=0.5)  # Initialize Groq LLM with lower temperature for more precise extraction

# Prompt templates
CONTEXT_SYSTEM_PROMPT = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

QA_SYSTEM_PROMPT = """You are a helpful and trusted representative for Raising100x. Your goal is to provide the best possible experience for each user by understanding their needs and offering tailored solutions regarding Raising100x. You must only refer to the context provided to you to guide your answers for questions about Raising100x. Never answer something that is out of context with your own knowledge except for very basic requests.

*Conversation Flow:*

1. *Acknowledge Briefly:*  Acknowledge the user's question with a brief and friendly statement (e.g., "Thanks for your question!", "I understand.").

2.  *Answer (Raising100x Specific Questions):* If the reference document contains information directly relevant to the user's Raising100x-related query, use that information to provide a clear, concise, and helpful answer. Prioritize questions about Raising100x.

3.  *Limited General Knowledge (Optional):* If the user asks a general question that falls into one of the following categories, you may answer it briefly using your existing knowledge:
    *   *Date and Time:* (e.g., "What is today's date?", "What time is it?")
    *   *General Weather:* (e.g., "What's the weather like today?") - If you cannot infer the user's location, respond with "I cannot provide the weather because I do not have access to location information".
    *  *Basic Definitions* (e.g. "What is marketing?") Limit these definitions to a single sentence.

4.  *If the user is ASKING a follow-up question:* Refer to the previous chat history. Do not acknowledge the questions from previous rounds. Directly answer the questions if the provided document has the answer.

5.  *Complete Solution (Raising100x only):* Once you have a good understanding of the user's Raising100x-related needs (either from their initial query or after asking a hook question), provide a complete and personalized solution based on the reference document. Prioritize solutions that align with the user's stated goals.

6.  *Build Trust:* Throughout the conversation, be transparent, honest, and prioritize the user's best interests. End the conversation by reassuring them that you're there to help and encouraging them to reach out with any further questions. Thank them for contacting Raising100x and offer to connect them with an account manager.

*Constraints:*

*   You MUST base your answers and solutions on the information contained within the provided reference document for all Raising100x related questions. Do NOT make up information or rely on external knowledge.
*   You are allowed to use your existing knowledge for a few specific, very basic questions as mentioned in the general knowledge section. Do not seek out new or expanded knowledge.
*   *If a question doesn't relate to Raising100x or the General Knowledge questions, then politely respond with, "I'm designed to be a trusted advisor regarding Raising100x's services and offerings. I'm not equipped to handle inquiries outside of that scope."*
*   Limit yourself to a maximum of three sentences per response.
*   Do not include phrases like "According to the provided context" in your responses when answering Raising100x.
{context}
Question: {input}
Helpful Answer:"""

LEAD_EXTRACTION_PROMPT = """

 Extract the following information from the conversation if available:
        - name
        - email_id
        - contact_number
        - location
        - service_interest

        Return ONLY a valid JSON object with these fields with NO additional text before or after.
        If information isn't found, leave the field empty.
        
        Do not include any explanatory text, notes, or code blocks. Return ONLY the raw JSON.
        
        Conversation: {conversation}
"""

# Create prompt templates
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", CONTEXT_SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", QA_SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Chat history management
chat_store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in chat_store:
        chat_store[session_id] = ChatMessageHistory()
    return chat_store[session_id]

# Atlas Search Index management functions
def create_atlas_search_index():
    url = f"https://cloud.mongodb.com/api/atlas/v2/groups/{ATLAS_GROUP_ID}/clusters/{ATLAS_CLUSTER_NAME}/search/indexes"
    headers = {'Content-Type': 'application/json', 'Accept': 'application/vnd.atlas.2024-05-30+json'}
    data = {
        "collectionName": collection_name,
        "database": DATABASE_NAME,
        "name": INDEX_NAME,
        "type": "vectorSearch",
        "definition": {
            "fields": [
                {"type": "vector", "path": "embedding", "numDimensions": 1536, "similarity": "cosine"}
            ]
        }
    }
    response = requests.post(
        url, 
        headers=headers, 
        auth=HTTPDigestAuth(ATLAS_PUBLIC_KEY, ATLAS_PRIVATE_KEY), 
        data=json.dumps(data)
    )
    if response.status_code != 201:
        raise Exception(f"Failed to create Atlas Search Index: {response.status_code}, Response: {response.text}")
    return response

def get_atlas_search_index():
    url = f"https://cloud.mongodb.com/api/atlas/v2/groups/{ATLAS_GROUP_ID}/clusters/{ATLAS_CLUSTER_NAME}/search/indexes/{DATABASE_NAME}/{collection_name}/{INDEX_NAME}"
    headers = {'Accept': 'application/vnd.atlas.2024-05-30+json'}
    response = requests.get(
        url, 
        headers=headers, 
        auth=HTTPDigestAuth(ATLAS_PUBLIC_KEY, ATLAS_PRIVATE_KEY)
    )
    return response

def delete_atlas_search_index():
    url = f"https://cloud.mongodb.com/api/atlas/v2/groups/{ATLAS_GROUP_ID}/clusters/{ATLAS_CLUSTER_NAME}/search/indexes/{DATABASE_NAME}/{collection_name}/{INDEX_NAME}"
    headers = {'Accept': 'application/vnd.atlas.2024-05-30+json'}
    response = requests.delete(
        url, 
        headers=headers, 
        auth=HTTPDigestAuth(ATLAS_PUBLIC_KEY, ATLAS_PRIVATE_KEY)
    )
    return response

# Initialize vector store
def initialize_vector_store():
    # Verify file exists
    if not os.path.exists(DOCUMENT_PATH):
        raise FileNotFoundError(f"Document not found at: {DOCUMENT_PATH}")
    
    # Load document
    loader = PyPDFLoader(DOCUMENT_PATH)
    docs = loader.load()
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(docs)
    
    # Check and manage Atlas Search Index
    response = get_atlas_search_index()
    if response.status_code == 200:
        print("Deleting existing Atlas Search Index...")
        delete_response = delete_atlas_search_index()
        if delete_response.status_code == 204:
            # Wait for index deletion to complete
            print("Waiting for index deletion to complete...")
            while get_atlas_search_index().status_code != 404:
                time.sleep(5)
        else:
            raise Exception(f"Failed to delete existing Atlas Search Index: {delete_response.status_code}, Response: {delete_response.text}")
    elif response.status_code != 404:
        raise Exception(f"Failed to check Atlas Search Index: {response.status_code}, Response: {response.text}")
    
    # Store embeddings
    vector_search = MongoDBAtlasVectorSearch.from_documents(
        documents=final_documents,
        embedding=OpenAIEmbeddings(disallowed_special=()),
        collection=db[collection_name],
        index_name=INDEX_NAME,
    )
    
    # Debug: Verify documents in collection
    doc_count = db[collection_name].count_documents({})
    print(f"Number of documents in {collection_name}: {doc_count}")
    if doc_count > 0:
        sample_doc = db[collection_name].find_one()
        print(f"Sample document structure (keys): {sample_doc.keys()}")
    
    # Create new Atlas Search Index
    print("Creating new Atlas Search Index...")
    create_response = create_atlas_search_index()
    print(f"Atlas Search Index creation status: {create_response.status_code}")
    
    return vector_search

# Extract lead information using Groq API
def extract_lead_info(session_id):
    # Get chat history
    chat_doc = chat_collection.find_one({"session_id": session_id})
    if not chat_doc or "messages" not in chat_doc:
        return
    
    # Convert conversation to plain text
    conversation = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_doc["messages"]])
    
    try:


        # Use the Groq LLM to extract lead info
        response = groq_llm.invoke(LEAD_EXTRACTION_PROMPT.format(conversation=conversation))
        response_text = response.content.strip()
        # Extract JSON from potential markdown code blocks
        if "```json" in response_text or "```" in response_text:
            # Extract content between code blocks if present
            import re
            json_match = re.search(r"```(?:json)?\n(.*?)\n```", response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1).strip()

        try:
            lead_data = json.loads(response.content)
            print(f"Successfully parsed lead data: {lead_data}")
        except json.JSONDecodeError:
            print(f"Failed to parse JSON from Groq response: {response_text }")
            print(f"JSON error: {str(e)}")

            # Alternative approach: Use regex to find JSON-like structure
            import re
            json_pattern = r'\{[^}]*"name"[^}]*"email_id"[^}]*"contact_number"[^}]*"location"[^}]*"service_interest"[^}]*\}'
            json_match = re.search(json_pattern, response_text, re.DOTALL)
            
            if json_match:
                try:
                    lead_data = json.loads(json_match.group(0))
                    print(f"Extracted JSON using regex: {lead_data}")
                except json.JSONDecodeError:
                    # Fallback if all parsing fails
                    lead_data = {
                        "name": "",
                        "email_id": "",
                        "contact_number": "",
                        "location": "",
                        "service_interest": "",
                        "parsing_error": "Failed to parse response"
                    }
            else:
                # Final fallback
                lead_data = {
                    "name": "",
                    "email_id": "",
                    "contact_number": "",
                    "location": "",
                    "service_interest": "",
                    "raw_response": response_text[:500]  # Store part of the raw response for debugging
                }




            
        
        # Add session_id & timestamp
        lead_data["session_id"] = session_id
        lead_data["updated_at"] = datetime.utcnow()
        
        # Add LLM metadata for tracking
        lead_data["extraction_model"] = "groq_" + GROQ_MODEL
        
        # Save to MongoDB
        lead_collection.update_one(
            {"session_id": session_id},
            {"$set": lead_data},
            upsert=True
        )
    except Exception as e:
        print(f"[Lead Extraction Error] {e}")

# Initialize vector store
try:
    vector_search = initialize_vector_store()
    print("Vector store initialized successfully")
except Exception as e:
    print(f"Failed to initialize vector store: {e}")
    raise

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_session', methods=['GET'])
def generate_session():
    session_id = str(uuid.uuid4())
    return jsonify({"session_id": session_id})

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('message')
    session_id = data.get('session_id', str(uuid.uuid4()))
    
    if not user_input:
        return jsonify({'error': 'No input provided'}), 400
    
    # Create RAG pipeline
    document_chain = create_stuff_documents_chain(llm, qa_prompt)
    retriever = vector_search.as_retriever(search_type="similarity", search_kwargs={"k": 5, "score_threshold": 0.75})
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    retrieval_chain = create_retrieval_chain(history_aware_retriever, document_chain)
    
    conversational_rag_chain = RunnableWithMessageHistory(
        retrieval_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    
    try:
        # Get response from RAG
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        answer = response['answer']
        
        # Store message in MongoDB
        chat_collection.update_one(
            {"session_id": session_id},
            {
                "$push": {
                    "messages": {
                        "$each": [
                            {"role": "user", "content": user_input},
                            {"role": "assistant", "content": answer}
                        ]
                    }
                },
                "$setOnInsert": {"created_at": datetime.utcnow()}
            },
            upsert=True
        )
        
        # Extract lead info after sufficient conversation
        message_count = len(chat_collection.find_one({"session_id": session_id}).get("messages", []))
        if message_count >= 4:  # Extract after 2 user messages
            extract_lead_info(session_id)
        
        return jsonify({'response': answer}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/leads', methods=['GET'])
def get_leads():
    # Simple admin route to get all leads (should be protected in production)
    leads = list(lead_collection.find({}, {"_id": 0}))
    return jsonify(leads)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')