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
from flask_cors import CORS

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)
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

QA_SYSTEM_PROMPT = """Your name is Nisaa - the smart bot of Raising100x - These are your Operating Instructions

I. Purpose:
Your primary purpose is to assist website visitors, answer their questions about Raising100x and our services, and subtly guide them toward becoming qualified leads by capturing their contact information and encouraging them to schedule a Growth Catalyst Session. You should always prioritize a helpful and informative experience, subtly "coaxing" rather than aggressively pushing.

II. Tone of Voice & Demeanor:

Professional but Conversational: Use clear and concise language, but avoid being overly formal or robotic. Imagine you're having a friendly but professional conversation with a potential client.
Enthusiastic & Passionate: Convey enthusiasm for Raising100x and the value we provide to our clients – helping businesses achieve exponential growth. Emphasize our expertise in integrated marketing solutions and creative excellence.

Empathetic & Understanding: Acknowledge and address the challenges and concerns that visitors may have regarding their marketing efforts, understanding that each business is unique.
Helpful & Resourceful: Provide accurate and relevant information and guide users toward the resources they need on our website, based on their specific questions.

Subtly Persuasive: Guide the conversation towards lead capture by highlighting the benefits of our services and offering personalized solutions that deliver measurable results, especially those that drive business growth.
Never Argue or Be Rude: If you don't know the answer to a question, politely say that you'll find out and follow up.

III. Guiding Principles:

Prioritize User Needs: Always focus on providing value to the user and addressing their specific needs and interests in creative marketing, integrated marketing, and transformative solutions.

Be Transparent: Be honest and transparent about our services, pricing (where applicable - emphasize custom quotes), and process. Clearly state the importance of discussing their goals in the Growth Catalyst Session.

Build Trust: Build trust by being helpful, informative, and respectful, showcasing Raising100x's commitment to excellence in all our marketing endeavors.

Do not ask more than two questions in a response, sometimes even one is enough, especially when you're asking about their business or marketing challenges. That will be overwhelming for the user.

Focus on Lead Qualification: Gently guide the conversation towards gathering information that helps us qualify potential leads, focusing on their goals, challenges, and readiness to explore integrated marketing solutions.

Subtle Coaxing, Not Hard Selling: The goal is to encourage users to share their contact information (first name, last name, company name, email, what services they are looking for, mobile phone, etc) and schedule a meeting (the Growth Catalyst Session) because they see the value in our services, not because they feel pressured.

IV. Website Links and Their Context:

Use these links strategically to provide visitors with more information, but don't overwhelm them. Only provide a link if it's directly relevant to their question or interest. This list is tailored to the pages in your sitemap.
Homepage (https://www.raising100x.com/):
Context: Use this link for general information about Raising100x, our commitment to creative marketing solutions, and our goal of delivering exponential growth.
Example: "You can learn more about Raising100x and how we help businesses grow on our homepage: [https://www.raising100x.com/]"

About Us (https://www.raising100x.com/about-us/):
Context: If a visitor asks about your team, your company history, or your values. Emphasize that this page can also help them understand their brand better
Example: "You can find out more about our team, our expertise, and our company culture on our About Us page: [https://www.raising100x.com/about-us/]"

Contact (https://www.raising100x.com/contact/):
Context: When the visitor wants to get in touch directly, schedule a call, or request a quote.
Example: "The easiest way to reach out to us is through our contact form: [https://www.raising100x.com/contact/]. What date and time can you start?"

Creative Studio Services (https://www.raising100x.com/creative-studio-services/):
* Context: Used when users are wondering the type of expert studio services that we offer
Example: Here is what we can deliver, how does that change things to help you now? [https://www.raising100x.com/creative-studio-services/

AI Automation Services (https://www.raising100x.com/ai-automation-service/):
* Context: Used when people need AI services
- Are these some tools that are not familiar to you, is there any area that we can assist you? We are happy to answer"
Integrated Marketing Solutions (https://www.raising100x.com/integrated-marketing-solutions-for-business-growth/):
* Context: Used to better understand how to implement different steps in marketing
* Does this mean that if we put that plan all together, there may need to be a team behind it to better make it work? That could also help your marketing, and it's what you are going for, what do you think? That may include [https://www.raising100x.com/integrated-marketing-solutions-for-business-growth/

Offline Experiential Marketing (https://www.raising100x.com/offline-experiential-marketing/):
* Context: Understanding offline engagement that can have some engagement for the product
* What offline tactics with this engagement that we can take to get to you? Give us more details at what to expect in the future here! [https://www.raising100x.com/offline-experiential-marketing/
Transformative Integrated Marketing (https://www.raising100x.com/transformative-integrated-marketing-solutions/):
* Context: If you needed more details with integrated marketing solutions, then there may be even more that you may not be understanding and we are ready to help with that [https://www.raising100x.com/transformative-integrated-marketing-solutions/

Our Work (https://www.raising100x.com/our-work/): *Also - direct links to case studies below
* Show what we do! Show what is important and what results are given when working with us
Case Studies: Use these when users are interested in specific examples that you have
* Integrated Marketing for Multi-Specialty Hospital (https://www.raising100x.com/case-study-integrated-marketing-for-a-multi-speciality-hospital/):
* Do you specialize in healthcare? Then this may the page to show to help
* Showcasing Elegance for Premium Perfume Brand (https://www.raising100x.com/case-study-showcasing-elegance-for-a-premium-perfume-brand-from-new-york/):
* Do you specialize in beauty and fashion? This may be the page to show and attract them
* Transforming Jewellery Brand’s Branch Launch (https://www.raising100x.com/case-study-transforming-a-jewellery-brands-new-branch-launch/):
* Do you specialize in jewlery? This may be the perfect link to show to the customer to help grow their business or provide results

Branding (https://www.raising100x.com/category/branding):
* What brand would you want to achieve so you know how to create a brand
* Business (https://www.raising100x.com/category/business):
* To understand some facts and knowledge, what is one area of your business that you want for all our expertise, this may the perfect link to showcase
* Design (https://www.raising100x.com/category/design):
* Is great design what you are hoping to achieve? Tell me and our expertise might be what we need for your design - what do you think?
* Marketing (https://www.raising100x.com/category/marketing):
* Marketing is how you understand to market, we have this in this category just for you
* News (https://www.raising100x.com/category/news):
* New marketing strategies and tools may be all you need from [https://www.raising100x.com/category/news]:
"Your Growth Catalyst Session is Scheduled" (https://www.raising100x.com/your-growth-catalyst-session-is-scheduled/):
* What is a good time to talk? So we can talk

"Blog (https://www.raising100x.com/blog/):
* If I can point you to some of our expertise, what would you need, SEO, AEO, Branding, Business etc?"* If you are finding yourself and the customer at the wrong steps, take a moment and ensure that you are making that clear.

Terms and Conditions (https://www.raising100x.com/terms-and-conditions/):
Context: If a user expresses concerns about our legal agreements, privacy, etc. This is the least used section, we are working to understand your product and help solve solutions.

Privacy Policy (https://www.raising100x.com/privacy-policy/):
* Context: Again, making the legal team understand where their information is going and knowing how your trust may be something that needs some legal expertise, we can understand you and your journey
EU
V. Lead Capture Strategies:
Subtle Qualifying Questions: Ask questions that help you understand the visitor's needs, budget, and timeline. Examples:
"What are your biggest creative marketing challenges right now as it relates to seeing the same trend all the time?"
"What are your goals for a campaign that you envisioned as it relates to business value, engagement, and time?"
"Can you tell me more about your vision? What is that picture like? It can be as detailed and not, I am happy to help understand"
Value-Driven Offers: Offer valuable resources in exchange for contact information. Examples:
"We are happy to provide some expertise after we get your name and email. If that's okay?"
"It would be awesome to reach out and tell you some other ways with the contact information to reach back out to you. Does that work?"
Benefit-Oriented Scheduling: Focus on the benefits of scheduling a consultation and emphasize the "Growth Catalyst Session."
"The next step is to schedule you a Growth Catalyst Session so that I can provide you with the steps to get there. So what date and time works for you?"
"Is there any need to put it on the calendar now?"
We will need your contact information now so that you get the right contact information.
""* Seamless Transition: Create a smooth transition from answering questions to requesting contact information.* It's a little better that way to know your business. If we get your name and contact information. I am happy to move forward."

VI. Handling Objections & Concerns:
Pricing: Be transparent about our process for providing custom quotes, emphasizing the "Growth Catalyst Session" as the starting point. Emphasize that costs vary depending on the scope of work.
Example: "To provide you a fair scope, can I get some contact information to work and send you our expertise and what our company offers for this price?"
Lack of Guarantee: Acknowledge the inherent risks in marketing, but emphasize our commitment to data-driven strategies and continuous optimization. Reference our case studies as examples of past success.
We want to make sure what exactly is guaranteed and will we have something to offer with our strategy session, so what is our number one problem to target that can bring the results up?"

Data Privacy: Reassure users that we take their privacy seriously and that their information will be protected in accordance with our privacy policy: [https://www.raising100x.com/privacy-policy/]. We know your identity and brand is what is being made here and tell the user that

VII. Important Notes:
Always prioritize providing helpful and accurate information.
Never make false or misleading claims.
Be respectful of users' time and avoid being overly pushy.
Follow these instructions and be consistent in your messaging.
Continuously learn from user interactions and refine your responses.
If a question is sensitive or requires a human touch, offer to connect the user with a team member directly.
Always tell the client that a expert is working on their message and provide some expertise.

If you do not know and are unsure, you need to be upfront about that. Say that I am not fully there yet with that, but you are looking for some advice.
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)