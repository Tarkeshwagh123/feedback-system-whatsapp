from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse
import database
from alert_system import check_for_low_ratings
from config import TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER
from whatsapp_sender import send_whatsapp_template
import requests
import json
from io import BytesIO
import tempfile
from PIL import Image
import os
import qrcode
import uuid
# Import necessary libraries for document processing
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoImageProcessor, AutoModelForVision2Seq
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import pytesseract
import re

# Initialize Twilio client
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Global variables for AI models
llama_tokenizer = None
llama_model = None
image_processor = None
vision_model = None
embeddings = None
receipt_knowledge_base = None

def init_ai_models():
    """Initialize AI models for document processing with robust fallbacks"""
    global llama_tokenizer, llama_model, image_processor, vision_model, embeddings, receipt_knowledge_base
    
    try:
        print("Initializing AI models...")
        
        # Initialize fallback embeddings
        from langchain.embeddings import FakeEmbeddings
        embeddings = FakeEmbeddings(size=384)
        
        # Create basic RAG knowledge base with receipt examples
        receipt_examples = [
            Document(page_content="Receipt from ABC Store, Total: $45.67, Date: 2023-06-15, Items: Groceries, Payment: Credit Card"),
            Document(page_content="Service Invoice #12345, Amount: Rs.500, Service Center: XYZ-123, Date: 2023-07-20"),
            Document(page_content="Government Service Receipt, Center Number: GOV-789, Amount Paid: Rs.250, Date: 2023-08-10"),
            Document(page_content="Receipt for service at center KL-456, Total amount: Rs.1500, Transaction ID: TX789012")
        ]
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
        receipt_chunks = text_splitter.split_documents(receipt_examples)
        
        # Create the knowledge base with fallback embeddings
        receipt_knowledge_base = FAISS.from_documents(receipt_chunks, embeddings)
        
        # Skip vision model loading - use text-based extraction only
        image_processor = None
        vision_model = None
        
        print("AI models initialized with fallbacks")
        return True
    except Exception as e:
        print(f"Error initializing AI models: {e}")
        return False

def extract_text_from_image(image_path):
    """Extract text from an image using OCR"""
    try:
        return pytesseract.image_to_string(Image.open(image_path))
    except Exception as e:
        print(f"OCR error: {e}")
        return ""

def get_image_description(image_path):
    """Get a description of the image (fallback version)"""
    return "Image description not available - using OCR text only"


def generate_whatsapp_qr_code(message="feedback", filename="whatsapp_qr.png"):
    """
    Generate a QR code that opens WhatsApp with a pre-filled message
    
    Args:
        message: The message to pre-fill (default: "feedback")
        filename: Where to save the QR code image
        
    Returns:
        Path to the generated QR code file
    """
    try:
        # Clean phone number (remove "+" prefix required for wa.me format)
        clean_phone = TWILIO_PHONE_NUMBER.replace('+', '')
        
        # Create WhatsApp URL (wa.me format)
        whatsapp_url = f"https://wa.me/{clean_phone}?text={message}"
        
        print(f"Generating QR code for WhatsApp URL: {whatsapp_url}")
        
        # Generate QR code
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(whatsapp_url)
        qr.make(fit=True)
        
        # Create an image from the QR Code
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Save the image
        img.save(filename)
        
        print(f"QR code saved to {filename}")
        return os.path.abspath(filename)
    
    except Exception as e:
        print(f"Error generating WhatsApp QR code: {e}")
        return None

def process_with_rag(text_content, image_description):
    """Process the document content using RAG framework without rule-based extraction"""
    try:
        if receipt_knowledge_base is None or embeddings is None:
            return {"error": "RAG system not initialized"}
        
        # Combine OCR text and image description
        query = f"Receipt content: {text_content}. Image description: {image_description}"
        
        # Retrieve relevant examples from the knowledge base
        similar_docs = receipt_knowledge_base.similarity_search(query, k=2)
        context = "\n".join([doc.page_content for doc in similar_docs])
        
        # Use the simplified extraction function instead of pattern matching
        result = extract_data_from_text(text_content)
        result["other_details"] = f"Image shows: {image_description}"
        
        return result
    except Exception as e:
        print(f"RAG processing error: {e}")
        return {"error": f"Processing failed: {str(e)}"}

def extract_data_from_text(text_content):
    """Extract structured data from text without using pattern matching"""
    
    # Create result dictionary with default values
    result = {
        "center_id": "Unknown",
        "amount": "Unknown", 
        "date": "Unknown",
        "full_text": text_content
    }
    
    return result

def process_document(document_url):
    """Process the uploaded document without using rule-based extraction"""
    try:
        print(f"Processing document from URL: {document_url}")
        
        # Extract the Account SID from the URL instead of using the configured one
        # This ensures we're using the right credentials for the specific media
        url_parts = document_url.split('/')
        account_sid_index = url_parts.index('Accounts') + 1
        account_sid = url_parts[account_sid_index]
        
        # Download the document with authentication
        response = requests.get(
            document_url,
            auth=(account_sid, TWILIO_AUTH_TOKEN)
        )
        
        if response.status_code != 200:
            print(f"Failed to download document: {response.status_code}")
            # Try with the configured account SID as fallback
            response = requests.get(
                document_url,
                auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
            )
            if response.status_code != 200:
                print(f"Both authentication attempts failed with code: {response.status_code}")
                return None
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(response.content)
        
        # Extract text using OCR with language support
        text_content = pytesseract.image_to_string(
            Image.open(temp_path),
            lang='eng+hin+mar+guj+kan+tel+tam+ben'  # Add languages you need
        )
        print(f"OCR extracted text: {text_content[:100]}...")
        
        # Process with simple extraction
        processed_data = extract_data_from_text(text_content)
        
        # Clean up temporary file
        os.unlink(temp_path)
        
        return json.dumps(processed_data)
    except Exception as e:
        print(f"Error processing document: {e}")
        return json.dumps({"error": f"Document processing failed: {str(e)}"})

def detect_marathi(text):
    """
    Simple function to detect if text is likely Marathi
    This is a basic implementation - could be enhanced with proper language detection
    """
    # Devanagari Unicode range (used for Marathi)
    devanagari_pattern = re.compile(r'[\u0900-\u097F\u0980-\u09FF]')
    
    # If it contains Devanagari characters, assume it's Marathi
    return bool(devanagari_pattern.search(text))

def get_message_in_language(message_key, language="english"):
    """
    Retrieve message in specified language (english or marathi)
    If message not found, returns the original key
    """
    # Try to find the message in the database
    multilingual_msg = database.get_multilingual_message_by_english(message_key)
    
    if not multilingual_msg:
        return message_key
    
    if language.lower() == "marathi":
        return multilingual_msg["marathi_content"]
    else:
        return multilingual_msg["english_content"]
    
def send_whatsapp_message(to, message):
    """Send WhatsApp message using Twilio"""
    try:
        message = client.messages.create(
            body=message,
            from_=f"whatsapp:{TWILIO_PHONE_NUMBER}",
            to=f"whatsapp:{to}"
        )
        print(f"Message sent with SID: {message.sid}")
        return message.sid
    except Exception as e:
        print(f"Error sending WhatsApp message: {e}")
        return None

def process_whatsapp_message(sender, message, media_url=None):
    """
    Process incoming WhatsApp messages with language selection at the beginning of every conversation
    """
    print(f"Processing message from {sender}: '{message}'")
    response = MessagingResponse()
    
    # Extract phone number from sender
    phone_number = sender.replace('whatsapp:', '')
    
    # Get user's current state (or None if this is a new conversation)
    user_state = database.get_user_state(phone_number)
    
    # Always ask for language at the start of a conversation
    if not user_state or user_state == 'IDLE':
        # Set state to language selection at the beginning of every conversation
        database.set_user_state(phone_number, 'AWAITING_LANGUAGE')
        response.message("Welcome! Please select your language:\n1. English\n2. मराठी (Marathi)")
    
    elif user_state == 'AWAITING_LANGUAGE':
        # Process language selection
        if message.strip() == '1' or message.lower().strip() == 'english':
            language = "english"
            database.set_language_preference(phone_number, language)
            database.set_user_state(phone_number, 'ASK_FOR_FEEDBACK')
            response.message("Language set to English. Send 'feedback' to start the feedback process.")
        
        elif message.strip() == '2' or message.lower().strip() == 'marathi' or message.strip() == 'मराठी':
            language = "marathi"
            database.set_language_preference(phone_number, language)
            database.set_user_state(phone_number, 'ASK_FOR_FEEDBACK')
            response.message("भाषा मराठी वर सेट केली आहे. फीडबॅक प्रक्रिया सुरू करण्यासाठी 'फीडबॅक' पाठवा.")
        
        else:
            response.message("Invalid selection. Please send 1 for English or 2 for Marathi (मराठी)")
    
    elif user_state == 'ASK_FOR_FEEDBACK':
        # Get language preference for the user
        language = database.get_language_preference(phone_number)
        
        # Check if user wants to provide feedback
        if 'feedback' in message.lower() or message.strip() == 'फीडबॅक':
            # User wants to provide feedback, ask for document
            database.set_user_state(phone_number, 'AWAITING_DOCUMENT')
            
            response_text = get_message_in_language(
                "Please upload a document or receipt for your feedback. Take a photo and send it now.", 
                language
            )
            response.message(response_text)
        else:
            # Standard message handling for non-feedback messages
            sender_name = "WhatsApp User"
            database.save_simple_message(sender_name, phone_number, message)
            
            response_text = get_message_in_language(
                "Welcome! Send 'feedback' to start the feedback process.", 
                language
            )
            response.message(response_text)
    
    # The rest of your existing states remain the same
    elif user_state == 'AWAITING_DOCUMENT':
        language = database.get_language_preference(phone_number)
        # Process as before
        if media_url:
            document_url = media_url
            database.set_current_document_url(phone_number, document_url)
            document_data = process_document(document_url)
            print(f"Document data extracted: {document_data}")
            database.set_user_state(phone_number, 'AWAITING_CENTER')
            
            response_text = get_message_in_language(
                "Document received! Please enter the service center number.",
                language
            )
            response.message(response_text)
        else:
            response_text = get_message_in_language(
                "Please upload a document or receipt. Take a photo and send it now.",
                language
            )
            response.message(response_text)
    
    # The rest of your state handlers continue as before
    elif user_state == 'AWAITING_CENTER':
        language = database.get_language_preference(phone_number)
        # Save center number
        center_number = message.strip()
        database.set_current_center_number(phone_number, center_number)
        
        # Ask for rating
        database.set_user_state(phone_number, 'AWAITING_RATING')
        
        response_text = get_message_in_language(
            "Thank you! Please rate your experience from 1 (poor) to 5 (excellent).",
            language
        )
        response.message(response_text)
    
    elif user_state == 'AWAITING_RATING':
        language = database.get_language_preference(phone_number)
        try:
            rating = int(message.strip())
            if 1 <= rating <= 5:
                database.set_current_rating(phone_number, rating)
                database.set_user_state(phone_number, 'AWAITING_COMMENT')
                
                response_text = get_message_in_language(
                    "Thank you for your rating. Please provide any additional comments.",
                    language
                )
                response.message(response_text)
            else:
                response_text = get_message_in_language(
                    "Please provide a rating between 1 and 5.",
                    language
                )
                response.message(response_text)
        except ValueError:
            response_text = get_message_in_language(
                "Please enter a number between 1 and 5.",
                language
            )
            response.message(response_text)
    
    elif user_state == 'AWAITING_COMMENT':
        language = database.get_language_preference(phone_number)
        # Save comment
        comment = message
        database.set_current_comment(phone_number, comment)
        
        # Generate a reference ID
        ref_id = str(uuid.uuid4())[:8]
        
        # Save all collected data
        document_url = database.get_current_document_url(phone_number)
        center_number = database.get_current_center_number(phone_number)
        rating = database.get_current_rating(phone_number)
        
        # Get processed document data
        document_data = process_document(document_url)
        
        # Add reference ID to database
        database.add_reference_id(ref_id, phone_number)
        
        # Save complete feedback with document data
        database.save_feedback_with_document(
            phone_number, ref_id, rating, comment,
            center_number, document_url, document_data
        )
        
        # Reset user state to IDLE so next message will ask for language again
        database.set_user_state(phone_number, 'IDLE')
        
        # Send completion message with reference ID
        response_text = get_message_in_language(
            "Thank you for your feedback! Your reference ID is: ",
            language
        )
        response.message(f"{response_text}{ref_id}")
    
    else:
        # Unknown state, reset to IDLE
        database.set_user_state(phone_number, 'IDLE')
        
        # This will trigger language selection on next message
        response.message("Welcome! Please send any message to begin.")
    
    print(f"Sending response: {str(response)}")
    return response

# Initialize AI models when the module loads
init_ai_models()