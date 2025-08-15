import boto3
import json
import numpy as np
import base64
from transformers import pipeline
import re
from langchain.embeddings import BedrockEmbeddings
import database
import os

# Initialize AWS client
bedrock_client = None
bedrock_runtime = None
nlp_pipeline = None
embedding_model = None

def init_ai_services():
    """Initialize AWS Bedrock and AI services"""
    global bedrock_client, bedrock_runtime, nlp_pipeline, embedding_model
    
    try:
        # Initialize Bedrock client
        bedrock_client = boto3.client('bedrock', region_name='us-east-1')  # Or another region where the model is available
        bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')
        
        # Initialize lightweight NLP pipelines as fallback
        if nlp_pipeline is None:
            from transformers import pipeline
            nlp_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        
        # Initialize embedding model through Bedrock
        embedding_model = BedrockEmbeddings(
            model_id="amazon.titan-embed-text-v1",
            client=bedrock_client
        )
        
        return True
    except Exception as e:
        print(f"Failed to initialize AWS AI services: {e}")
        
        # Fallback to local models
        try:
            from sentence_transformers import SentenceTransformer
            embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            print("Fallback to local embedding model")
            return True
        except:
            print("Failed to initialize fallback models")
            return False

def invoke_llama_model(prompt, max_tokens=512):
    """Invoke Llama 4 Maverick model through Bedrock"""
    if not bedrock_runtime:
        print("AWS Bedrock not initialized")
        return None
    
    try:
        # Update the model ID to include the version suffix
        model_id = "meta.llama4-maverick-17b-instruct-v1:0"  # Changed from meta.llama4-maverick-17b-instruct
        body = json.dumps({
            "prompt": prompt,
            "max_gen_len": max_tokens,
            "temperature": 0.7,
            "top_p": 0.9
        })
        
        response = bedrock_runtime.invoke_model(
            modelId=model_id,
            body=body
        )
        
        response_body = json.loads(response['body'].read())
        return response_body['generation']
    except Exception as e:
        print(f"Error invoking Llama model: {e}")
        return None

# Implementation of AI services for WhatsApp feedback

def analyze_sentiment(text):
    """Analyze sentiment using Llama 4 or fallback to Hugging Face"""
    try:
        if bedrock_runtime:
            # Using Llama 4 for sentiment analysis
            prompt = f"""Analyze the sentiment of this text and classify it as POSITIVE, NEGATIVE, or NEUTRAL. 
                    Also provide a sentiment score between -1.0 (most negative) and 1.0 (most positive).
                    
                    Text: "{text}"
                    
                    Sentiment classification:"""
            
            response = invoke_llama_model(prompt, 100)
            
            if response:
                # Parse the response
                if "POSITIVE" in response:
                    score = 0.8  # Approximate score since Llama doesn't provide exact score
                    return {"label": "POSITIVE", "score": score}
                elif "NEGATIVE" in response:
                    score = -0.8
                    return {"label": "NEGATIVE", "score": score}
                else:
                    return {"label": "NEUTRAL", "score": 0.0}
        
        # Fallback to HuggingFace pipeline
        if nlp_pipeline:
            result = nlp_pipeline(text[:512])[0]
            
            # Convert to standard format
            if result["label"] == "POSITIVE":
                return {"label": "POSITIVE", "score": result["score"]}
            else:
                return {"label": "NEGATIVE", "score": -result["score"]}
    except Exception as e:
        print(f"Sentiment analysis error: {e}")
        return {"label": "NEUTRAL", "score": 0.0}

def detect_language(text):
    """Detect language of text"""
    try:
        if bedrock_runtime:
            prompt = f"""Identify the language of this text. Respond with only the language code (en, hi, mr, etc.)
                    
                    Text: "{text}"
                    
                    Language code:"""
                    
            response = invoke_llama_model(prompt, 50)
            
            if response:
                # Extract language code
                response = response.strip().lower()
                if "mar" in response or "mr" in response:
                    return "marathi"
                elif "hi" in response or "hin" in response:
                    return "hindi"
                elif "en" in response or "eng" in response:
                    return "english"
                else:
                    return response
        
        # Fallback to regex detection
        devanagari_pattern = re.compile(r'[\u0900-\u097F\u0980-\u09FF]')
        if bool(devanagari_pattern.search(text)):
            # Simple heuristic for differentiating Hindi vs Marathi
            marathi_chars = re.compile(r'[\u0950\u0902\u0903\u0904\u0905\u0906\u0907\u0908\u0909\u090A\u090B\u090C\u090D\u090E\u090F\u0910\u0911\u0912\u0913\u0914\u0915\u0916\u0917\u0918\u0919\u091A\u091B\u091C\u091D\u091E\u091F\u0920\u0921\u0922\u0923\u0924\u0925\u0926\u0927\u0928\u0929\u092A\u092B\u092C\u092D\u092E\u092F\u0930\u0931\u0932\u0933\u0934\u0935\u0936\u0937\u0938\u0939\u093A\u093B\u093C\u093D\u093E\u093F\u0940\u0941\u0942\u0943\u0944\u0945\u0946\u0947\u0948\u0949\u094A\u094B\u094C\u094D\u094E\u094F\u0950\u0951\u0952\u0953\u0954\u0955\u0956\u0957\u0958\u0959\u095A\u095B\u095C\u095D\u095E\u095F\u0960\u0961\u0962\u0963\u0964\u0965\u0966\u0967\u0968\u0969\u096A\u096B\u096C\u096D\u096E\u096F\u0970\u0971]')
            if bool(marathi_chars.search(text)):
                return "marathi"
            else:
                return "hindi" 
        return "english"
    except Exception as e:
        print(f"Language detection error: {e}")
        return "english"

def detect_toxicity(text):
    """Detect toxicity/abuse in text"""
    try:
        if bedrock_runtime:
            prompt = f"""Analyze this text and determine if it contains toxic, abusive, threatening, or inappropriate content.
                    Rate on a scale of 0 to 1, where 0 is completely safe and 1 is extremely toxic.
                    
                    Text: "{text}"
                    
                    Toxicity score (0-1):"""
                    
            response = invoke_llama_model(prompt, 50)
            
            if response:
                # Try to extract a numerical score
                match = re.search(r'([0-9]*[.]?[0-9]+)', response)
                if match:
                    score = float(match.group(1))
                    if score > 1:
                        score = score / 10  # Normalize if model gives score like 8/10
                    return score
                elif "0" in response:
                    return 0.0
                else:
                    # Estimate based on keywords
                    if any(word in response.lower() for word in ["high", "toxic", "inappropriate", "unsafe"]):
                        return 0.8
                    elif any(word in response.lower() for word in ["moderate", "some", "mild"]):
                        return 0.5
                    else:
                        return 0.1
        
        # Fallback to keyword-based detection
        toxic_words = ["abuse", "kill", "hate", "idiot", "stupid", "damn", "useless", "terrible"]
        count = sum(1 for word in toxic_words if word in text.lower())
        return min(count / 10, 1.0)  # Simple ratio with a cap at 1.0
    except Exception as e:
        print(f"Toxicity detection error: {e}")
        return 0.0

def extract_entities(text):
    """Extract entities (center names, amounts, dates) from text"""
    try:
        if bedrock_runtime:
            prompt = f"""Extract the following entities from this text:
                    1. Service center name or ID
                    2. Amount of money (if any)
                    3. Date (if any)
                    
                    Format your response as JSON with keys: service_center, amount, date
                    
                    Text: "{text}"
                    
                    Extracted entities:"""
                    
            response = invoke_llama_model(prompt, 200)
            
            if response:
                # Try to extract JSON
                try:
                    # Find JSON-like content in the response
                    json_match = re.search(r'({[^}]*})', response.replace('\n', ' '))
                    if json_match:
                        entity_data = json.loads(json_match.group(1))
                        return entity_data
                except:
                    pass
                
                # Fallback to regex extraction if JSON parsing fails
                entities = {}
                
                # Extract service center
                center_match = re.search(r'service_center["\s:]+([^,"}\n]+)', response)
                if center_match:
                    entities["service_center"] = center_match.group(1).strip('" \t')
                
                # Extract amount
                amount_match = re.search(r'amount["\s:]+([^,"}\n]+)', response)
                if amount_match:
                    entities["amount"] = amount_match.group(1).strip('" \t')
                
                # Extract date
                date_match = re.search(r'date["\s:]+([^,"}\n]+)', response)
                if date_match:
                    entities["date"] = date_match.group(1).strip('" \t')
                
                return entities
        
        # Fallback to simple regex extraction
        entities = {}
        
        # Simple regex for center ID patterns (alphanumeric with possible dash)
        center_match = re.search(r'([A-Za-z]{2,3}-\d{3,6}|\d{5,8})', text)
        if center_match:
            entities["service_center"] = center_match.group(1)
        
        # Amount pattern (Rs. or ₹ followed by numbers)
        amount_match = re.search(r'(Rs\.?\s*\d+(?:[,.]\d+)*|₹\s*\d+(?:[,.]\d+)*)', text)
        if amount_match:
            entities["amount"] = amount_match.group(1)
        
        # Date pattern (various formats)
        date_match = re.search(r'(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|\d{2,4}[-/.]\d{1,2}[-/.]\d{1,2})', text)
        if date_match:
            entities["date"] = date_match.group(1)
        
        return entities
    except Exception as e:
        print(f"Entity extraction error: {e}")
        return {}

def translate_text(text, source_lang=None, target_lang="english"):
    """Translate text between languages"""
    try:
        if not source_lang:
            source_lang = detect_language(text)
        
        if source_lang == target_lang:
            return text
        
        if bedrock_runtime:
            prompt = f"""Translate this text from {source_lang} to {target_lang}.
                    
                    Text: "{text}"
                    
                    Translation:"""
                    
            response = invoke_llama_model(prompt, 512)
            return response.strip() if response else text
        
        return text  # If Bedrock is not available, return original text
    except Exception as e:
        print(f"Translation error: {e}")
        return text

def generate_embedding(text):
    """Generate embedding vector for text using Bedrock or fallback"""
    try:
        if not text:
            return None
            
        if isinstance(embedding_model, BedrockEmbeddings):
            # Using AWS Bedrock embeddings
            embedding = embedding_model.embed_query(text)
            return ','.join(str(x) for x in embedding)
        else:
            # Using local model
            embedding = embedding_model.encode(text)
            return ','.join(str(x) for x in embedding)
    except Exception as e:
        print(f"Embedding generation error: {e}")
        return None

def classify_intent(text):
    """Classify user intent"""
    try:
        if bedrock_runtime:
            prompt = f"""Classify the intent of this message into one of these categories:
                    - feedback (user wants to provide feedback)
                    - service_info (user wants information about services)
                    - support (user needs help or has a question)
                    - complaint (user is making a specific complaint)
                    - other (none of the above)
                    
                    Respond with only the category name.
                    
                    Message: "{text}"
                    
                    Intent:"""
                    
            response = invoke_llama_model(prompt, 50)
            
            if response:
                response = response.strip().lower()
                if "feedback" in response:
                    return "feedback"
                elif "service_info" in response or "service" in response or "info" in response:
                    return "service_info"
                elif "support" in response or "help" in response:
                    return "support" 
                elif "complaint" in response:
                    return "complaint"
                else:
                    return "other"
        
        # Fallback to keyword matching
        text_lower = text.lower()
        if any(keyword in text_lower for keyword in ["feedback", "rate", "rating", "फीडबॅक", "अभिप्राय"]):
            return "feedback"
        elif any(keyword in text_lower for keyword in ["info", "service", "सेवा", "माहिती"]):
            return "service_info"
        elif any(keyword in text_lower for keyword in ["help", "question", "how", "what", "मदत"]):
            return "support"
        elif any(keyword in text_lower for keyword in ["complaint", "problem", "issue", "bad", "तक्रार"]):
            return "complaint"
        else:
            return "other"
    except Exception as e:
        print(f"Intent classification error: {e}")
        return "other"

def detect_duplicate_feedback(new_text, days=7):
    """Detect if this is a duplicate/similar feedback to recent submissions"""
    try:
        # Get recent feedback
        recent_feedback = database.get_all_feedback(days)
        
        if not recent_feedback:
            return False
            
        # Generate embedding for new text
        new_embedding = generate_embedding(new_text)
        
        if not new_embedding:
            return False
            
        new_vector = np.array([float(x) for x in new_embedding.split(',')])
        
        # Check similarity with recent feedback
        for feedback in recent_feedback:
            if 'comment' not in feedback or not feedback['comment']:
                continue
                
            # Skip comparing with own previous feedback
            if 'embedding' in feedback and feedback['embedding']:
                old_vector = np.array([float(x) for x in feedback['embedding'].split(',')])
                
                # Calculate cosine similarity
                similarity = np.dot(new_vector, old_vector) / (np.linalg.norm(new_vector) * np.linalg.norm(old_vector))
                
                if similarity > 0.85:  # High similarity threshold
                    return True
        
        return False
    except Exception as e:
        print(f"Duplicate detection error: {e}")
        return False

def generate_summary(feedback_list):
    """Generate a summary of feedback"""
    try:
        if not feedback_list:
            return "No feedback to summarize."
            
        if bedrock_runtime:
            # Prepare the input for the model
            feedback_text = "\n".join([
                f"- Rating: {item['rating']}/5, Comment: {item['comment']}" 
                for item in feedback_list if 'comment' in item and item['comment']
            ])
            
            prompt = f"""Summarize the key points from these customer feedback entries:
            
            {feedback_text[:2000]}  # Limit input size
            
            Provide a concise summary including:
            1. Overall sentiment
            2. Common themes
            3. Main issues (if any)
            4. Positive highlights (if any)
            
            Summary:"""
            
            response = invoke_llama_model(prompt, 300)
            return response if response else "Summary generation failed."
        
        # Fallback to simple statistical summary
        ratings = [item['rating'] for item in feedback_list if 'rating' in item]
        avg_rating = sum(ratings) / len(ratings) if ratings else 0
        low_ratings = sum(1 for r in ratings if r <= 2)
        high_ratings = sum(1 for r in ratings if r >= 4)
        
        return f"Summary of {len(feedback_list)} feedback entries: Average rating {avg_rating:.1f}/5, with {low_ratings} negative and {high_ratings} positive ratings."
    except Exception as e:
        print(f"Summary generation error: {e}")
        return "Summary generation failed."

def run_ai_processing(comment, document_text=None):
    """Process feedback with all AI capabilities"""
    result = {}
    
    # Combine comment and document text if available
    full_text = comment
    if document_text:
        full_text = f"{comment} {document_text}"
    
    # Detect language
    result['language'] = detect_language(full_text)
    
    # Translate non-English text for processing
    english_text = full_text
    if result['language'] != "english":
        english_text = translate_text(full_text, result['language'], "english")
    
    # Sentiment analysis
    sentiment = analyze_sentiment(english_text)
    result['sentiment'] = f"{sentiment['label']}:{sentiment['score']:.3f}"
    
    # Intent classification
    result['intent'] = classify_intent(english_text)
    
    # Toxicity detection
    result['toxicity_score'] = detect_toxicity(english_text)
    
    # Entity extraction
    entities = extract_entities(english_text)
    result['entities'] = entities
    
    # Generate embedding for similarity search and duplicate detection
    result['embedding'] = generate_embedding(english_text)
    
    return result

# Add new database columns for AI features
def add_ai_columns_to_db():
    """Add AI-related columns to database"""
    conn = sqlite3.connect('feedback.db')
    cursor = conn.cursor()
    
    columns = {
        "sentiment": "TEXT",
        "intent": "TEXT", 
        "toxicity_score": "REAL",
        "language": "TEXT",
        "entities": "TEXT",
        "embedding": "TEXT"
    }
    
    for column_name, data_type in columns.items():
        try:
            cursor.execute(f'ALTER TABLE feedback ADD COLUMN {column_name} {data_type}')
            print(f"Added {column_name} column")
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e):
                print(f"{column_name} column already exists")
            else:
                print(f"Error adding {column_name} column: {e}")
    
    conn.commit()
    conn.close()

# Update the save_feedback function to include AI fields
def save_feedback_with_ai(citizen_contact, ref_id, rating, comment, center_number, document_url, 
                        document_data=None, sentiment=None, intent=None, toxicity=None, 
                        language=None, entities=None, embedding=None):
    """Save feedback with AI analysis to the database"""
    conn = sqlite3.connect('feedback.db')
    cursor = conn.cursor()
    
    # Convert entities dict to JSON string if present
    entities_json = json.dumps(entities) if entities else None
    
    cursor.execute(
        '''INSERT INTO feedback (
        ref_id, citizen_contact, rating, comment, center_number, document_url, document_data,
        sentiment, intent, toxicity_score, language, entities, embedding)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
        (ref_id, citizen_contact, rating, comment, center_number, document_url, document_data,
        sentiment, intent, toxicity, language, entities_json, embedding)
    )
    conn.commit()
    conn.close()

# Initialize the AI services
init_ai_services()