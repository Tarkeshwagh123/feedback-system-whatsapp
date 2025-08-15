from flask import Flask, request, jsonify, render_template, send_file
import database
from whatsapp_bot import process_whatsapp_message, send_whatsapp_document
from alert_system import check_for_low_ratings
from admin_dashboard import admin_bp
import os
from datetime import datetime
from ai_services import init_ai_services

app = Flask(__name__)
app.register_blueprint(admin_bp, url_prefix='/admin')

# Initialize the database
database.init_db()

@app.route('/webhook', methods=['POST'])
def webhook():
    """Endpoint to receive WhatsApp messages via Twilio"""
    incoming_msg = request.values.get('Body', '').strip()
    sender = request.values.get('From', '')
    phone_number = sender.replace('whatsapp:', '')
    num_media = int(request.values.get('NumMedia', 0))
    media_url = None
    if num_media > 0:
        media_url = request.values.get('MediaUrl0')

    # Process the message to get TwiML response and next state
    response, next_state = process_whatsapp_message(sender, incoming_msg, media_url)
    
    # Handle special case for document sending
    if isinstance(next_state, tuple) and next_state[0] == 'SEND_DOCUMENT':
        _, recipient, doc_path, caption, after_state = next_state
        
        # We'll handle document sending asynchronously to avoid blocking
        import threading
        def send_doc():
            from whatsapp_bot import send_whatsapp_document
            send_whatsapp_document(recipient, doc_path, caption)
            database.set_user_state(phone_number, after_state)
            
        # Start document sending in background
        threading.Thread(target=send_doc).start()
    
    # For normal responses, update state based on TwiML response
    elif next_state:
        database.set_user_state(phone_number, next_state)
        database.update_last_interaction(phone_number)
        print(f"Updated state for {phone_number} to {next_state}")

    # Return only the TwiML response
    return str(response)

@app.route('/qr', methods=['GET'])
def whatsapp_qr():
    """Generate and serve a QR code to start WhatsApp conversation"""
    from whatsapp_bot import generate_whatsapp_qr_code
    
    # Get optional message parameter from query string
    message = request.args.get('message', 'feedback')
    
    # Generate QR code with specified message
    qr_path = generate_whatsapp_qr_code(message)
    
    if qr_path:
        return send_file(qr_path, mimetype='image/png')
    else:
        return "Failed to generate QR code", 500

@app.route('/documents/<filename>')
def serve_document(filename):
    """Serve a document from the documents directory"""
    return send_file(os.path.join('d:\\projects', filename))

@app.route('/contact', methods=['GET'])
def contact_page():
    """Landing page with QR code and instructions"""
    return render_template('contact.html')

@app.route('/reset/<phone_number>', methods=['GET'])
def reset_user(phone_number):
    """Admin endpoint to reset a user's state"""
    database.set_user_state(phone_number, 'IDLE')
    return f"Reset state for {phone_number}"

@app.route('/cleanup', methods=['GET'])
def cleanup_stale_conversations():
    """Cleanup endpoint to reset stale conversations"""
    count = database.cleanup_stale_conversations()
    return f"Reset {count} stale conversations"

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
    init_ai_services()