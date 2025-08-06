from flask import Flask, request, jsonify, render_template, send_file
import database
from whatsapp_bot import process_whatsapp_message, send_whatsapp_document
from alert_system import check_for_low_ratings
from admin_dashboard import admin_bp
import os
from datetime import datetime

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

    response, next_state = process_whatsapp_message(sender, incoming_msg, media_url)

    # Only update state if Twilio message was sent successfully
    message_sent = True
    try:
        # Try sending the response using Twilio REST API
        from whatsapp_bot import send_whatsapp_message
        # If response is TwiML, extract text
        resp_text = str(response)
        # You may need to parse TwiML to get the actual message text
        # For simplicity, let's assume it's plain text
        sid = send_whatsapp_message(phone_number, resp_text)
        if not sid:
            message_sent = False
    except Exception as e:
        print(f"Twilio send error: {e}")
        message_sent = False

    if message_sent and next_state:
        database.set_user_state(phone_number, next_state)
        database.update_last_interaction(phone_number)
        print(f"Updated state for {phone_number} to {next_state}")
    else:
        print("Message not sent, state not updated.")

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