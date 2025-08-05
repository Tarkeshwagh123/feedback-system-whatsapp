from flask import Flask, request, jsonify, render_template
import database
from whatsapp_bot import process_whatsapp_message
from alert_system import check_for_low_ratings
from admin_dashboard import admin_bp
from flask import Flask, request, jsonify, render_template, send_file

app = Flask(__name__)
app.register_blueprint(admin_bp, url_prefix='/admin')


@app.route('/webhook', methods=['POST'])
def webhook():
    """Endpoint to receive WhatsApp messages via Twilio"""
    incoming_msg = request.values.get('Body', '').strip()
    sender = request.values.get('From', '')
    
    # Check for media attachments
    num_media = int(request.values.get('NumMedia', 0))
    media_url = None
    
    if num_media > 0:
        media_url = request.values.get('MediaUrl0')
    
    # Process the incoming message
    response = process_whatsapp_message(sender, incoming_msg, media_url)
    
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

@app.route('/service-center', methods=['GET', 'POST'])
def service_center():
    """Interface for Service Center to manage reference IDs"""
    if request.method == 'POST':
        ref_id = request.form.get('reference_id')
        citizen_contact = request.form.get('citizen_contact')
        database.add_reference_id(ref_id, citizen_contact)
        return jsonify({"status": "success"})
    
    return render_template('service_center.html')

if __name__ == '__main__':
    database.init_db()
    app.run(debug=True)