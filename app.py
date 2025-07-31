from flask import Flask, request, jsonify, render_template
import database
from whatsapp_bot import process_whatsapp_message
from alert_system import check_for_low_ratings
from admin_dashboard import admin_bp

app = Flask(__name__)
app.register_blueprint(admin_bp, url_prefix='/admin')

@app.route('/webhook', methods=['POST'])
def webhook():
    """Endpoint to receive WhatsApp messages via Twilio"""
    incoming_msg = request.values.get('Body', '').strip()
    sender = request.values.get('From', '')
    
    # Process the incoming message
    response = process_whatsapp_message(sender, incoming_msg)
    
    return str(response)

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