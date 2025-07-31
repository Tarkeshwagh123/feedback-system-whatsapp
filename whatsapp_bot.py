from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse
import database
from alert_system import check_for_low_ratings
from config import TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER
from whatsapp_sender import send_whatsapp_template

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# send_whatsapp_template(
#     '+917710817591',
#     'HXb5b62575e6e4ff6129ad7c8efe1f983e',
#     '{"1":"12/1","2":"10pm"}'
# )

def send_whatsapp_message(to, message):
    """Send WhatsApp message using Twilio"""
    message = client.messages.create(
        body=message,
        from_=f"whatsapp:{TWILIO_PHONE_NUMBER}",
        to=f"whatsapp:{to}"
    )
    return message.sid

def process_whatsapp_message(sender, message):
    """Process incoming WhatsApp messages"""
    response = MessagingResponse()
    
    # Extract phone number from sender
    phone_number = sender.replace('whatsapp:', '')
    
    # Check if user is in a feedback session
    user_state = database.get_user_state(phone_number)
    
    if not user_state or user_state == 'IDLE':
        # New conversation
        if 'feedback' in message.lower():
            # Ask for reference ID
            response.message("Please enter your Service Reference ID to provide feedback")
            database.set_user_state(phone_number, 'AWAITING_REF_ID')
        else:
            response.message("Welcome to Citizen Feedback System. Send 'feedback' to start the feedback process.")
    
    elif user_state == 'AWAITING_REF_ID':
        # Validate reference ID
        ref_id = message.strip()
        if database.validate_reference_id(ref_id, phone_number):
            database.set_user_state(phone_number, 'AWAITING_RATING')
            database.set_current_ref_id(phone_number, ref_id)
            response.message("Please rate your experience from 1 (poor) to 5 (excellent)")
        else:
            response.message("Invalid Reference ID. Please try again or contact the service center.")
    
    elif user_state == 'AWAITING_RATING':
        try:
            rating = int(message.strip())
            if 1 <= rating <= 5:
                database.set_user_state(phone_number, 'AWAITING_COMMENT')
                database.set_current_rating(phone_number, rating)
                response.message("Thank you for your rating. Please provide any additional comments or feedback.")
                
                # Check for low ratings
                if rating <= 2:
                    check_for_low_ratings(phone_number, database.get_current_ref_id(phone_number), rating)
            else:
                response.message("Please provide a rating between 1 and 5.")
        except ValueError:
            response.message("Please enter a number between 1 and 5.")
    
    elif user_state == 'AWAITING_COMMENT':
        comment = message
        ref_id = database.get_current_ref_id(phone_number)
        rating = database.get_current_rating(phone_number)
        
        # Save feedback to database
        database.save_feedback(phone_number, ref_id, rating, comment)
        
        # Reset user state
        database.set_user_state(phone_number, 'IDLE')
        
        response.message("Thank you for your feedback! Your input helps us improve our services.")
    
    return response