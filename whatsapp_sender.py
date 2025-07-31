from twilio.rest import Client

ACCOUNT_SID = 'ACb89f994be55774a5fb3ffa814a87c333'
AUTH_TOKEN = 'f0e5b804b4fb37bce09ad2d438b07534'  # Replace with your actual auth token
WHATSAPP_NUMBER = '+14155238886'

client = Client(ACCOUNT_SID, AUTH_TOKEN)

def send_whatsapp_template(to, content_sid, content_variables):
    """
    Send a WhatsApp template message using Twilio.
    :param to: Recipient WhatsApp number (e.g., '+917710817591')
    :param content_sid: Twilio Content SID for the template
    :param content_variables: JSON string of variables for the template
    :return: Message SID
    """
    message = client.messages.create(
        from_=f'whatsapp:{WHATSAPP_NUMBER}',
        content_sid=content_sid,
        content_variables=content_variables,
        to=f'whatsapp:{to}'
    )
    print(message.sid)
    return message.sid

# Example usage:
# send_whatsapp_template('+917710817591', 'HXb5b62575e6e4ff6129ad7c8efe1f983e', '{"1":"12/1","2":"3pm"}')