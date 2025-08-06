from twilio.rest import Client

ACCOUNT_SID = 'AC1c7991ceb3c0b24a3d5cd0ebe8f4bdd7'
AUTH_TOKEN = 'c4e514d51ed28e35b67f927a180f34f6'  # Replace with your actual auth token
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