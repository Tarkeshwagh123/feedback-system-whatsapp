from twilio.rest import Client

account_sid = 'ACb89f994be55774a5fb3ffa814a87c333'
auth_token = 'f0e5b804b4fb37bce09ad2d438b07534'
client = Client(account_sid, auth_token)

message = client.messages.create(
    body="Test message from direct API call",
    from_='whatsapp:+14155238886',
    to='whatsapp:+917710817591'  # Your test number
)

print(message.sid)