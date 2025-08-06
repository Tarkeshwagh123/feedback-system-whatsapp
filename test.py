from twilio.rest import Client

account_sid = 'AC1c7991ceb3c0b24a3d5cd0ebe8f4bdd7'
auth_token = 'c4e514d51ed28e35b67f927a180f34f6'
client = Client(account_sid, auth_token)

message = client.messages.create(
    body="Test message from direct API call",
    from_='whatsapp:+14155238886',
    to='whatsapp:+917710817591'  # Your test number
)

print(message.sid)