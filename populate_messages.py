import database

def populate_multilingual_messages():
    """Populate the database with common multilingual messages"""
    messages = [
        {
            "english": "Welcome! Send 'feedback' to start the feedback process.",
            "marathi": "स्वागत! अभिप्राय  प्रक्रिया सुरू करण्यासाठी 'फीडबॅक' पाठवा."
        },
        {
            "english": "Please upload a document or receipt for your feedback or Take a photo and send it now.",
            "marathi": "कृपया आपल्या अभिप्राय साठी दस्तऐवज किंवा पावती अपलोड करा किंवा फोटो काढा आणि पाठवा."
        },
        {
            "english": "Document received! Please enter the service center owner name or CSC ID or MOL ID.",
            "marathi": "दस्तऐवज प्राप्त झाला! कृपया सेवा केंद्र चालकाचे नाव किवा CSC ID किवा MOL ID प्रविष्ट करा."
        },
        {
            "english":  "Thank you! Please rate your experience:\n\n 1️⃣ Very Poor\n 2️⃣ Poor\n 3️⃣ Average\n 4️⃣ Good\n 5️⃣ Excellent",
            "marathi": "धन्यवाद! आपली सेवा समाधानकारक होती का?\n\n 1️⃣ खूप वाईट\n 2️⃣ अत्यंत असमाधानकारक\n  3️⃣ असमाधानकारक\n 4️⃣ समाधानकारक\n  5️⃣ अत्यंत समाधानकारक",
        },
        {
            "english": "Thank you for your rating. Please provide any additional comments.",
            "marathi": "आपल्या रेटिंगसाठी धन्यवाद. कृपया अतिरिक्त टिप्पणी द्या."
        },
        {
            "english": "Thank you for your feedback! Your reference ID is: ",
            "marathi": "आपल्या अभिप्राय साठी धन्यवाद! आपला संदर्भ ID आहे: "
        },
        {
            "english": "Thank you for your rating. Please provide any additional comments.",
            "marathi": "आपल्या रेटिंगसाठी धन्यवाद. कृपया अतिरिक्त टिप्पणी द्या."
        },
        {
            "english": "Thank you! We have received your feedback! Your reference ID is: ",
            "marathi": "धन्यवाद! आपण दिलेला अभिप्राय आम्हाला मिळाला! आपला संदर्भ ID आहे: "
        },
        # Add these to your messages list
        {
            "english": "Welcome! Please select your language:",
            "marathi": "स्वागत! कृपया आपली भाषा निवडा:"
        },
        {
            "english": "1. English\n2. मराठी (Marathi)",
            "marathi": "1. English\n2. मराठी (Marathi)"
        },
        {
            "english": "Invalid selection. Please send 1 for English or 2 for Marathi (मराठी)",
            "marathi": "अवैध निवड. कृपया इंग्रजीसाठी 1 किंवा मराठीसाठी 2 पाठवा"
        },
        # Add these messages to your existing messages list:
        {
            "english": "What would you like to do?",
            "marathi": "आपण काय करू इच्छिता?"
        },
        {
            "english": "1. Provide feedback\n2. Get services information",
            "marathi": "1. अभिप्राय द्या\n2. सेवा माहिती मिळवा"
        },
        {
            "english": "Invalid selection. Please enter 1 for feedback or 2 for services information.",
            "marathi": "अवैध निवड. कृपया अभिप्रायासाठी 1 किंवा सेवा माहितीसाठी 2 प्रविष्ट करा."
        },
        {
            "english": "We're sending you our services information document. Please wait a moment...",
            "marathi": "आम्ही तुम्हाला आमच्या सेवांची माहिती पाठवत आहोत. कृपया थोडा वेळ थांबा..."
        },
        {
            "english": "Here is information about our services.",
            "marathi": "येथे आमच्या सेवांबद्दल माहिती आहे."
        }
    ]
    
    for msg in messages:
        database.save_multilingual_message(msg["marathi"], msg["english"], "System")
    
    print(f"Added {len(messages)} multilingual messages to the database")

if __name__ == "__main__":
    populate_multilingual_messages()