import database

def populate_multilingual_messages():
    """Populate the database with common multilingual messages"""
    messages = [
        {
            "english": "Welcome! Send 'feedback' to start the feedback process.",
            "marathi": "स्वागत! फीडबॅक प्रक्रिया सुरू करण्यासाठी 'फीडबॅक' पाठवा."
        },
        {
            "english": "Please upload a document or receipt for your feedback. Take a photo and send it now.",
            "marathi": "कृपया आपल्या फीडबॅकसाठी दस्तऐवज किंवा पावती अपलोड करा. आता फोटो काढा आणि पाठवा."
        },
        {
            "english": "Document received! Please enter the service center number.",
            "marathi": "दस्तऐवज प्राप्त झाला! कृपया सेवा केंद्र क्रमांक प्रविष्ट करा."
        },
        {
            "english": "Thank you! Please rate your experience from 1 (poor) to 5 (excellent).",
            "marathi": "धन्यवाद! कृपया आपला अनुभव 1 (वाईट) ते 5 (उत्कृष्ट) या प्रमाणे रेट करा."
        },
        {
            "english": "Thank you for your rating. Please provide any additional comments.",
            "marathi": "आपल्या रेटिंगसाठी धन्यवाद. कृपया अतिरिक्त टिप्पणी द्या."
        },
        {
            "english": "Thank you for your feedback! Your reference ID is: ",
            "marathi": "आपल्या फीडबॅकसाठी धन्यवाद! आपला संदर्भ ID आहे: "
        }
    ]
    
    for msg in messages:
        database.save_multilingual_message(msg["marathi"], msg["english"], "System")
    
    print(f"Added {len(messages)} multilingual messages to the database")

if __name__ == "__main__":
    populate_multilingual_messages()