import re

def chatbot_response(user_input):
    user_input = user_input.lower().strip()

    # Greetings and small talk
    if re.search(r"\b(hello|hi|hey)\b", user_input):
        return "Hey there! How can I assist you?"
    
    elif re.search(r"how.*you", user_input):
        return "I'm a bot, but I’m here to help you!"
    
    elif re.search(r"\b(name|who are you)\b", user_input):
        return "I'm called RuleBot 😎"

    elif re.search(r"\b(help|support|assist)\b", user_input):
        return "Sure! Ask me about general knowledge, India, Telangana, or just say hi."

    # General Knowledge Questions
    elif "capital of india" in user_input:
        return "The capital of India is New Delhi."

    elif "president of india" in user_input:
        return "As of 2025, the President of India is Droupadi Murmu."

    elif "prime minister of india" in user_input:
        return "The Prime Minister of India is Narendra Modi."

    elif "national animal" in user_input:
        return "The national animal of India is the Bengal Tiger 🐅."

    elif "national flower" in user_input:
        return "The national flower of India is the Lotus 🌸."

    elif "national bird" in user_input:
        return "The national bird of India is the Indian Peacock 🦚."

    elif "largest state" in user_input:
        return "Rajasthan is the largest state in India by area."

    elif "smallest state" in user_input:
        return "Goa is the smallest state in India by area."

    # Info about Telangana
    elif "capital of telangana" in user_input:
        return "The capital of Telangana is Hyderabad."

    elif "cm of telangana" in user_input:
        return "As of 2025, the Chief Minister of Telangana is A. Revanth Reddy."

    elif "famous in telangana" in user_input:
        return "Telangana is famous for Charminar, Golconda Fort, biryani, and Bathukamma festival."

    elif "language of telangana" in user_input:
        return "Telugu is the most widely spoken language in Telangana."

    elif "telangana formed" in user_input:
        return "Telangana was officially formed on June 2, 2014."

    # Goodbye
    elif re.search(r"\b(bye|exit|goodbye)\b", user_input):
        return "Bye! Take care!"

    # Fallback
    else:
        return "Hmm... I didn't get that. You can ask me about India, Telangana, or some GK!"

# Start chat loop
print("🤖 RuleBot: Hello! I'm your friendly chatbot. Ask me about India, Telangana, or some GK. Type 'bye' to exit.")

while True:
    user_input = input("You: ")
    response = chatbot_response(user_input)
    print("RuleBot:", response)

    if re.search(r"\b(bye|exit|goodbye)\b", user_input.lower()):
        break
