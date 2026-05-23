from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Simple chatbot logic
def chatbot_response(user_input):
    user_input = user_input.lower()

    if "hello" in user_input:
        return "Hello! How can I help you?"
    elif "how are you" in user_input:
        return "I'm just code, but I'm doing great!"
    elif "bye" in user_input:
        return "Goodbye! Have a nice day!"
    elif "your name" in user_input:
        return "I am your AI chatbot."
    else:
        return "Sorry, I don't understand."

# Home route
@app.route("/")
def home():
    return render_template("index.html")

# Chat route
@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json["message"]
    response = chatbot_response(user_message)
    return jsonify({"response": response})

# Run app
if __name__ == "__main__":
    app.run(debug=True)