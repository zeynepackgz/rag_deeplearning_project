from flask import Flask, request, jsonify, render_template
from chatbot import ChatBot
from modules.knowledge_enhancer import KnowledgeEnhancer

app = Flask(__name__, template_folder='templates')
ke = KnowledgeEnhancer()
chatbot = ChatBot()


@app.route("/")
def index():
    return render_template("chat.html")


@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get("message", "").strip()

        if not user_message:
            return jsonify({"error": "Empty message"}), 400

        response = chatbot.process_message(user_message, ke)
        return jsonify(response)

    except Exception as e:
        app.logger.error(f"Error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)