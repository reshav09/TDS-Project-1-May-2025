# Reshav Sharma-2025-TDS-PROJECT 1: tds_virtual_ta/index.py

import os
from flask import Flask, request, jsonify
from rag import answer_question
from dotenv import load_dotenv

app = Flask(__name__)
load_dotenv()

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"message": "TDS Virtual TA API is running!"})

@app.route("/api", methods=["POST"])
def api():
    try:
        data = request.get_json()
        question = data.get("question")

        if not question:
            return jsonify({"error": "Question is required."}), 400

        answer, links = answer_question(question)
        return jsonify({"answer": answer, "links": links})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)