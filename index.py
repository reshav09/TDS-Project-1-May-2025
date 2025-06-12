# Reshav Sharma-2025-TDS-PROJECT 1: tds_virtual_ta/index.py

from flask import Flask, request, jsonify
from rag import answer_question
from dotenv import load_dotenv

app = Flask(__name__)
load_dotenv()

@app.route("/", methods=["POST"])
def api():
    data = request.get_json()
    question = data.get("question")

    if not question:
        return jsonify({"error": "Question is required."}), 400

    answer, links = answer_question(question)
    return jsonify({"answer": answer, "links": links})
