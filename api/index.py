# Reshav Sharma-2025-TDS-PROJECT 1: tds_virtual_ta /index.py
"""
Author: Reshav Sharma
Project: TDS Virtual TA
Date: June 2025
Description: IITM TDS Virtual TA project integrating RAG using OpenAI embeddings.
"""


from flask import Flask, request, jsonify
from rag import answer_question
from dotenv import load_dotenv
import base64
import os

app = Flask(__name__)
load_dotenv()

@app.route("/", methods=["POST"])
def api():
    data = request.get_json()
    question = data.get("question")
    image_b64 = data.get("image")

    if not question:
        return jsonify({"error": "Question is required."}), 400

    answer, links = answer_question(question, image_b64 or None)

    return jsonify({
        "answer": answer,
        "links": links
    })
