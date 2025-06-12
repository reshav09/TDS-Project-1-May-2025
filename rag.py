# Reshav Sharma-2025-TDS-PROJECT 1: tds_virtual_ta/rag.py

import numpy as np
from embed import load_index, load_metadata, embed_text
from openai import OpenAI
import os
import base64
from PIL import Image
import pytesseract
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("AIPROXY_TOKEN"),
    base_url="https://aiproxy.sanand.workers.dev/openai/v1"
)

index = load_index()
metadata = load_metadata()

# OCR helper (optional image processing)
def extract_text_from_image(image_b64):
    try:
        image = Image.open(BytesIO(base64.b64decode(image_b64)))
        return pytesseract.image_to_string(image)
    except Exception as e:
        return ""

# Retrieve top_k relevant chunks from FAISS index
def retrieve_relevant_chunks(query_embedding, top_k=5):
    D, I = index.search(np.array([query_embedding]), top_k)
    results = []
    sources = []

    for i in I[0]:
        if i < len(metadata):
            m = metadata[i]
            source_type = m.get("source_type", "course")  # default is course page

            if source_type == "course":
                chunk_text = f"- {m['chunk']} (Source: {m['source']})"
                link = {"url": m["source"], "text": m["chunk"][:80] + "..."}
            elif source_type == "discourse":
                full_url = "https://discourse.onlinedegree.iitm.ac.in" + m["post_url"]
                chunk_text = f"- {m['chunk']} (Post by {m['author']}: {full_url})"
                link = {"url": full_url, "text": m["chunk"][:80] + "..."}
            else:
                chunk_text = f"- {m['chunk']} (Source unknown)"
                link = {"url": "", "text": m["chunk"][:80] + "..."}

            results.append(chunk_text)
            sources.append(link)

    return results, sources

# Main RAG answer pipeline
def answer_question(question, image_b64=None):
    if image_b64:
        ocr_text = extract_text_from_image(image_b64)
        full_query = f"{question}\nImage context: {ocr_text}"
    else:
        full_query = question

    query_embedding = embed_text(full_query)
    relevant_chunks, links = retrieve_relevant_chunks(query_embedding)

    prompt = f"""
You are a helpful Virtual TA for the TDS course.

Answer the student's query below, using only the provided context.

Query: {question}

Relevant Context:
{chr(10).join(relevant_chunks)}

Always include helpful explanations, but only based on the context provided. Include source references when useful.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful virtual TA for the TDS course."},
            {"role": "user", "content": prompt}
        ]
    )

    answer_text = response.choices[0].message.content.strip()

    return answer_text, links
