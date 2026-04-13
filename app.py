import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from openai import OpenAI
from pinecone import Pinecone

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("ibnbaz")
oai = OpenAI(api_key=OPENAI_API_KEY)

@app.route("/")
def home():
    return send_from_directory("static", "index.html")

@app.route("/api/status")
def status():
    stats = index.describe_index_stats()
    return jsonify({
        "status": "ok",
        "db_loaded": True,
        "document_count": stats['total_vector_count']
    })

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "السؤال فارغ"}), 400

    # Embedding avec le même modèle (768 dimensions)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    query_vector = model.encode(question).tolist()

    # Recherche Pinecone
    results = index.query(vector=query_vector, top_k=5, include_metadata=True)

    context_parts = []
    sources = []
    for match in results['matches']:
        meta = match['metadata']
        text = meta.get('text', '')
        context_parts.append(f"[{meta.get('book','?')} | ص {meta.get('printed_page','?')}]\n{text}")
        sources.append({
            "book": meta.get('book', '?'),
            "page": meta.get('printed_page', '?'),
            "scholar": meta.get('scholar', 'ابن باز'),
            "preview": text[:200],
            "score": round(match['score'], 3)
        })

    context = "\n\n---\n\n".join(context_parts)

    response = oai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "أنت مساعد إسلامي. اقتبس النص حرفياً من المصادر مع ذكر الكتاب والصفحة."},
            {"role": "user", "content": f"السؤال: {question}\n\nالمصادر:\n{context}"}
        ],
        temperature=0.0
    )

    return jsonify({
        "answer": response.choices[0].message.content,
        "sources": sources
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
