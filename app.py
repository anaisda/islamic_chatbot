import os
import time
from functools import wraps
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from openai import OpenAI
from pinecone import Pinecone

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app, origins="*")

OPENAI_API_KEY   = os.environ.get("OPENAI_API_KEY", "")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
API_KEY          = os.environ.get("CHATBOT_API_KEY", "dev-secret-key")

pc    = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("ibnbaz")
oai   = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = """أنت مساعد إسلامي متخصص في نقل كلام الشيخ ابن باز رحمه الله.

قواعد صارمة لا تُخالَف أبداً:
1. انقل النص الحرفي كما هو من المصادر دون تغيير أي كلمة
2. الآيات القرآنية: اكتبها بالرسم العثماني الصحيح تماماً كما في المصحف الشريف
3. الأحاديث النبوية: انقلها حرفياً كما وردت في المصدر دون تعديل
4. إذا وجدت خطأ مطبعياً في آية أو حديث في المصدر، صححه من المصحف أو كتب الحديث المعتمدة
5. اذكر المصدر: (كتاب: [اسم الكتاب]، ص [رقم]) بعد كل اقتباس
6. لا تضف أي كلام من عندك خارج المصادر
7. إذا لم تجد إجابة صريحة، قل: لم أجد نصاً صريحاً في هذه المصادر"""

# ── Auth ──
def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get("X-API-Key") or request.args.get("api_key")
        if key != API_KEY:
            return jsonify({"error": "Unauthorized", "code": 401}), 401
        return f(*args, **kwargs)
    return decorated

# ── Core functions ──
def search_pinecone(question, top_k=5):
    emb = oai.embeddings.create(input=question, model="text-embedding-3-small")
    results = index.query(
        vector=emb.data[0].embedding,
        top_k=top_k,
        include_metadata=True
    )
    return results['matches']

def build_answer(question, matches):
    context_parts = []
    sources = []
    for match in matches:
        meta = match['metadata']
        text = meta.get('text', '')
        context_parts.append(f"[{meta.get('book','?')} | ص {meta.get('printed_page','?')}]\n{text}")
        sources.append({
            "book":    meta.get('book', '?'),
            "page":    meta.get('printed_page', '?'),
            "scholar": meta.get('scholar', 'ابن باز'),
            "text":    text,
            "preview": text[:300],
            "score":   round(match['score'], 3)
        })
    context  = "\n\n---\n\n".join(context_parts)
    response = oai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"السؤال: {question}\n\nالمصادر:\n{context}"}
        ],
        temperature=0.0
    )
    return response.choices[0].message.content, sources

# ════════════════════════════
# PUBLIC (frontend)
# ════════════════════════════

@app.route("/")
def home():
    return send_from_directory("static", "index.html")

@app.route("/docs")
def docs():
    return send_from_directory("static", "docs.html")

@app.route("/api/status")
def status():
    stats = index.describe_index_stats()
    return jsonify({
        "status":         "ok",
        "db_loaded":      True,
        "document_count": stats['total_vector_count'],
        "version":        "2.0.0"
    })

@app.route("/api/chat", methods=["POST"])
def chat():
    data     = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "السؤال فارغ"}), 400
    if len(question) > 1000:
        return jsonify({"error": "السؤال طويل جداً"}), 400
    matches        = search_pinecone(question, top_k=5)
    answer, sources = build_answer(question, matches)
    return jsonify({"answer": answer, "sources": sources})

# ════════════════════════════
# PRIVATE API (your app)
# ════════════════════════════

@app.route("/v1/query", methods=["POST"])
@require_api_key
def api_query():
    data         = request.get_json()
    question     = data.get("question", "").strip()
    top_k        = min(int(data.get("top_k", 5)), 10)
    sources_only = data.get("sources_only", False)

    if not question:
        return jsonify({"error": "question is required", "code": 400}), 400
    if len(question) > 1000:
        return jsonify({"error": "question too long", "code": 400}), 400

    start   = time.time()
    matches = search_pinecone(question, top_k=top_k)
    sources = [{
        "book":    m['metadata'].get('book', '?'),
        "page":    m['metadata'].get('printed_page', '?'),
        "scholar": m['metadata'].get('scholar', 'Ibn Baz'),
        "text":    m['metadata'].get('text', ''),
        "score":   round(m['score'], 4)
    } for m in matches]

    if sources_only:
        return jsonify({
            "question":   question,
            "sources":    sources,
            "latency_ms": round((time.time() - start) * 1000)
        })

    answer, _ = build_answer(question, matches)
    return jsonify({
        "question":   question,
        "answer":     answer,
        "sources":    sources,
        "latency_ms": round((time.time() - start) * 1000),
        "model":      "gpt-4o-mini",
        "top_k":      top_k
    })

@app.route("/v1/search", methods=["POST"])
@require_api_key
def api_search():
    data  = request.get_json()
    query = data.get("query", "").strip()
    top_k = min(int(data.get("top_k", 5)), 20)
    if not query:
        return jsonify({"error": "query is required"}), 400
    matches = search_pinecone(query, top_k=top_k)
    return jsonify({
        "query":   query,
        "results": [{
            "book":    m['metadata'].get('book', '?'),
            "page":    m['metadata'].get('printed_page', '?'),
            "scholar": m['metadata'].get('scholar', 'Ibn Baz'),
            "text":    m['metadata'].get('text', ''),
            "score":   round(m['score'], 4)
        } for m in matches]
    })

@app.route("/v1/status", methods=["GET"])
@require_api_key
def api_status():
    stats = index.describe_index_stats()
    return jsonify({
        "status":         "ok",
        "document_count": stats['total_vector_count'],
        "version":        "2.0.0"
    })

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found", "code": 404}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error", "code": 500}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
