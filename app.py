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

pc             = Pinecone(api_key=PINECONE_API_KEY)
index_baz      = pc.Index("ibnbaz")        # ← Ibn Baz
index_othaymeen = pc.Index("ibnothaymeen") # ← Ibn Othaymeen
oai            = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT_BOTH = """أنت مساعد إسلامي متخصص في نقل كلام العلماء الكبار رحمهم الله.

المصادر المتاحة: فتاوى الشيخ ابن باز رحمه الله، ومؤلفات الشيخ ابن عثيمين رحمه الله.

قواعد صارمة لا تُخالَف أبداً:
1. ابحث في جميع المصادر واذكر رأي كل عالم إن وُجد
2. إذا اتفق العالمان فاذكر ذلك صراحةً
3. إذا اختلفا فاعرض رأي كل منهما بوضوح مع ذكر اسمه
4. انقل النص الحرفي كما هو دون تغيير أي كلمة
5. الآيات القرآنية: اكتبها بالرسم العثماني الصحيح
6. الأحاديث النبوية: انقلها حرفياً كما وردت
7. إذا وجدت خطأ مطبعياً في آية أو حديث صححه
8. اذكر المصدر بعد كل اقتباس: (كتاب: [اسم الكتاب]، ص [رقم])
9. لا تضف أي كلام من عندك خارج المصادر
10. إذا لم تجد إجابة صريحة اذكر أقرب النصوص ثم قل: لم أجد نصاً صريحاً في هذه المصادر"""

SYSTEM_PROMPT_ONE = """أنت مساعد إسلامي متخصص في نقل كلام الشيخ {scholar} رحمه الله.

قواعد صارمة لا تُخالَف أبداً:
1. انقل كلام الشيخ {scholar} فقط من المصادر المعطاة
2. انقل النص الحرفي كما هو دون تغيير أي كلمة
3. الآيات القرآنية: اكتبها بالرسم العثماني الصحيح
4. الأحاديث النبوية: انقلها حرفياً كما وردت
5. إذا وجدت خطأ مطبعياً في آية أو حديث صححه
6. اذكر المصدر بعد كل اقتباس: (كتاب: [اسم الكتاب]، ص [رقم])
7. لا تضف أي كلام من عندك خارج المصادر
8. إذا لم تجد إجابة صريحة قل: لم أجد نصاً صريحاً للشيخ {scholar} في هذه المصادر"""

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get("X-API-Key") or request.args.get("api_key")
        if key != API_KEY:
            return jsonify({"error": "Unauthorized", "code": 401}), 401
        return f(*args, **kwargs)
    return decorated

def search_one(question, idx, scholar_label, top_k=5):
    """Chercher dans un index spécifique."""
    emb = oai.embeddings.create(input=question, model="text-embedding-3-small")
    results = idx.query(
        vector=emb.data[0].embedding,
        top_k=top_k,
        include_metadata=True
    )
    # Injecter le scholar dans les métadonnées si absent
    for m in results['matches']:
        if not m['metadata'].get('scholar'):
            m['metadata']['scholar'] = scholar_label
    return results['matches']

def search_both(question, top_k_each=5):
    """Chercher dans les deux index."""
    emb = oai.embeddings.create(input=question, model="text-embedding-3-small")
    vec = emb.data[0].embedding

    r_baz = index_baz.query(vector=vec, top_k=top_k_each, include_metadata=True)
    r_oth = index_othaymeen.query(vector=vec, top_k=top_k_each, include_metadata=True)

    matches_baz = r_baz['matches']
    matches_oth = r_oth['matches']

    for m in matches_baz:
        if not m['metadata'].get('scholar'):
            m['metadata']['scholar'] = 'Ibn Baz'
    for m in matches_oth:
        if not m['metadata'].get('scholar'):
            m['metadata']['scholar'] = 'Ibn Othaymeen'

    return matches_baz + matches_oth

def build_answer(question, matches, scholar=None):
    context_parts = []
    sources = []
    for match in matches:
        meta         = match['metadata']
        text         = meta.get('text', '')
        scholar_name = meta.get('scholar', '?')
        context_parts.append(
            f"[العالم: {scholar_name} | كتاب: {meta.get('book','?')} | ص {meta.get('printed_page','?')}]\n{text}"
        )
        sources.append({
            "book":    meta.get('book', '?'),
            "page":    meta.get('printed_page', '?'),
            "scholar": scholar_name,
            "text":    text,
            "preview": text[:300],
            "score":   round(match['score'], 3)
        })

    context = "\n\n---\n\n".join(context_parts)
    system  = SYSTEM_PROMPT_ONE.replace("{scholar}", scholar) if scholar else SYSTEM_PROMPT_BOTH

    response = oai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
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
    baz_count = index_baz.describe_index_stats()['total_vector_count']
    oth_count = index_othaymeen.describe_index_stats()['total_vector_count']
    return jsonify({
        "status":    "ok",
        "db_loaded": True,
        "document_count": baz_count + oth_count,
        "ibn_baz_count":       baz_count,
        "ibn_othaymeen_count": oth_count,
        "version":   "3.0.0"
    })

@app.route("/api/chat", methods=["POST"])
def chat():
    data     = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "السؤال فارغ"}), 400
    if len(question) > 1000:
        return jsonify({"error": "السؤال طويل جداً"}), 400
    matches         = search_both(question, top_k_each=5)
    answer, sources = build_answer(question, matches, scholar=None)
    return jsonify({"answer": answer, "sources": sources})

# ════════════════════════════
# PRIVATE API
# ════════════════════════════

@app.route("/v1/query", methods=["POST"])
@require_api_key
def api_query():
    data         = request.get_json()
    question     = data.get("question", "").strip()
    scholar_key  = data.get("scholar", None)
    top_k        = min(int(data.get("top_k", 5)), 10)
    sources_only = data.get("sources_only", False)

    if not question:
        return jsonify({"error": "question is required"}), 400
    if len(question) > 1000:
        return jsonify({"error": "question too long"}), 400

    start = time.time()

    if scholar_key == "ibn_baz":
        matches = search_one(question, index_baz, "Ibn Baz", top_k=top_k)
        scholar_label = "Ibn Baz"
    elif scholar_key == "ibn_othaymeen":
        matches = search_one(question, index_othaymeen, "Ibn Othaymeen", top_k=top_k)
        scholar_label = "Ibn Othaymeen"
    elif scholar_key is None:
        matches = search_both(question, top_k_each=top_k)
        scholar_label = "both"
    else:
        return jsonify({"error": f"Invalid scholar '{scholar_key}'. Use: ibn_baz, ibn_othaymeen, or null"}), 400

    sources = [{
        "book":    m['metadata'].get('book', '?'),
        "page":    m['metadata'].get('printed_page', '?'),
        "scholar": m['metadata'].get('scholar', '?'),
        "text":    m['metadata'].get('text', ''),
        "score":   round(m['score'], 4)
    } for m in matches]

    if sources_only:
        return jsonify({
            "question":   question,
            "scholar":    scholar_label,
            "sources":    sources,
            "latency_ms": round((time.time() - start) * 1000)
        })

    answer, _ = build_answer(question, matches, scholar=scholar_label if scholar_key else None)
    return jsonify({
        "question":   question,
        "scholar":    scholar_label,
        "answer":     answer,
        "sources":    sources,
        "latency_ms": round((time.time() - start) * 1000),
        "model":      "gpt-4o-mini",
        "top_k":      top_k
    })

@app.route("/v1/search", methods=["POST"])
@require_api_key
def api_search():
    data        = request.get_json()
    query       = data.get("query", "").strip()
    scholar_key = data.get("scholar", None)
    top_k       = min(int(data.get("top_k", 5)), 20)

    if not query:
        return jsonify({"error": "query is required"}), 400

    if scholar_key == "ibn_baz":
        matches = search_one(query, index_baz, "Ibn Baz", top_k=top_k)
        scholar_label = "Ibn Baz"
    elif scholar_key == "ibn_othaymeen":
        matches = search_one(query, index_othaymeen, "Ibn Othaymeen", top_k=top_k)
        scholar_label = "Ibn Othaymeen"
    elif scholar_key is None:
        matches = search_both(query, top_k_each=top_k)
        scholar_label = "both"
    else:
        return jsonify({"error": f"Invalid scholar '{scholar_key}'"}), 400

    return jsonify({
        "query":   query,
        "scholar": scholar_label,
        "results": [{
            "book":    m['metadata'].get('book', '?'),
            "page":    m['metadata'].get('printed_page', '?'),
            "scholar": m['metadata'].get('scholar', '?'),
            "text":    m['metadata'].get('text', ''),
            "score":   round(m['score'], 4)
        } for m in matches]
    })

@app.route("/v1/status", methods=["GET"])
@require_api_key
def api_status():
    baz_count = index_baz.describe_index_stats()['total_vector_count']
    oth_count = index_othaymeen.describe_index_stats()['total_vector_count']
    return jsonify({
        "status":              "ok",
        "document_count":      baz_count + oth_count,
        "ibn_baz_count":       baz_count,
        "ibn_othaymeen_count": oth_count,
        "scholars":            ["Ibn Baz", "Ibn Othaymeen"],
        "version":             "3.0.0"
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
