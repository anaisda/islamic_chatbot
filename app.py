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
PINECONE_API_KEY = "pcsk_58M9di_B5eUQtz9rTRMFiuJB5jsJcytDy6ka7Z78vbLPv7HFyxNqtCUa7qTsBHovyLeHte"        # Your Pinecone key
API_KEY          = os.environ.get("CHATBOT_API_KEY", "dev-secret-key")

pc              = Pinecone(api_key=PINECONE_API_KEY)
index_baz       = pc.Index("fatawa-ibnbaz")
index_othaymeen = pc.Index("fatawa-ibnothaymeen")
oai             = OpenAI(api_key=OPENAI_API_KEY)

# ─── SYSTEM PROMPTS ───────────────────────────────────────────────────────────

SYSTEM_PROMPT_BOTH = """أنت مساعد إسلامي متخصص في نقل كلام العلماء الكبار رحمهم الله.
المصادر المتاحة: فتاوى الشيخ ابن باز رحمه الله، وفتاوى الشيخ ابن عثيمين رحمه الله.

قواعد صارمة لا تُخالَف أبداً:
1. اقرأ جميع المصادر المعطاة واختر منها ما يتعلق بالسؤال مباشرةً
2. تجاهل المصادر التي لا علاقة لها بالسؤال
3. إذا وجدت نصاً يجيب السؤال ولو جزئياً فاذكره
4. اذكر رأي كل عالم إن وُجد له نص ذو صلة
5. إذا اتفق العالمان فاذكر ذلك صراحةً
6. إذا اختلفا فاعرض رأي كل منهما بوضوح مع ذكر اسمه
7. انقل النص كما هو دون تغيير أي كلمة
8. الآيات القرآنية: اكتبها بالرسم العثماني الصحيح
9. الأحاديث النبوية: انقلها حرفياً كما وردت
10. اذكر المصدر بعد كل اقتباس: (المصدر: [اسم الكتاب]، ص [رقم])
11. لا تضف أي كلام من عندك خارج المصادر المعطاة
12. إذا لم يكن في المصادر المعطاة أي نص ذي صلة بالسؤال قل فقط: لم أجد نصاً صريحاً في هذه المصادر"""

SYSTEM_PROMPT_BAZ = """أنت مساعد إسلامي متخصص في نقل فتاوى الشيخ عبدالعزيز بن باز رحمه الله.

قواعد صارمة لا تُخالَف أبداً:
1. اقرأ جميع المصادر المعطاة واختر منها ما يتعلق بالسؤال مباشرةً
2. تجاهل المصادر التي لا علاقة لها بالسؤال
3. إذا وجدت نصاً يجيب السؤال ولو جزئياً فاذكره
4. انقل النص كما هو دون تغيير أي كلمة
5. الآيات القرآنية: اكتبها بالرسم العثماني الصحيح
6. الأحاديث النبوية: انقلها حرفياً كما وردت
7. اذكر المصدر بعد كل اقتباس: (المصدر: [اسم الكتاب]، ص [رقم])
8. لا تضف أي كلام من عندك خارج المصادر المعطاة
9. إذا لم يكن في المصادر المعطاة أي نص ذي صلة بالسؤال قل فقط: لم أجد نصاً صريحاً للشيخ ابن باز في هذه المصادر"""

SYSTEM_PROMPT_OTHAYMEEN = """أنت مساعد إسلامي متخصص في نقل فتاوى الشيخ محمد بن صالح العثيمين رحمه الله.

قواعد صارمة لا تُخالَف أبداً:
1. اقرأ جميع المصادر المعطاة واختر منها ما يتعلق بالسؤال مباشرةً
2. تجاهل المصادر التي لا علاقة لها بالسؤال
3. إذا وجدت نصاً يجيب السؤال ولو جزئياً فاذكره
4. انقل النص كما هو دون تغيير أي كلمة
5. الآيات القرآنية: اكتبها بالرسم العثماني الصحيح
6. الأحاديث النبوية: انقلها حرفياً كما وردت
7. اذكر المصدر بعد كل اقتباس: (المصدر: [اسم الكتاب]، ص [رقم])
8. لا تضف أي كلام من عندك خارج المصادر المعطاة
9. إذا لم يكن في المصادر المعطاة أي نص ذي صلة بالسؤال قل فقط: لم أجد نصاً صريحاً للشيخ ابن عثيمين في هذه المصادر"""

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get("X-API-Key") or request.args.get("api_key")
        if key != API_KEY:
            return jsonify({"error": "Unauthorized", "code": 401}), 401
        return f(*args, **kwargs)
    return decorated


def embed(text):
    return oai.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    ).data[0].embedding


def search_one(question, idx, scholar_label, top_k=8):
    results = idx.query(
        vector=embed(question),
        top_k=top_k,
        include_metadata=True
    )
    for m in results['matches']:
        if not m['metadata'].get('scholar'):
            m['metadata']['scholar'] = scholar_label
    return results['matches']


def search_both(question, top_k_each=8):
    vec   = embed(question)
    r_baz = index_baz.query(vector=vec, top_k=top_k_each, include_metadata=True)
    r_oth = index_othaymeen.query(vector=vec, top_k=top_k_each, include_metadata=True)

    for m in r_baz['matches']:
        if not m['metadata'].get('scholar'):
            m['metadata']['scholar'] = 'ابن باز'
    for m in r_oth['matches']:
        if not m['metadata'].get('scholar'):
            m['metadata']['scholar'] = 'ابن عثيمين'

    combined = r_baz['matches'] + r_oth['matches']
    combined.sort(key=lambda x: x['score'], reverse=True)
    return combined


def format_sources(matches):
    return [{
        "book"   : m['metadata'].get('book', '?'),
        "page"   : m['metadata'].get('printed_page', '?'),
        "scholar": m['metadata'].get('scholar', '?'),
        "section": m['metadata'].get('section', '?'),
        "url"    : m['metadata'].get('url', ''),
        "text"   : m['metadata'].get('text', ''),
        "preview": m['metadata'].get('text', '')[:300],
        "score"  : round(m['score'], 3),
    } for m in matches]


def build_answer(question, matches, system_prompt):
    context_parts = []
    for match in matches:
        meta = match['metadata']
        context_parts.append(
            f"[العالم: {meta.get('scholar','?')} | "
            f"المصدر: {meta.get('book','?')} | "
            f"ص {meta.get('printed_page','?')}]\n"
            f"{meta.get('text','')}"
        )

    context  = "\n\n---\n\n".join(context_parts)
    response = oai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"السؤال: {question}\n\nالمصادر المتاحة (اختر منها ما يتعلق بالسؤال):\n{context}"}
        ],
        temperature=0.0,
        max_tokens=2000,
    )
    return response.choices[0].message.content

# ─── PUBLIC ROUTES ────────────────────────────────────────────────────────────

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
        "status"             : "ok",
        "db_loaded"          : True,
        "document_count"     : baz_count + oth_count,
        "ibn_baz_count"      : baz_count,
        "ibn_othaymeen_count": oth_count,
        "version"            : "4.0.0"
    })

@app.route("/api/chat", methods=["POST"])
def chat():
    data        = request.get_json()
    question    = (data.get("question") or "").strip()
    scholar_key = data.get("scholar", None)

    if not question:
        return jsonify({"error": "السؤال فارغ"}), 400
    if len(question) > 1000:
        return jsonify({"error": "السؤال طويل جداً"}), 400

    if scholar_key == "ibn_baz":
        matches = search_one(question, index_baz, "ابن باز", top_k=8)
        prompt  = SYSTEM_PROMPT_BAZ
    elif scholar_key == "ibn_othaymeen":
        matches = search_one(question, index_othaymeen, "ابن عثيمين", top_k=8)
        prompt  = SYSTEM_PROMPT_OTHAYMEEN
    else:
        matches = search_both(question, top_k_each=8)
        prompt  = SYSTEM_PROMPT_BOTH

    answer  = build_answer(question, matches, prompt)
    sources = format_sources(matches)

    return jsonify({
        "answer" : answer,
        "sources": sources,
        "scholar": scholar_key or "both",
    })

# ─── PRIVATE API ──────────────────────────────────────────────────────────────

@app.route("/v1/query", methods=["POST"])
@require_api_key
def api_query():
    data         = request.get_json()
    question     = (data.get("question") or "").strip()
    scholar_key  = data.get("scholar", None)
    top_k        = min(int(data.get("top_k", 8)), 15)
    sources_only = data.get("sources_only", False)

    if not question:
        return jsonify({"error": "question is required"}), 400
    if len(question) > 1000:
        return jsonify({"error": "question too long"}), 400

    start = time.time()

    if scholar_key == "ibn_baz":
        matches       = search_one(question, index_baz, "ابن باز", top_k=top_k)
        scholar_label = "Ibn Baz"
        prompt        = SYSTEM_PROMPT_BAZ
    elif scholar_key == "ibn_othaymeen":
        matches       = search_one(question, index_othaymeen, "ابن عثيمين", top_k=top_k)
        scholar_label = "Ibn Othaymeen"
        prompt        = SYSTEM_PROMPT_OTHAYMEEN
    elif scholar_key is None:
        matches       = search_both(question, top_k_each=top_k)
        scholar_label = "both"
        prompt        = SYSTEM_PROMPT_BOTH
    else:
        return jsonify({"error": f"Invalid scholar '{scholar_key}'. Use: ibn_baz, ibn_othaymeen, or null"}), 400

    sources = format_sources(matches)

    if sources_only:
        return jsonify({
            "question"  : question,
            "scholar"   : scholar_label,
            "sources"   : sources,
            "latency_ms": round((time.time() - start) * 1000),
        })

    answer = build_answer(question, matches, prompt)
    return jsonify({
        "question"  : question,
        "scholar"   : scholar_label,
        "answer"    : answer,
        "sources"   : sources,
        "latency_ms": round((time.time() - start) * 1000),
        "model"     : "gpt-4o-mini",
        "top_k"     : top_k,
    })


@app.route("/v1/search", methods=["POST"])
@require_api_key
def api_search():
    data        = request.get_json()
    query       = (data.get("query") or "").strip()
    scholar_key = data.get("scholar", None)
    top_k       = min(int(data.get("top_k", 8)), 20)

    if not query:
        return jsonify({"error": "query is required"}), 400

    if scholar_key == "ibn_baz":
        matches       = search_one(query, index_baz, "ابن باز", top_k=top_k)
        scholar_label = "Ibn Baz"
    elif scholar_key == "ibn_othaymeen":
        matches       = search_one(query, index_othaymeen, "ابن عثيمين", top_k=top_k)
        scholar_label = "Ibn Othaymeen"
    elif scholar_key is None:
        matches       = search_both(query, top_k_each=top_k)
        scholar_label = "both"
    else:
        return jsonify({"error": f"Invalid scholar '{scholar_key}'"}), 400

    return jsonify({
        "query"  : query,
        "scholar": scholar_label,
        "results": format_sources(matches),
    })


@app.route("/v1/status", methods=["GET"])
@require_api_key
def api_status():
    baz_count = index_baz.describe_index_stats()['total_vector_count']
    oth_count = index_othaymeen.describe_index_stats()['total_vector_count']
    return jsonify({
        "status"             : "ok",
        "document_count"     : baz_count + oth_count,
        "ibn_baz_count"      : baz_count,
        "ibn_othaymeen_count": oth_count,
        "scholars"           : ["Ibn Baz", "Ibn Othaymeen"],
        "version"            : "4.0.0",
    })

# ─── ERROR HANDLERS ───────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found", "code": 404}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error", "code": 500}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
