import os
import json
import pickle
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

# ─────────────────────────────────────────────
# CONFIGURATION — modifier selon ton setup
# ─────────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "sk-...")  # Mettre ta clé ici ou dans .env
FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "./islamic_faiss_index.pkl")  # Chemin vers ton fichier FAISS
USE_CHROMADB = True   # ← changer False en True
CHROMADB_PATH = "ChromaDB_export"  # ← ton chemin Drive# ─────────────────────────────────────────────

# Chargement de la base de données
db = None
index = None
texts = None
metadata = None
collection = None
model = None

def load_database():
    global db, index, texts, metadata, collection, model

    if USE_CHROMADB:
        try:
            import chromadb
            from chromadb.utils import embedding_functions
            emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
            )
            client = chromadb.PersistentClient(path=CHROMADB_PATH)
            collection = client.get_collection("ibn_baz_library", embedding_function=emb_fn)
            print(f"✅ ChromaDB chargé — {collection.count()} documents")
        except Exception as e:
            print(f"❌ Erreur ChromaDB: {e}")
    else:
        try:
            import faiss
            from sentence_transformers import SentenceTransformer

            print(f"📂 Chargement de {FAISS_INDEX_PATH}...")
            with open(FAISS_INDEX_PATH, 'rb') as f:
                data = pickle.load(f)

            index = faiss.deserialize_index(data['index'])
            texts = data['texts']
            metadata = data['metadata']
            model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
            print(f"✅ FAISS chargé — {index.ntotal} documents")
        except Exception as e:
            print(f"❌ Erreur FAISS: {e}")
            print("⚠️  Mode démo activé (sans base réelle)")

def search_database(query, top_k=5):
    """Recherche dans FAISS ou ChromaDB"""
    if USE_CHROMADB and collection:
        results = collection.query(query_texts=[query], n_results=top_k)
        output = []
        for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
            output.append({"text": doc, "meta": meta, "score": 1.0})
        return output

    elif index is not None and model is not None:
        import faiss
        q_emb = model.encode([query]).astype('float32')
        faiss.normalize_L2(q_emb)
        scores, indices = index.search(q_emb, top_k)
        output = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0:
                output.append({
                    "text": texts[idx],
                    "meta": metadata[idx],
                    "score": float(score)
                })
        return output
    else:
        # Mode démo — retourner des données fictives
        return [
            {
                "text": "هذا نص تجريبي. يرجى تحميل قاعدة البيانات أولاً.",
                "meta": {"book": "قاعدة البيانات غير محملة", "printed_page": "—", "scholar": "—", "filename": "demo"},
                "score": 0.99
            }
        ]

def ask_openai(question, context_results):
    """Appel OpenAI avec le contexte trouvé"""
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    context_parts = []
    for r in context_results:
        meta = r['meta']
        context_parts.append(
            f"[من كتاب: {meta.get('book','?')} | صفحة: {meta.get('printed_page','?')}]\n{r['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    system_prompt = """أنت مساعد إسلامي متخصص في فتاوى الشيخ ابن باز رحمه الله.

قواعد صارمة يجب اتباعها:
1. اقتبس النص كما هو بالضبط من المصادر دون تعديل أي حرف أو كلمة
2. لا تُضِف أي معلومات من خارج المصادر المعطاة
3. اذكر اسم الكتاب ورقم الصفحة مع كل اقتباس بهذا الشكل: (كتاب: [الاسم]، ص [رقم])
4. إذا لم تجد إجابة واضحة في المصادر، قل: "لم أجد نصاً صريحاً في هذه المصادر"
5. لا تُفتِ من عندك، اكتفِ بنقل كلام الشيخ حرفياً"""

    user_prompt = f"""السؤال: {question}

المصادر المتاحة:
{context}

أجب على السؤال مستشهداً بالنصوص الحرفية مع ذكر المصدر لكل اقتباس."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0,
        max_tokens=1500
    )
    return response.choices[0].message.content


# ─────────────────────────────────────────────
# ROUTES API
# ─────────────────────────────────────────────

@app.route("/")
def index_page():
    return send_from_directory("static", "index.html")

@app.route("/api/status")
def status():
    db_loaded = (index is not None) or (collection is not None)
    count = 0
    if index is not None:
        count = index.ntotal
    elif collection is not None:
        count = collection.count()
    return jsonify({
        "status": "ok",
        "db_loaded": db_loaded,
        "document_count": count,
        "mode": "ChromaDB" if USE_CHROMADB else "FAISS"
    })

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "السؤال فارغ"}), 400

    # 1. Recherche sémantique
    results = search_database(question, top_k=5)

    if not results:
        return jsonify({
            "answer": "لم أجد معلومات ذات صلة في قاعدة البيانات.",
            "sources": []
        })

    # 2. Générer la réponse avec OpenAI
    try:
        if OPENAI_API_KEY and OPENAI_API_KEY != "sk-...":
            answer = ask_openai(question, results)
        else:
            # Mode sans OpenAI: retourner les résultats bruts
            answer = "⚠️ مفتاح OpenAI غير مُعيَّن. إليك النصوص ذات الصلة مباشرةً:\n\n"
            for r in results[:3]:
                answer += f"({r['meta'].get('book','?')} | ص {r['meta'].get('printed_page','?')}):\n{r['text'][:500]}...\n\n---\n\n"
    except Exception as e:
        return jsonify({"error": f"خطأ في OpenAI: {str(e)}"}), 500

    # 3. Préparer les sources
    sources = []
    seen = set()
    for r in results:
        key = f"{r['meta'].get('book','')}_{r['meta'].get('printed_page','')}"
        if key not in seen:
            seen.add(key)
            sources.append({
                "book": r['meta'].get('book', 'غير معروف'),
                "page": r['meta'].get('printed_page', '—'),
                "scholar": r['meta'].get('scholar', 'ابن باز'),
                "preview": r['text'][:200] + "..." if len(r['text']) > 200 else r['text'],
                "score": round(r.get('score', 0), 3)
            })

    return jsonify({"answer": answer, "sources": sources})


@app.route("/api/search", methods=["POST"])
def search_only():
    """Recherche brute sans LLM"""
    data = request.get_json()
    query = data.get("query", "").strip()
    top_k = min(int(data.get("top_k", 5)), 20)

    if not query:
        return jsonify({"error": "الاستعلام فارغ"}), 400

    results = search_database(query, top_k=top_k)
    return jsonify({"results": [
        {
            "text": r['text'],
            "book": r['meta'].get('book', '?'),
            "page": r['meta'].get('printed_page', '?'),
            "score": round(r.get('score', 0), 3)
        }
        for r in results
    ]})


if __name__ == "__main__":
    load_database()
    print("\n🚀 Serveur démarré sur http://localhost:5000\n")

    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)