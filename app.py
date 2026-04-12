import os
import json
import pickle
import numpy as np
import threading
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
CHROMADB_PATH = os.environ.get("CHROMADB_PATH", "./ChromaDB_export")
GDRIVE_FILE_ID = os.environ.get("GDRIVE_FILE_ID", "")
SERVICE_ACCOUNT_B64 = os.environ.get("GOOGLE_SERVICE_ACCOUNT_B64", "")

collection = None
db_ready = False
db_error = ""

def download_and_load():
    global collection, db_ready, db_error

    try:
        # Essayer de charger directement si déjà présent
        if os.path.exists(CHROMADB_PATH):
            subdirs = [f for f in os.listdir(CHROMADB_PATH)
                      if os.path.isdir(os.path.join(CHROMADB_PATH, f))]
            if subdirs:
                load_chromadb()
                return

        # Télécharger depuis Drive
        if GDRIVE_FILE_ID and SERVICE_ACCOUNT_B64:
            download_from_drive()
            load_chromadb()
        else:
            db_error = "Variables GDRIVE manquantes"

    except Exception as e:
        db_error = str(e)
        print(f"❌ Erreur: {e}")

def download_from_drive():
    import base64, zipfile, io
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload

    print("📥 Téléchargement ChromaDB...")
    json_bytes = base64.b64decode(SERVICE_ACCOUNT_B64)
    info = json.loads(json_bytes)

    creds = service_account.Credentials.from_service_account_info(
        info, scopes=["https://www.googleapis.com/auth/drive.readonly"]
    )
    service = build("drive", "v3", credentials=creds)
    req = service.files().get_media(fileId=GDRIVE_FILE_ID)

    buf = io.BytesIO()
    dl = MediaIoBaseDownload(buf, req, chunksize=10*1024*1024)
    done = False
    while not done:
        status, done = dl.next_chunk()
        print(f"  {int(status.progress()*100)}%")

    os.makedirs(CHROMADB_PATH, exist_ok=True)
    buf.seek(0)
    with zipfile.ZipFile(buf, 'r') as z:
        z.extractall(CHROMADB_PATH)
    print("✅ Téléchargement terminé")

def load_chromadb():
    global collection, db_ready
    import chromadb
    from chromadb.utils import embedding_functions

    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
    client = chromadb.PersistentClient(path=CHROMADB_PATH)
    
    # Lister les collections disponibles
    cols = client.list_collections()
    print(f"Collections disponibles: {[c.name for c in cols]}")
    
    collection = client.get_collection(cols[0].name, embedding_function=emb_fn)
    db_ready = True
    print(f"✅ ChromaDB prêt — {collection.count()} documents")

# Lancer le téléchargement en arrière-plan
thread = threading.Thread(target=download_and_load, daemon=True)
thread.start()

# ─── ROUTES ───

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/api/status")
def status():
    return jsonify({
        "status": "ok",
        "db_loaded": db_ready,
        "document_count": collection.count() if db_ready else 0,
        "error": db_error
    })

@app.route("/api/chat", methods=["POST"])
def chat():
    if not db_ready:
        return jsonify({
            "answer": "⏳ قاعدة البيانات لا تزال تُحمَّل... انتظر دقيقة ثم أعد المحاولة.",
            "sources": []
        })

    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "السؤال فارغ"}), 400

    results = collection.query(query_texts=[question], n_results=5)

    context_parts = []
    sources = []
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        context_parts.append(f"[{meta.get('book','?')} | ص {meta.get('printed_page','?')}]\n{doc}")
        sources.append({
            "book": meta.get('book', '?'),
            "page": meta.get('printed_page', '?'),
            "scholar": meta.get('scholar', 'ابن باز'),
            "preview": doc[:200],
            "score": 1.0
        })

    context = "\n\n---\n\n".join(context_parts)

    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "أنت مساعد إسلامي. اقتبس النص حرفياً من المصادر مع ذكر الكتاب والصفحة. لا تضف شيئاً من عندك."},
            {"role": "user", "content": f"السؤال: {question}\n\nالمصادر:\n{context}"}
        ],
        temperature=0.0
    )

    return jsonify({"answer": response.choices[0].message.content, "sources": sources})

@app.route("/api/search", methods=["POST"])
def search():
    if not db_ready:
        return jsonify({"error": "DB not ready"}), 503
    data = request.get_json()
    results = collection.query(query_texts=[data.get("query","")], n_results=5)
    return jsonify({"results": results['documents'][0]})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
