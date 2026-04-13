# 📚 المكتبة الإسلامية — Islamic Library Chatbot

Chatbot RAG pour les fatwas de Cheikh Ibn Baz — répond avec des citations exactes depuis la base de données.

---

## 🗂️ Structure du projet

```
islamic_chatbot/
├── app.py                     ← Serveur Flask (backend API)
├── static/
│   └── index.html             ← Interface web (frontend)
├── requirements.txt
├── .env.example               ← Template de configuration
├── start.sh                   ← Démarrage Linux/Mac
├── start.bat                  ← Démarrage Windows
└── islamic_faiss_index.pkl    ← ⬅️ À PLACER ICI (télécharger depuis Drive)
```

---

## 🚀 Installation et lancement local

### Étape 1 — Télécharger la base de données
Dans Google Colab, télécharge ton fichier FAISS :
```python
from google.colab import files
files.download('/content/drive/MyDrive/islamic_faiss_index.pkl')
```
Place le fichier dans le dossier `islamic_chatbot/`.

### Étape 2 — Configurer les variables
```bash
cp .env.example .env
# Ouvre .env et configure :
# OPENAI_API_KEY=sk-ton-vrai-cle
# FAISS_INDEX_PATH=./islamic_faiss_index.pkl
```

### Étape 3 — Installer les dépendances
```bash
pip install -r requirements.txt
```

### Étape 4 — Lancer
```bash
# Linux / Mac
chmod +x start.sh && ./start.sh

# Windows
start.bat

# Ou directement
python app.py
```

### Étape 5 — Ouvrir dans le navigateur
```
http://localhost:5000
```

---

## ☁️ Déploiement pour le client

### Option A — Railway (le plus simple, gratuit)
1. Créer un compte sur [railway.app](https://railway.app)
2. New Project → Deploy from GitHub
3. Ajouter les variables d'environnement dans Settings → Variables
4. Upload le fichier FAISS via la CLI Railway :
   ```bash
   railway login
   railway up
   ```

### Option B — Render (gratuit avec limitations)
1. Créer un compte sur [render.com](https://render.com)
2. New Web Service → Connect GitHub repo
3. Build Command : `pip install -r requirements.txt`
4. Start Command : `gunicorn app:app`
5. Ajouter les ENV vars dans le dashboard

### Option C — Ton serveur Hetzner existant
```bash
# Sur le serveur
git clone ... /var/www/islamic_chatbot
cd /var/www/islamic_chatbot
pip install -r requirements.txt
cp .env.example .env && nano .env  # configurer les clés

# Uploader le fichier FAISS
scp ./islamic_faiss_index.pkl user@hetzner:/var/www/islamic_chatbot/

# Lancer avec PM2
pm2 start "gunicorn app:app --bind 0.0.0.0:5001" --name islamic-chatbot

# Nginx config
# location /chatbot/ { proxy_pass http://localhost:5001/; }
```

---

## 🔌 API Endpoints

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `GET /api/status` | GET | État de la base de données |
| `POST /api/chat` | POST | Poser une question (avec LLM) |
| `POST /api/search` | POST | Recherche brute sans LLM |

### Exemple d'appel API
```bash
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "ما حكم تارك الصلاة؟"}'
```

---

## ⚙️ Configuration avancée

Dans `app.py`, modifier les variables en haut du fichier :

```python
USE_CHROMADB = False          # True pour ChromaDB, False pour FAISS
FAISS_INDEX_PATH = "./..."    # Chemin vers le fichier FAISS
CHROMADB_PATH = "./..."       # Chemin vers le dossier ChromaDB
```
