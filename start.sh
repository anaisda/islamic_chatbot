#!/bin/bash
# ══════════════════════════════════════════════
# Script de démarrage — Islamic Chatbot
# ══════════════════════════════════════════════

echo ""
echo "📚 Islamic Library Chatbot"
echo "══════════════════════════"

# Vérifier si .env existe
if [ ! -f ".env" ]; then
    echo "⚠️  Fichier .env introuvable. Copie .env.example en .env et configure tes clés."
    cp .env.example .env
    echo "✅ .env créé — ouvre-le et ajoute ta clé OpenAI et le chemin FAISS"
    exit 1
fi

# Charger le .env
export $(grep -v '^#' .env | xargs)

# Vérifier la clé OpenAI
if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" = "sk-..." ]; then
    echo "❌ OPENAI_API_KEY non configurée dans .env"
    exit 1
fi

# Vérifier l'index FAISS
if [ ! -f "$FAISS_INDEX_PATH" ]; then
    echo "⚠️  Fichier FAISS non trouvé : $FAISS_INDEX_PATH"
    echo "   → Télécharge islamic_faiss_index.pkl depuis Google Drive"
    echo "   → Place-le dans ce dossier ou configure FAISS_INDEX_PATH dans .env"
    echo ""
    echo "   Mode démo activé (réponses sans base réelle)"
fi

# Installer les dépendances si nécessaire
if ! python -c "import flask" 2>/dev/null; then
    echo "📦 Installation des dépendances..."
    pip install -r requirements.txt -q
fi

echo ""
echo "🚀 Démarrage du serveur..."
echo "   → http://localhost:5000"
echo ""

python app.py
