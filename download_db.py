import os
import json
import zipfile
import io
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

CHROMA_PATH = "./ChromaDB_export"
FILE_ID = os.environ.get("GDRIVE_FILE_ID", "")
SERVICE_ACCOUNT_JSON = os.environ.get("GOOGLE_SERVICE_ACCOUNT", "")

def download_if_needed():
    # Vérifie si collection déjà présente
    if os.path.exists(CHROMA_PATH):
        subdirs = [f for f in os.listdir(CHROMA_PATH)
                   if os.path.isdir(os.path.join(CHROMA_PATH, f))]
        if subdirs:
            print(f"✅ ChromaDB déjà présent : {subdirs[0]}")
            return

    if not FILE_ID or not SERVICE_ACCOUNT_JSON:
        print("⚠️  Variables manquantes — mode démo")
        return

    print("📥 Téléchargement depuis Google Drive (Service Account)...")

    # Charger les credentials depuis la variable d'environnement
    service_account_info = json.loads(SERVICE_ACCOUNT_JSON)
    credentials = service_account.Credentials.from_service_account_info(
        service_account_info,
        scopes=["https://www.googleapis.com/auth/drive.readonly"]
    )

    service = build("drive", "v3", credentials=credentials)

    # Télécharger le fichier
    request = service.files().get_media(fileId=FILE_ID)
    zip_buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(zip_buffer, request)

    done = False
    while not done:
        status, done = downloader.next_chunk()
        print(f"  {int(status.progress() * 100)}%")

    # Extraire le ZIP
    print("📦 Extraction...")
    os.makedirs(CHROMA_PATH, exist_ok=True)
    zip_buffer.seek(0)
    with zipfile.ZipFile(zip_buffer, 'r') as z:
        z.extractall(CHROMA_PATH)

    print("✅ ChromaDB extrait et prêt !")
