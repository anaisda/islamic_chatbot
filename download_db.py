import os
import gdown
import shutil

CHROMA_PATH = "./ChromaDB_export"
FOLDER_ID = os.environ.get("GDRIVE_FOLDER_ID", "")

def download_if_needed():
    if os.path.exists(CHROMA_PATH) and os.path.exists(f"{CHROMA_PATH}/chroma.sqlite3"):
        size = os.path.getsize(f"{CHROMA_PATH}/chroma.sqlite3")
        if size > 1000:
            print(f"✅ ChromaDB déjà présent ({size/1e6:.1f} MB)")
            return

    if not FOLDER_ID:
        print("⚠️  GDRIVE_FOLDER_ID non configuré — mode démo")
        return

    print("📥 Téléchargement ChromaDB depuis Google Drive...")
    os.makedirs(CHROMA_PATH, exist_ok=True)

    url = f"https://drive.google.com/drive/folders/{FOLDER_ID}"
    gdown.download_folder(url, output=CHROMA_PATH, quiet=False, use_cookies=False)
    print("✅ ChromaDB téléchargé !")
