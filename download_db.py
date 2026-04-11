import os
import gdown
import shutil

CHROMA_PATH = "./ChromaDB_export"
FILE_ID = "https://drive.google.com/file/d/1YIkOsiF-c6MVEHVmkMcxf3pvTOh3NWAJ/view?usp=sharing"

def download_if_needed():
    # Vérifie si la collection existe déjà (dossier avec UUID dedans)
    if os.path.exists(CHROMA_PATH):
        subdirs = [f for f in os.listdir(CHROMA_PATH) 
                   if os.path.isdir(os.path.join(CHROMA_PATH, f))]
        if subdirs:
            print(f"✅ ChromaDB déjà présent avec collection : {subdirs[0]}")
            return

    if not FILE_ID:
        print("⚠️  GDRIVE_FILE_ID non configuré")
        return

    print("📥 Téléchargement du ZIP ChromaDB...")
    zip_path = "./chromadb.zip"
    
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, zip_path, quiet=False)

    print("📦 Extraction du ZIP...")
    os.makedirs(CHROMA_PATH, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(CHROMA_PATH)
    
    os.remove(zip_path)
    print("✅ ChromaDB extrait et prêt !")
