from pathlib import Path
import os
import sys
import zipfile
import requests

# 1) Onde salvar
DATA_DIR = Path("data/alfa")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 2) URL do ZIP do ALFA (Substituído com o link direto)
ALFA_ZIP_URL = "https://www.eecis.udel.edu/~trn/alfa/alfa_dataset.zip"

if not ALFA_ZIP_URL:
    print("\nERRO: Link de download não encontrado.\n")
    sys.exit(1)

zip_path = DATA_DIR / "alfa_dataset.zip"

print(f"Baixando para: {zip_path}")
try:
    with requests.get(ALFA_ZIP_URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = 100 * downloaded / total
                        print(f"\r{pct:5.1f}% ({downloaded/1e6:.1f} MB)", end="")
    print("\nDownload concluído.")
except Exception as e:
    print(f"\nErro durante o download: {e}")
    sys.exit(1)

extract_dir = DATA_DIR / "unzipped"
extract_dir.mkdir(parents=True, exist_ok=True)

print(f"Extraindo em: {extract_dir}")
with zipfile.ZipFile(zip_path, "r") as z:
    z.extractall(extract_dir)

print("OK! Dataset extraído.")
print("Evitar fazer o commit da pasta data/. Ela é grande.")
