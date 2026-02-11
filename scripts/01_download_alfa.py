from pathlib import Path
import zipfile
import os

# 1) Configuração de caminhos baseada no seu upload
DATA_DIR = Path("data")
ZIP_NAME = "processed-20260211T020118Z-1-001.zip"
ZIP_PATH = DATA_DIR / ZIP_NAME

# Onde os dados serão descompactados
EXTRACT_DIR = DATA_DIR / "alfa" / "unzipped"

def extrair_dados():
    # Verifica se o arquivo que você subiu está no lugar certo
    if ZIP_PATH.exists():
        print(f"Arquivo {ZIP_NAME} encontrado!")
        EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
        
        print(f"Extraindo arquivos para: {EXTRACT_DIR}...")
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(EXTRACT_DIR)
        print("✅ Concluído! O dataset foi extraído com sucesso.")
    else:
        print(f"❌ Erro: O arquivo {ZIP_NAME} não foi encontrado na pasta data.")
        print(f"Caminho verificado: {ZIP_PATH.absolute()}")

if __name__ == "__main__":
    extrair_dados()