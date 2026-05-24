"""Baixa o dataset oficial da Fase 3 a partir do Google Drive.

Link oficial fornecido no enunciado do desafio:
    https://drive.google.com/drive/folders/1aS7exW5N0qq1uIxvIBcAfc18OHojOMjj

Requer:
    pip install gdown

Uso:
    python scripts/download_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402

DRIVE_FOLDER_URL = (
    "https://drive.google.com/drive/folders/1aS7exW5N0qq1uIxvIBcAfc18OHojOMjj"
)
EXPECTED_FILES = ["flights.csv", "airlines.csv", "airports.csv"]


def already_downloaded() -> bool:
    return all((config.DATA_DIR / f).exists() for f in EXPECTED_FILES)


def download() -> None:
    if already_downloaded():
        print("✓ Todos os arquivos já estão em data/. Nada a fazer.")
        return

    try:
        import gdown
    except ImportError:
        sys.exit(
            "Erro: pacote `gdown` não instalado. Rode:\n"
            "    pip install gdown\n"
            "ou:\n"
            "    pip install -r requirements.txt"
        )

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"→ Baixando dataset do Google Drive para {config.DATA_DIR}…")
    print(f"  Fonte: {DRIVE_FOLDER_URL}")

    gdown.download_folder(
        url=DRIVE_FOLDER_URL,
        output=str(config.DATA_DIR),
        quiet=False,
        use_cookies=False,
    )

    missing = [f for f in EXPECTED_FILES if not (config.DATA_DIR / f).exists()]
    if missing:
        sys.exit(
            f"Erro: arquivos esperados não encontrados após download: {missing}\n"
            f"Verifique o conteúdo da pasta {config.DATA_DIR} e o link do Drive."
        )
    print("✓ Download concluído.")


if __name__ == "__main__":
    download()
