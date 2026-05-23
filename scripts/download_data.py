"""Baixa o dataset 'Flight Delays and Cancellations' do Kaggle.

Requer:
    pip install kaggle
    ~/.kaggle/kaggle.json com suas credenciais (chmod 600)

Uso:
    python scripts/download_data.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402

DATASET = "usdot/flight-delays"
EXPECTED_FILES = ["flights.csv", "airlines.csv", "airports.csv"]


def already_downloaded() -> bool:
    return all((config.DATA_DIR / f).exists() for f in EXPECTED_FILES)


def download() -> None:
    if already_downloaded():
        print("✓ Todos os arquivos já estão em data/. Nada a fazer.")
        return

    if shutil.which("kaggle") is None:
        sys.exit(
            "Erro: comando `kaggle` não encontrado. Instale com:\n"
            "    pip install kaggle\n"
            "Configure ~/.kaggle/kaggle.json conforme:\n"
            "    https://github.com/Kaggle/kaggle-api#api-credentials"
        )

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"→ Baixando {DATASET} para {config.DATA_DIR}…")
    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            DATASET,
            "-p",
            str(config.DATA_DIR),
            "--force",
        ],
        check=True,
    )

    zip_path = config.DATA_DIR / "flight-delays.zip"
    print(f"→ Descompactando {zip_path}…")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(config.DATA_DIR)
    zip_path.unlink()

    missing = [f for f in EXPECTED_FILES if not (config.DATA_DIR / f).exists()]
    if missing:
        sys.exit(f"Erro: arquivos esperados não encontrados após download: {missing}")
    print("✓ Download concluído.")


if __name__ == "__main__":
    download()
