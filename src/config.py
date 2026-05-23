"""Configurações globais do projeto.

Lê variáveis de ambiente do `.env` (se existir) e expõe constantes usadas
por todos os módulos: caminhos de dados, seed, tamanho de amostra etc.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", PROJECT_ROOT / "models_artifacts"))
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", PROJECT_ROOT / "reports"))
FIGURES_DIR = REPORTS_DIR / "figures"

RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", "0"))  # 0 = dataset completo

FLIGHTS_CSV = DATA_DIR / "flights.csv"
AIRLINES_CSV = DATA_DIR / "airlines.csv"
AIRPORTS_CSV = DATA_DIR / "airports.csv"

DELAY_THRESHOLD_MIN = 15
"""Limiar (em minutos) usado pelo U.S. DOT para considerar um voo atrasado."""

DELAY_REASON_COLS = [
    "AIR_SYSTEM_DELAY",
    "SECURITY_DELAY",
    "AIRLINE_DELAY",
    "LATE_AIRCRAFT_DELAY",
    "WEATHER_DELAY",
]

LEAKY_COLS = [
    "DEPARTURE_TIME",
    "DEPARTURE_DELAY",
    "TAXI_OUT",
    "WHEELS_OFF",
    "ELAPSED_TIME",
    "AIR_TIME",
    "WHEELS_ON",
    "TAXI_IN",
    "ARRIVAL_TIME",
    "ARRIVAL_DELAY",
    "DIVERTED",
    "CANCELLED",
    "CANCELLATION_REASON",
    *DELAY_REASON_COLS,
]
"""Colunas que são conhecidas APENAS após o voo — nunca devem entrar no X de treino."""


for d in (DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)
