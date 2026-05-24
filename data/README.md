# Dados — Tech Challenge Fase 3

Esta pasta deve conter o dataset **Flight Delays and Cancellations** (U.S. DOT, 2015), fornecido oficialmente pelo enunciado do desafio via Google Drive:

🔗 **Link oficial:** https://drive.google.com/drive/folders/1aS7exW5N0qq1uIxvIBcAfc18OHojOMjj

O dataset é composto por 3 arquivos:

| Arquivo | Conteúdo | Tamanho |
| :--- | :--- | ---: |
| `flights.csv` | ~5.8M voos comerciais nos EUA em 2015 | ~580 MB |
| `airlines.csv` | Companhias aéreas (código → nome) | ~1 KB |
| `airports.csv` | Aeroportos (IATA, cidade, estado, lat/lon) | ~30 KB |

Os arquivos **não são versionados no Git** (ver `.gitignore`).

## Como baixar

### Opção 1 — Script automático (recomendado)

Requer `gdown` (já incluído em `requirements.txt`):

```bash
pip install -r requirements.txt
python scripts/download_data.py
```

O script baixa a pasta inteira do Google Drive para `data/` e valida que
os três CSVs foram extraídos.

### Opção 2 — Manual

1. Acesse: https://drive.google.com/drive/folders/1aS7exW5N0qq1uIxvIBcAfc18OHojOMjj
2. Baixe os três arquivos (`flights.csv`, `airlines.csv`, `airports.csv`).
3. Mova-os para esta pasta (`data/`).

## Verificação

Após o download a estrutura deve ser:

```
data/
├── README.md
├── airlines.csv
├── airports.csv
└── flights.csv
```

E o `flights.csv` deve ter ~5.8M linhas:

```bash
wc -l data/flights.csv
# 5819080 data/flights.csv
```

## Dicionário de dados

Ver [`docs/dicionario_dados_flights.pdf`](../docs/dicionario_dados_flights.pdf).
