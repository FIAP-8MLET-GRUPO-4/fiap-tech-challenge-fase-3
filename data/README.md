# Dados — Tech Challenge Fase 3

Esta pasta deve conter o dataset **2015 Flight Delays and Cancellations** (Kaggle / U.S. DOT), composto por 3 arquivos:

| Arquivo | Conteúdo | Tamanho |
| :--- | :--- | ---: |
| `flights.csv` | ~5.8M voos comerciais nos EUA em 2015 | ~580 MB |
| `airlines.csv` | Companhias aéreas (código → nome) | ~1 KB |
| `airports.csv` | Aeroportos (IATA, cidade, estado, lat/lon) | ~30 KB |

Os arquivos **não são versionados no Git** (ver `.gitignore`).

## Como baixar

### Opção 1 — Script automático (recomendado)

Requer [Kaggle CLI](https://github.com/Kaggle/kaggle-api) configurado com `~/.kaggle/kaggle.json`:

```bash
python scripts/download_data.py
```

O script baixa e descompacta os três CSVs nesta pasta.

### Opção 2 — Manual

1. Acesse: https://www.kaggle.com/datasets/usdot/flight-delays
2. Faça login e clique em **Download**
3. Descompacte o `.zip` e mova `flights.csv`, `airlines.csv` e `airports.csv` para esta pasta (`data/`)

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
