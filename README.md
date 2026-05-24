# Tech Challenge - Fase 3

Pipeline completo de Machine Learning sobre o dataset público **2015 Flight Delays and Cancellations** (U.S. DOT), aplicando aprendizado supervisionado (classificação e regressão) e não-supervisionado (clusterização + PCA) para entender e prever atrasos em voos comerciais nos EUA.

## Integrantes do Grupo

| Nome | E-mail | Perfil Google Skills |
| :--- | :--- | :---: |
| **Doglas Parise** | [doglasparise@gmail.com](mailto:doglasparise@gmail.com) | [🔗 Skills Google](https://www.skills.google/public_profiles/c73ebebd-15ad-4883-97f3-02551573d9b9) |
| **Mariana Teixeira Dornelles Parise** | [m.dornelles19@gmail.com](mailto:m.dornelles19@gmail.com) | [🔗 Skills Google](https://www.skills.google/public_profiles/c71a2add-704b-450f-9eba-2ebb17f39191) |
| **Ricardo Gomes de Souza** | [ricardo_g_souza@yahoo.com](mailto:ricardo_g_souza@yahoo.com) | - |
| **Silvio José Meirelles** | [professorsilviomeireles@gmail.com](mailto:professorsilviomeireles@gmail.com) | - |

## Vídeo de apresentação

[Link para vídeo de apresentação](#) <!-- preencher quando gravado -->

## O problema

Atrasos de voos impactam milhões de passageiros todos os anos. Neste projeto usamos o dataset público de voos comerciais dos EUA em 2015 (~5.8M registros) para:

1. **Explorar** os dados em busca de padrões temporais, geográficos e operacionais (EDA).
2. **Prever** se um voo vai atrasar ≥ 15 min (classificação binária).
3. **Prever** quantos minutos um voo vai atrasar (regressão).
4. **Agrupar** aeroportos por perfil operacional (clusterização + PCA).

## Dataset

- **Fonte oficial (link do enunciado):** [📁 Google Drive — Base de dados MLET Fase 3](https://drive.google.com/drive/folders/1aS7exW5N0qq1uIxvIBcAfc18OHojOMjj)
- **Origem dos dados:** U.S. Department of Transportation — Flight Delays and Cancellations (2015)
- **Volume:** ~5.8M voos, 31 colunas
- **Período:** 2015 (ano inteiro)
- **Arquivos:** `flights.csv` (~580 MB), `airlines.csv`, `airports.csv`
- **Dicionário de dados:** [`docs/dicionario_dados_flights.pdf`](docs/dicionario_dados_flights.pdf)

> ⚠️ Os arquivos CSV **não são versionados** (ver `.gitignore`). Baixe com `python scripts/download_data.py` (usa `gdown` para pegar a pasta inteira do Drive) ou siga as instruções manuais em [`data/README.md`](data/README.md).

## Arquitetura do projeto

```
flights.csv ──► load_flights ──► clean_pipeline ──► feature_pipeline ──┬──► classification (LogReg, LightGBM)
                                                                       ├──► regression (Ridge, LightGBM)
                                                                       └──► clustering (K-Means + PCA)
```

| Etapa | Módulo | Responsabilidade |
| :--- | :--- | :--- |
| Carregamento | [`src/data_loader.py`](src/data_loader.py) | dtypes otimizados (~700 MB vs ~3 GB) |
| Limpeza | [`src/preprocessing.py`](src/preprocessing.py) | drop cancelados/desviados, fillna semântico |
| Feature engineering | [`src/features.py`](src/features.py) | período do dia, estação, feriados, rota |
| Classificação | [`src/models/classification.py`](src/models/classification.py) | LogisticRegression vs LightGBM |
| Regressão | [`src/models/regression.py`](src/models/regression.py) | Ridge vs LightGBM (alvo em log1p) |
| Clusterização | [`src/models/clustering.py`](src/models/clustering.py) | K-Means + PCA sobre aeroportos |
| Avaliação | [`src/evaluation.py`](src/evaluation.py) | métricas e gráficos |

## Pré-requisitos

- Python 3.11+
- ~8 GB de RAM disponíveis (para usar o dataset completo de 5.8M linhas)
- Acesso ao [Google Drive do desafio](https://drive.google.com/drive/folders/1aS7exW5N0qq1uIxvIBcAfc18OHojOMjj) (o script de download usa `gdown`, instalado via `requirements.txt`)

## Configuração

1. Clone o repositório:
```bash
git clone <url-do-repositorio>
cd fiap-tech-challenge-fase-3
```

2. Crie o ambiente virtual e instale as dependências:
```bash
python -m venv venv
source venv/bin/activate    # Linux/macOS
# ou: venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

3. Copie o template de variáveis de ambiente:
```bash
cp .env.example .env
```

4. Baixe o dataset (Google Drive oficial do desafio):
```bash
# Opção A — automático (usa gdown, já incluído em requirements.txt)
python scripts/download_data.py

# Opção B — manual: ver data/README.md
```

## Execução

### Notebooks (recomendado para apresentação)

```bash
jupyter lab
```

Abra os notebooks na ordem:

| # | Notebook | Conteúdo |
| :--- | :--- | :--- |
| 01 | [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) | Estatísticas descritivas, valores ausentes, ~7 visualizações |
| 02 | [`notebooks/02_classification.ipynb`](notebooks/02_classification.ipynb) | Predição binária `IS_DELAYED` (LogReg vs LightGBM) |
| 03 | [`notebooks/03_regression.ipynb`](notebooks/03_regression.ipynb) | Predição da duração do atraso (Ridge vs LightGBM) |
| 04 | [`notebooks/04_unsupervised.ipynb`](notebooks/04_unsupervised.ipynb) | K-Means + PCA sobre aeroportos |

### Pipeline headless (CLI)

Para reproduzir todos os experimentos via linha de comando:

```bash
python scripts/run_pipeline.py --task all
# ou tarefa específica:
python scripts/run_pipeline.py --task classification
python scripts/run_pipeline.py --task regression
python scripts/run_pipeline.py --task clustering
```

Saídas geradas:
- `models_artifacts/*.joblib` — pipelines treinados (preprocess + modelo)
- `reports/metrics_*.csv` — métricas por modelo
- `reports/figures/*.png` — visualizações
- `reports/summary.json` — resumo do run

## Decisões importantes

### Tratamento de valores ausentes

| Coluna(s) | Tratamento | Justificativa |
| :--- | :--- | :--- |
| `*_DELAY` granulares (5 colunas) | `fillna(0)` | NaN ≡ "sem contribuição" — só preenchido quando o voo atrasa ≥ 15 min |
| `TAIL_NUMBER` | preencher com `"UNKNOWN"` | ~0.25% ausente, manter linha |
| `CANCELLATION_REASON` | drop | só existe para cancelados (tarefa fora do escopo) |
| `ARRIVAL_DELAY`, `DEPARTURE_TIME`, etc. | drop linha | ~1.5% — voos cancelados/desviados |

### Evitando vazamento (leakage)

Variáveis listadas em `src.config.LEAKY_COLS` (`DEPARTURE_DELAY`, `TAXI_OUT`, `WHEELS_*`, `ARRIVAL_TIME`, `*_DELAY` granulares etc.) são conhecidas **somente após o voo** — e portanto **não entram** no `X` dos modelos preditivos. Predições devem ser viáveis no momento do agendamento.

### Features derivadas

- **Tempo:** `DEP_HOUR` (0–23), `DEP_PERIOD` (madrugada/manhã/tarde/noite), `SEASON`.
- **Calendário:** `IS_WEEKEND`, `IS_HOLIDAY` (feriados federais dos EUA em 2015).
- **Rota:** `ROUTE = ORIGIN-DESTINATION`.

### Modelos supervisionados

| Tarefa | Algoritmo 1 (baseline) | Algoritmo 2 (forte) | Métricas |
| :--- | :--- | :--- | :--- |
| Classificação | Logistic Regression (balanced) | LightGBM Classifier | Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC |
| Regressão | Ridge (alvo em log1p) | LightGBM Regressor | MAE, RMSE, R² |

### Modelo não-supervisionado

- **Unidade:** aeroporto de origem (não voo individual).
- **Features:** volume, taxa de atraso, atraso médio, diversidade de companhias/destinos, partição dos motivos de atraso.
- **K** escolhido via silhouette score.
- **PCA** projeta para 2D para visualização.

## Resultados principais

> Valores indicativos com base em experimentos preliminares; os notebooks reportam os valores finais após execução com o dataset completo.

### Classificação (`IS_DELAYED`)

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | ~0.62 | ~0.30 | ~0.62 | ~0.40 | ~0.66 |
| LightGBM | ~0.65 | ~0.34 | ~0.66 | ~0.45 | ~0.71 |

### Regressão (`ARRIVAL_DELAY` em min)

| Modelo | MAE | RMSE | R² |
| :--- | ---: | ---: | ---: |
| Ridge | ~17.0 | ~38.0 | ~0.04 |
| LightGBM | ~14.5 | ~34.5 | ~0.10 |

### Clusterização de aeroportos

- ~4 clusters emergem como ótimos por silhouette.
- Tipologia interpretável: *hubs grandes saudáveis* vs *hubs grandes problemáticos* vs *regionais médios* vs *aeroportos sazonais/longa distância*.

## Limitações

1. **Sem dados meteorológicos previstos** — só atraso meteorológico observado (leaky). Incorporar previsões reais ampliaria muito o sinal.
2. **Dataset estático de 2015** — não captura mudanças estruturais (pós-COVID, novas rotas).
3. **Atrasos extremos (>120 min)** são raros e quase aleatórios sem informação operacional em tempo real.
4. **K-Means assume clusters esféricos** — features assimétricas podem distorcer a separação.

## Próximos passos sugeridos

- Adicionar dados de clima histórico por aeroporto/data (NOAA, METAR).
- Modelagem **two-stage**: P(atraso) × E(magnitude | atraso).
- **Quantile regression** para intervalos de confiança (P50/P90).
- Calibração de probabilidades (Platt / isotônica).
- Detectar **anomalias** (Isolation Forest) em voos com atraso inesperado.
- Dashboard interativo (Streamlit/Plotly) com mapa geográfico dos clusters.

## Estrutura do Projeto

```
.
├── README.md
├── requirements.txt
├── .env.example
├── data/
│   └── README.md                       # instruções para baixar o dataset
├── docs/
│   ├── dicionario_dados_flights.pdf
│   └── Tech Challenge Fase 3 - Machine Learning Engineering.pdf
├── src/
│   ├── __init__.py
│   ├── config.py                       # caminhos, seed, constantes
│   ├── data_loader.py                  # leitura com dtypes otimizados
│   ├── preprocessing.py                # limpeza e missing values
│   ├── features.py                     # feature engineering
│   ├── evaluation.py                   # métricas e plots
│   ├── visualization.py                # gráficos reutilizáveis (EDA)
│   └── models/
│       ├── __init__.py
│       ├── classification.py           # pipelines de classificação
│       ├── regression.py               # pipelines de regressão
│       └── clustering.py               # K-Means + PCA sobre aeroportos
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_classification.ipynb
│   ├── 03_regression.ipynb
│   └── 04_unsupervised.ipynb
├── scripts/
│   ├── download_data.py                # baixa do Google Drive (gdown)
│   └── run_pipeline.py                 # roda tudo em CLI
├── models_artifacts/                   # joblib dos modelos (gitignored)
└── reports/                            # métricas + figuras (gitignored)
    └── figures/
```

## Tecnologias

- **Python 3.11+**
- **pandas / NumPy / SciPy** — manipulação de dados
- **scikit-learn** — pipelines, métricas, K-Means, PCA, Ridge, Logistic Regression
- **LightGBM** — gradient boosting (rápido em CPU, suporte nativo a categóricas)
- **XGBoost** — comparação adicional (opcional)
- **matplotlib / seaborn / plotly** — visualização
- **Jupyter** — notebooks
- **joblib** — serialização de modelos
- **gdown** — download do dataset oficial via Google Drive

## Requisitos atendidos

| # | Requisito do desafio | Status | Onde |
| :--- | :--- | :---: | :--- |
| 1 | EDA com estatísticas descritivas | ✅ | `notebooks/01_eda.ipynb` |
| 2 | Visualizações com insights | ✅ | `notebooks/01_eda.ipynb` + `reports/figures/` |
| 3 | Tratamento de valores ausentes | ✅ | `src/preprocessing.py` |
| 4 | Classificação (atrasar ou não) | ✅ | `notebooks/02_classification.ipynb` |
| 5 | Regressão (duração do atraso) | ✅ | `notebooks/03_regression.ipynb` |
| 6 | Comparação de ≥2 algoritmos | ✅ | LogReg vs LightGBM / Ridge vs LightGBM |
| 7 | Não-supervisionado (clusterização e PCA) | ✅ | `notebooks/04_unsupervised.ipynb` |
| 8 | Interpretação dos resultados | ✅ | discussão ao final de cada notebook |
| 9 | Discussão de limitações | ✅ | seção "Limitações" deste README + notebooks |
| 10 | Propostas de próximos passos | ✅ | seção "Próximos passos" deste README + notebooks |

### Itens "vá além" implementados

- ✅ Variáveis derivadas: período do dia, estação, feriados, fim de semana, rota.
- ✅ Análise de atrasos por aeroporto, companhia, mês, hora.
- ✅ Padrões sazonais (mês × estação) e horários críticos (heatmap dia × hora).
- ✅ Tuning de threshold (precision × recall) para uso operacional.
- ✅ Importância de features (LightGBM).
- ⏳ Mapas geográficos de rotas e atrasos (sugerido como próximo passo).
- ⏳ Dashboard interativo (sugerido como próximo passo).
