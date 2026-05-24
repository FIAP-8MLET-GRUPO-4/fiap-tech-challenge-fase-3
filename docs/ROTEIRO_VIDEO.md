# Roteiro do Vídeo — Tech Challenge Fase 3 (máx. 10 min)

> Predição de atrasos em voos comerciais dos EUA (2015) com ML supervisionado
> e não-supervisionado.

---

## Silvio (~2min30s) — Introdução, dataset e EDA

- Apresentar o grupo e o desafio (90% da nota da fase).
- Contextualizar o problema: atrasos aéreos impactam milhões de passageiros;
  queremos **entender padrões** e **prever atrasos**.
- Mostrar o dataset: ~5.8M voos de 2015 do U.S. DOT, 31 colunas
  (rapidamente abrir `docs/dicionario_dados_flights.pdf`).
- Abrir `notebooks/01_eda.ipynb` e mostrar 3-4 visualizações-chave:
  - Distribuição de `ARRIVAL_DELAY` (cauda longa positiva, ~18% atrasam ≥15 min).
  - Heatmap dia da semana × hora do dia (tarde/noite piores).
  - Taxa de atraso por companhia e por mês (jun/jul/dez piores).
  - Contribuição de cada causa de atraso (aircraft tardio + companhia dominam).
- Explicar o tratamento de missing values:
  - `*_DELAY` granulares → `fillna(0)` (NaN ≡ "sem contribuição").
  - Cancelados/desviados → dropados (tarefa diferente).
- Passar para o Doglas.

---

## Doglas (~2min30s) — Feature engineering e classificação

- Abrir `src/features.py` rapidamente — destacar features derivadas:
  período do dia, estação do ano, **feriados federais US 2015**,
  fim de semana, rota (origem→destino).
- Falar do **cuidado anti-leakage**: `config.LEAKY_COLS` exclui colunas
  conhecidas só pós-voo (`DEPARTURE_DELAY`, `TAXI_OUT`, `WHEELS_*`,
  `*_DELAY`). Sem isso, o modelo "trapaceia".
- Abrir `notebooks/02_classification.ipynb`:
  - Tarefa: `IS_DELAYED` (atraso ≥ 15 min na chegada).
  - **Algoritmo 1** — Logistic Regression (`class_weight='balanced'`).
  - **Algoritmo 2** — LightGBM (gradient boosting em árvores).
  - Mostrar tabela comparativa (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC).
  - Matriz de confusão + curva ROC lado a lado.
  - Importância das features (LightGBM): hora de partida, companhia, aeroporto.
  - Tuning de threshold (curva precision×recall) para uso operacional.
- Passar para o Ricardo.

---

## Ricardo (~2min30s) — Regressão da duração do atraso

- Abrir `notebooks/03_regression.ipynb`:
  - Tarefa: prever `ARRIVAL_DELAY` em minutos (regressão contínua).
  - Mostrar a distribuição assimétrica do alvo e a transformação
    `signed log1p` aplicada via `TransformedTargetRegressor`.
  - **Algoritmo 1** — Ridge Regression (linear, regularização L2).
  - **Algoritmo 2** — LightGBM Regressor.
  - Mostrar tabela de métricas (MAE, RMSE, R²).
  - Gráfico **predito × real** + plot de **resíduos** lado a lado.
  - Quebra do MAE por bucket de magnitude — modelo acerta o cenário comum
    e erra mais nos atrasos extremos (>120 min, ~aleatórios sem clima).
- Resumir: LightGBM > Ridge em todas as métricas; sem dados meteorológicos
  previstos, o teto é limitado.
- Passar para a Mariana.

---

## Mariana (~2min30s) — Não-supervisionado + conclusões

- Abrir `notebooks/04_unsupervised.ipynb`:
  - Pergunta-guia: *"É possível agrupar aeroportos com perfis semelhantes?"*
  - Features por aeroporto: volume, taxa de atraso, atraso médio,
    diversidade de companhias/destinos, partição dos motivos.
  - Mostrar a escolha de **K** via elbow + silhouette.
  - **K-Means** (4 clusters) + **PCA 2D** para visualização.
  - Interpretar os clusters: hubs grandes saudáveis vs hubs problemáticos
    vs regionais médios vs sazonais/longa distância. Citar 2-3 exemplos
    (ATL, ORD, EWR).
- Encerrar com conclusões e limitações (slide ou tela):
  - Cumprimos todos os requisitos obrigatórios (EDA + supervisionado com
    ≥2 algoritmos + não-supervisionado + interpretação).
  - Limitações: sem clima previsto, dataset estático de 2015, atrasos
    extremos imprevisíveis.
  - Próximos passos: integração de clima, modelagem two-stage
    (P(atraso) × E(magnitude|atraso)), calibração de probabilidades,
    dashboard interativo.
- Agradecer e encerrar.

---

## Resumo

| Apresentador | Seções | Notebooks | Tempo |
| :--- | :--- | :---: | :---: |
| **Silvio** | Introdução + dataset + EDA | 01 | ~2min30s |
| **Doglas** | Feature engineering + classificação | 02 | ~2min30s |
| **Ricardo** | Regressão da duração | 03 | ~2min30s |
| **Mariana** | Clusterização + conclusões | 04 | ~2min30s |
| | | **Total** | **~10min** |

---

## Checklist antes de gravar

- [ ] `flights.csv`, `airlines.csv`, `airports.csv` baixados em `data/`
- [ ] `jupyter lab` rodando com os 4 notebooks **já executados** (células
      com outputs visíveis — evita rodar ao vivo, que demora minutos)
- [ ] `docs/dicionario_dados_flights.pdf` aberto numa aba
- [ ] Editor com `src/features.py` e `src/config.py` abertos (para mostrar
      LEAKY_COLS e features derivadas)
- [ ] Figuras em `reports/figures/` revisadas
- [ ] Roteiro em outra tela / impresso
- [ ] Áudio de todos testado, screen share configurado
- [ ] Link do repositório GitHub à mão para mostrar no encerramento
