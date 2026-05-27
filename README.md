# Euro Macro Backtester

**Un framework de backtesting conditionné par les régimes macroéconomiques de la zone euro.**

![Interface du Euro Macro Backtester](assets/screenshot.gif)

La plupart des backtesters répondent à la question : *"Est-ce que cette stratégie est rentable ?"*
Celui-ci répond à une question plus précise : *"Dans quel contexte macroéconomique cette stratégie fonctionne-t-elle — et est-ce que ça tient hors-échantillon ?"*

---

## Concept

Le projet classe chaque mois historique depuis 2004 en un **régime macroéconomique** à partir de trois indicateurs zone euro :

| Dimension | Source | Indicateur |
|---|---|---|
| Inflation | Eurostat `prc_hicp_midx` | HICP YoY — seuil 3% |
| Courbe des taux | FRED `IRLTLT01EZM156N` / `IR3TIB01EZM156N` | Spread 10Y–3M — seuil 0% |
| Croissance | Eurostat `namq_10_gdp` | PIB réel YoY — seuil 0.5% |

La combinaison de ces trois signaux produit **8 régimes étiquetés** — *Expansion saine*, *Surchauffe*, *Stagflation*, *Récession-désinflation*, etc. — qui servent de contexte pour évaluer 5 stratégies systématiques.

L'idée centrale : une stratégie ne se juge pas en absolu, mais par rapport au contexte macroéconomique dans lequel elle a opéré. C'est la logique qu'utilisent les équipes macro-quant des hedge funds.

---

## Résultats clés

> Les chiffres ci-dessous sont obtenus sur l'Euro Stoxx 50 (2004–2026), avec des coûts de transaction de 5 bps.

### Heatmap Sharpe par stratégie et par régime

| Stratégie | Expansion saine | Désinflation douce | Surchauffe | Stagflation |
|---|---|---|---|---|
| Mean reversion RSI14 | +0.47 | +0.41 | +0.52 | −1.07 |
| Macro tilt | +0.63 | +0.18 | −0.21 | −0.44 |
| Vol target 10% | +0.12 | +0.09 | −0.18 | −0.38 |
| Momentum 12M | −0.21 | +0.08 | −0.37 | — |
| SMA 50/200 | −0.08 | +0.11 | −0.29 | — |

→ Le mean reversion domine en régimes d'expansion mais s'effondre en stagflation.
→ Le macro tilt offre le meilleur profil en expansion saine, avec un drawdown maîtrisé.
→ Aucune stratégie n'est universellement supérieure — le régime prime.

### Validation walk-forward (14 fenêtres annuelles, 2012–2026)

Le walk-forward teste si les résultats tiennent sur des données que le modèle n'a jamais vues :
on calibre sur le passé, on teste sur l'année suivante, on avance d'un an et on recommence.

| Critère | Résultat | Seuil |
|---|---|---|
| Fenêtres OOS positives | 73% | ≥ 65% |
| Dégradation IS→OOS | +0.374 | > −0.20 |
| Stabilité de sélection | 100% | ≥ 70% des fenêtres |

---

## Architecture

```
euro-macro-backtester/
│
├── regime_detector.py   # Détection des régimes macro (eurostat + FRED)
├── strategy_zoo.py      # 5 stratégies systématiques
├── backtester.py        # Moteur de backtest avec décomposition par régime
├── metrics.py           # Agrégation multi-stratégies, heatmap, corrélation
├── walk_forward.py      # Validation out-of-sample (expanding window)
│
├── app.py               # Interface Streamlit (3 onglets)
│
├── .env                 # Clé API FRED (non committé)
├── .gitignore
└── requirements.txt
```

### Pipeline de données

```
Eurostat (prc_hicp_midx)  ──┐
Eurostat (namq_10_gdp)    ──┼──▶  regime_detector.py  ──▶  8 régimes mensuels
FRED (taux 10Y, 3M)       ──┘                                      │
                                                                    │
yfinance (prix actifs)    ──────▶  strategy_zoo.py    ──▶  signaux journaliers
                                                                    │
                                                                    ▼
                                   backtester.py       ──▶  perf. par régime
                                                                    │
                                   walk_forward.py     ──▶  validation OOS
```

### Les 5 stratégies

| Stratégie | Logique | Signal |
|---|---|---|
| **Momentum 12M** | Long si la performance sur 252 jours est positive | {0, 1} |
| **SMA 50/200** | Long si la moyenne mobile 50j > moyenne mobile 200j | {0, 1} |
| **Mean reversion RSI14** | Long si RSI < 35 (survendu), cash si RSI > 65 | {0, 1} |
| **Vol target 10%** | Exposition ajustée pour cibler 10% de vol annualisée | [0, 1] |
| **Macro tilt** | Long uniquement en régimes d'expansion, cash sinon | {0, 1} |

---

## Installation

### Prérequis

- Python 3.11+
- Une clé API FRED gratuite : [fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html)

### Étapes

```bash
# 1. Cloner le dépôt
git clone https://github.com/<votre-username>/euro-macro-backtester.git
cd euro-macro-backtester

# 2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate.bat       # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Configurer la clé FRED
echo "FRED_API_KEY=votre_clé_ici" > .env

# 5. Lancer l'application
streamlit run app.py
```

L'interface s'ouvre sur `http://localhost:8501`.

> **Note** : le premier chargement télécharge les données macro depuis Eurostat et FRED (~15 s). Les appels suivants sont mis en cache par Streamlit.

---

## Utilisation

### Onglet Vue globale

Point d'entrée de l'application. Paramétrez les seuils de régime et les coûts de transaction, puis explorez :

- **Tableau de synthèse** : meilleure stratégie par indice vs buy-and-hold
- **Heatmap multi-indices** : Sharpe de la meilleure stratégie par régime pour chaque indice
- **Recommandations** : quelle stratégie et quel indice privilégier dans le régime actuel
- **Timeline** : historique des régimes depuis 2004

### Onglet Analyse par actif

Analyse détaillée pour un actif sélectionné :

- Heatmap stratégie × régime (Sharpe, rendement, drawdown)
- Equity curves avec bandes de régime en arrière-plan
- **Matrice de corrélation** entre stratégies — identifie les paires complémentaires

### Onglet Walk-Forward

Validation de robustesse out-of-sample :

- Sélection de la fenêtre de calibration et de test
- Comparaison Sharpe IS vs OOS par stratégie
- Equity curve OOS continue (2012–2026)
- **Verdict automatique** : Robuste / Partiellement robuste / Fragile

---

## Sources de données

| Indicateur | Source | Série | Fréquence |
|---|---|---|---|
| HICP zone euro | Eurostat | `prc_hicp_midx` | Mensuelle |
| PIB réel zone euro | Eurostat | `namq_10_gdp` | Trimestrielle → mensuelle |
| Taux souverain 10 ans | FRED | `IRLTLT01EZM156N` | Mensuelle |
| EURIBOR 3 mois | FRED | `IR3TIB01EZM156N` | Mensuelle |
| Prix des actifs | yfinance | `^STOXX50E`, `^GDAXI`, `^FCHI`, `^FTSE`, `^STOXX` | Journalière |

Toutes les sources sont publiques et gratuites.

---

## Limites connues

**Intervalles de confiance** — les ratios de Sharpe affichés ne sont pas accompagnés d'intervalles de confiance. Sur des fenêtres courtes (< 3 ans), l'estimation du Sharpe est intrinsèquement bruitée.

**Seuils de régime arbitraires** — les seuils (inflation > 3%, courbe > 0%, croissance > 0.5%) sont économiquement motivés mais non optimisés. Une approche par clustering (K-means, HMM) serait plus rigoureuse.

**Univers limité** — cinq indices actions européens. L'extension à d'autres classes d'actifs (obligations, matières premières, FX) renforcerait la généralité des conclusions.

**Coûts de transaction simplifiés** — les coûts sont modélisés comme un pourcentage fixe. L'impact de marché et le bid-ask spread variable selon le régime de volatilité ne sont pas pris en compte.

---

## Stack technique

`Python 3.11` · `pandas` · `numpy` · `yfinance` · `fredapi` · `eurostat` · `Streamlit` · `Plotly` · `python-dateutil`

---

*Projet personnel*
