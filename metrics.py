"""
metrics.py
==========
Agrège les résultats de plusieurs backtests et prépare les données
pour l'interface Streamlit.

Fonctions principales :
  run_all_strategies()   → lance tous les backtests, retourne un dict de résultats
  build_heatmap_df()     → pivot stratégie × régime → Sharpe  (input du heatmap)
  build_comparison_df()  → métriques globales par stratégie    (input du tableau)
  get_equity_curves()    → equity curves de toutes les stratégies alignées
"""

import numpy as np
import pandas as pd
import yfinance as yf

from backtester import Backtester, BacktestResult
from strategy_zoo import (
    momentum,
    sma_crossover,
    mean_reversion,
    macro_tilt,
    vol_target,
)


# ─── Actifs disponibles dans l'interface ──────────────────────────────────────

ASSETS = {
    "Euro Stoxx 50":    "^STOXX50E",
    "DAX (Allemagne)":  "^GDAXI",
    "CAC 40 (France)":  "^FCHI",
    "FTSE 100 (UK)":    "^FTSE",
    "STOXX Europe 600": "^STOXX",
}


# ─── Chargement des prix ──────────────────────────────────────────────────────

def load_prices(ticker: str, start: str = "2003-01-01") -> pd.Series:
    """Télécharge les prix de clôture via yfinance."""
    raw = yf.download(ticker, start=start, progress=False)
    if raw.empty:
        raise ValueError(
            f"Aucune donnée reçue pour le ticker '{ticker}'. "
            "Vérifie le symbole sur finance.yahoo.com."
        )
    prices = raw["Close"].squeeze()
    prices.name = ticker
    return prices.dropna()


# ─── Lancement de tous les backtests ──────────────────────────────────────────

def run_all_strategies(
    prices: pd.Series,
    regime_df: pd.DataFrame,
    tc_bps: float = 5.0,
) -> dict[str, BacktestResult]:
    """
    Lance les 5 stratégies sur l'actif fourni et retourne un dictionnaire
    {nom_stratégie: BacktestResult}.

    Paramètres
    ----------
    prices    : pd.Series      Prix de clôture journaliers.
    regime_df : pd.DataFrame   Sortie de RegimeDetector.detect().
    tc_bps    : float          Coûts de transaction en bps. Défaut : 5.
    """
    strategies = {
        "Momentum 12M":         (momentum,       {"window": 252}),
        "SMA 50/200":           (sma_crossover,  {"short_window": 50, "long_window": 200}),
        "Mean reversion RSI14": (mean_reversion, {"rsi_window": 14}),
        "Vol target 10%":       (vol_target,     {"target_vol": 0.10}),
        "Macro tilt":           (macro_tilt,     {"regime_series": regime_df["regime"]}),
    }

    results: dict[str, BacktestResult] = {}
    for name, (fn, params) in strategies.items():
        bt = Backtester(
            prices=prices,
            regime_df=regime_df,
            strategy_fn=fn,
            strategy_params=params,
            strategy_name=name,
            transaction_cost_bps=tc_bps,
        )
        results[name] = bt.run()

    return results


# ─── Heatmap stratégie × régime ───────────────────────────────────────────────

def build_heatmap_df(
    results: dict[str, BacktestResult],
    metric: str = "sharpe",
) -> pd.DataFrame:
    """
    Construit le pivot stratégie × régime pour la heatmap.

    Paramètres
    ----------
    results : dict de BacktestResult
    metric  : colonne à utiliser comme valeur ('sharpe', 'rendement_ann', 'max_drawdown')

    Retourne un DataFrame avec :
      - index   : noms des stratégies
      - colonnes : noms des régimes
      - valeurs  : la métrique choisie (NaN si régime absent dans la période)
    """
    rows = []
    for strategy_name, result in results.items():
        regime_metrics = result.by_regime().reset_index()
        for _, row in regime_metrics.iterrows():
            rows.append({
                "strategy": strategy_name,
                "regime":   row["Régime"],
                metric:     row[metric],
                "n_mois":   row["n_mois_approx"],
            })

    if not rows:
        return pd.DataFrame()

    long_df = pd.DataFrame(rows)
    pivot = long_df.pivot(index="strategy", columns="regime", values=metric)

    # Ordre des régimes : du plus favorable au moins favorable (économiquement)
    regime_order = [
        "Expansion saine",
        "Expansion fragile",
        "Désinflation douce",
        "Ralentissement inflationniste",
        "Stagflation naissante",
        "Surchauffe",
        "Stagflation",
        "Récession-désinflation",
    ]
    # Garder uniquement les colonnes présentes, dans l'ordre
    cols = [r for r in regime_order if r in pivot.columns]
    cols += [c for c in pivot.columns if c not in regime_order]
    return pivot[cols]


# ─── Tableau de comparaison globale ───────────────────────────────────────────

def build_comparison_df(results: dict[str, BacktestResult]) -> pd.DataFrame:
    """
    Tableau récapitulatif des métriques globales de chaque stratégie,
    trié par Sharpe décroissant.
    """
    rows = []
    for name, result in results.items():
        m = result.global_metrics()
        rows.append({
            "Stratégie":      name,
            "Sharpe":         m["sharpe"],
            "Rendement ann.": m["rendement_ann"],
            "Vol. ann.":      m["vol_ann"],
            "Max drawdown":   m["max_drawdown"],
            "Investi (%)":    m["pct_investis"],
            "Win rate (%)":   m["win_rate"],
        })

    df = pd.DataFrame(rows).set_index("Stratégie")
    return df.sort_values("Sharpe", ascending=False)


# ─── Equity curves alignées ───────────────────────────────────────────────────

def get_equity_curves(results: dict[str, BacktestResult]) -> pd.DataFrame:
    """
    Retourne un DataFrame avec les equity curves de toutes les stratégies,
    alignées sur le même index temporel, base 100 au premier jour commun.
    """
    curves = {name: res.equity_curve for name, res in results.items()}
    df = pd.DataFrame(curves)

    # Base 100 au premier jour où toutes les stratégies ont des données
    first_valid = df.dropna().index[0]
    df = df.loc[first_valid:]
    df = df / df.iloc[0] * 100
    return df


# ─── Matrice de corrélation des stratégies ────────────────────────────────────

def build_correlation_matrix(results: dict[str, BacktestResult]) -> pd.DataFrame:
    """
    Matrice de corrélation des rendements journaliers entre stratégies.

    Pourquoi c'est utile :
      - Corrélation proche de 1 : les deux stratégies réagissent de façon
        identique — les combiner n'apporte aucune diversification.
      - Corrélation proche de 0 ou négative : elles sont complémentaires.
        Quand l'une perd, l'autre peut gagner. C'est la base de la
        construction de portefeuille multi-stratégies.

    Retourne un DataFrame carré (stratégies × stratégies), valeurs dans [-1, 1].
    """
    returns_df = pd.DataFrame({
        name: result.returns for name, result in results.items()
    }).dropna()
    return returns_df.corr().round(3)
