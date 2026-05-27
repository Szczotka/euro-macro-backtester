"""
strategy_zoo.py
===============
Collection de stratégies prêtes à être passées au Backtester.

Toutes les fonctions ont la même signature minimale :
    f(prices: pd.Series, **params) -> pd.Series

Les signaux sont dans [-1, 1] :
  +1  : position longue à 100%
   0  : cash (pas de position)
  -1  : position courte (si autorisée par la stratégie)

Les stratégies qui acceptent du shorting le précisent explicitement.

Utilisation :
    from strategy_zoo import momentum, sma_crossover, macro_tilt, vol_target, mean_reversion
    bt = Backtester(prices=prices, regime_df=df, strategy_fn=momentum, strategy_params={"window": 252})
"""

import numpy as np
import pandas as pd


# ─── 1. Momentum prix ─────────────────────────────────────────────────────────

def momentum(prices: pd.Series, window: int = 252) -> pd.Series:
    """
    Momentum de prix sur `window` jours.
    Long si le rendement sur la fenêtre est positif, sinon cash.

    Rationnel : les actifs qui ont bien performé continuent de surperformer
    (effet documenté dans la littérature depuis Jegadeesh & Titman 1993).

    Paramètres
    ----------
    window : int  Fenêtre de calcul en jours. 252 = 1 an (défaut).
                  Variantes courantes : 63 (3M), 126 (6M), 252 (12M).

    Signaux : {0, 1}  — pas de short
    """
    perf = prices / prices.shift(window) - 1
    signal = (perf > 0).astype(float)
    signal.name = f"momentum_{window}j"
    return signal


# ─── 2. Croisement de moyennes mobiles ────────────────────────────────────────

def sma_crossover(
    prices: pd.Series,
    short_window: int = 50,
    long_window: int = 200,
) -> pd.Series:
    """
    Croisement de moyennes mobiles simples (SMA).
    Long quand SMA_court > SMA_long (golden cross), cash sinon.

    Rationnel : filtre la tendance de fond et évite les grands drawdowns
    en sortant du marché pendant les tendances baissières prolongées.
    Stratégie de référence dans la gestion systématique depuis les années 80.

    Paramètres
    ----------
    short_window : int  Fenêtre courte en jours. Défaut : 50.
    long_window  : int  Fenêtre longue en jours. Défaut : 200.

    Signaux : {0, 1}  — pas de short
    """
    sma_short = prices.rolling(short_window, min_periods=short_window).mean()
    sma_long  = prices.rolling(long_window,  min_periods=long_window).mean()
    signal = (sma_short > sma_long).astype(float)
    signal.name = f"sma_{short_window}_{long_window}"
    return signal


# ─── 3. Mean reversion (RSI) ──────────────────────────────────────────────────

def mean_reversion(
    prices: pd.Series,
    rsi_window: int = 14,
    oversold: float = 35,
    overbought: float = 65,
) -> pd.Series:
    """
    Stratégie de retour à la moyenne basée sur le RSI.
    Long quand RSI < oversold (actif survendu), cash quand RSI > overbought.
    Neutre dans la zone intermédiaire.

    Rationnel : opposé du momentum — suppose que les excès de prix se corrigent.
    Performante sur les actifs cycliques et en régimes à forte volatilité.

    Paramètres
    ----------
    rsi_window  : int    Fenêtre RSI en jours. Défaut : 14.
    oversold    : float  Seuil d'entrée longue (RSI bas). Défaut : 35.
    overbought  : float  Seuil de sortie (RSI haut). Défaut : 65.

    Signaux : {0, 1}  — pas de short
    """
    delta = prices.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)

    avg_gain = gain.ewm(span=rsi_window, min_periods=rsi_window, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_window, min_periods=rsi_window, adjust=False).mean()

    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # Logique de signal avec mémoire (on reste long tant qu'on n'est pas overbought)
    signal = pd.Series(np.nan, index=prices.index, name=f"mean_rev_rsi{rsi_window}")
    signal[rsi < oversold]  = 1.0   # entrer long
    signal[rsi > overbought] = 0.0  # sortir

    # Forward-fill pour maintenir la position entre les signaux
    signal = signal.ffill().fillna(0.0)
    return signal


# ─── 4. Macro tilt ────────────────────────────────────────────────────────────

def macro_tilt(
    prices: pd.Series,
    regime_series: pd.Series,
    bull_regimes: list[str] | None = None,
) -> pd.Series:
    """
    Stratégie conduite par le régime macro.
    Long dans les régimes 'bull', cash dans les autres.

    C'est LA stratégie distinctive du projet : elle utilise directement
    la classification macroéconomique comme signal de trading. Elle ne
    dépend pas du tout des prix — seulement du contexte économique.

    Rationnel : les actifs risqués surperforment systématiquement dans
    les régimes d'expansion saine (croissance + désinflation + courbe normale).
    La stratégie cherche à capturer cette prime sans porter le risque
    des régimes défavorables.

    Paramètres
    ----------
    regime_series : pd.Series  Série MENSUELLE de labels de régime
                               (sortie de RegimeDetector.detect()["regime"]).
    bull_regimes  : list[str]  Régimes considérés comme favorables.
                               Défaut : expansion saine + fragile.

    Signaux : {0, 1}  — pas de short
    """
    if bull_regimes is None:
        bull_regimes = ["Expansion saine", "Expansion fragile"]

    # Régime mensuel → journalier par forward-fill
    regime_daily = regime_series.reindex(prices.index, method="ffill")

    signal = regime_daily.isin(bull_regimes).astype(float)
    signal.name = "macro_tilt"
    return signal


# ─── 5. Ciblage de volatilité ─────────────────────────────────────────────────

def vol_target(
    prices: pd.Series,
    target_vol: float = 0.10,
    vol_window: int = 21,
    max_leverage: float = 1.0,
) -> pd.Series:
    """
    Stratégie de ciblage de volatilité (toujours investie, exposition variable).
    L'exposition est ajustée chaque jour pour viser une volatilité annualisée cible.

    Différence clé par rapport aux autres stratégies : le signal n'est pas
    binaire (0/1) mais continu dans [0, max_leverage]. Ce n'est pas une
    stratégie d'entrée/sortie mais de gestion de taille de position.

    Rationnel : réduit l'exposition pendant les phases de forte volatilité
    (souvent des crises) et l'augmente en période calme. Produit un profil
    de risque plus régulier que le buy-and-hold.

    Paramètres
    ----------
    target_vol   : float  Volatilité annualisée cible. Défaut : 10% (0.10).
    vol_window   : int    Fenêtre de calcul de la vol réalisée (jours). Défaut : 21.
    max_leverage : float  Exposition maximale (1.0 = pas de levier). Défaut : 1.0.

    Signaux : [0, max_leverage]  — exposition variable, jamais short
    """
    daily_returns = prices.pct_change()

    # Volatilité réalisée annualisée sur la fenêtre glissante
    realized_vol = daily_returns.rolling(vol_window, min_periods=vol_window).std() * np.sqrt(252)

    # Taille de position = target_vol / realized_vol, plafonnée à max_leverage
    position = (target_vol / realized_vol.replace(0, np.nan)).clip(upper=max_leverage)

    # Décaler d'un jour (on connaît la vol d'hier, on ajuste demain)
    signal = position.shift(1).fillna(0.0)
    signal.name = f"vol_target_{int(target_vol*100)}pct"
    return signal


# ─── Registre des stratégies ──────────────────────────────────────────────────

STRATEGIES = {
    "Momentum 12M":          (momentum,        {"window": 252}),
    "SMA 50/200":            (sma_crossover,   {"short_window": 50, "long_window": 200}),
    "Mean reversion RSI14":  (mean_reversion,  {"rsi_window": 14}),
    "Vol target 10%":        (vol_target,      {"target_vol": 0.10}),
    # macro_tilt est ajouté dynamiquement dans metrics.py car il nécessite regime_series
}


# ─── Test rapide ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import yfinance as yf
    from regime_detector import RegimeDetector
    from metrics import run_all_strategies, build_comparison_df

    print("Chargement régimes...")
    rd = RegimeDetector(start_date="2003-01-01")
    regime_df = rd.detect()

    print("Téléchargement Euro Stoxx 50...")
    prices = yf.download("^STOXX50E", start="2003-01-01", progress=False)["Close"].squeeze()
    prices.name = "Euro Stoxx 50"

    results = run_all_strategies(prices, regime_df)

    print("\nComparaison globale :")
    print(build_comparison_df(results).to_string())

    print("\nDétail Mean reversion RSI14 :")
    results["Mean reversion RSI14"].summary()
