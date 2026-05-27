"""
backtester.py
=============
Backteste une stratégie sur des données de prix journalières et décompose
les performances par régime macroéconomique.

Utilisation minimale :
    from backtester import Backtester
    from regime_detector import RegimeDetector
    import yfinance as yf

    rd = RegimeDetector()
    regime_df = rd.detect()

    prices = yf.download("^STOXX50E", start="2000-01-01")["Close"].squeeze()

    bt = Backtester(prices=prices, regime_df=regime_df, strategy_fn=momentum)
    result = bt.run()
    result.summary()
    print(result.by_regime())

Dépendances :
    pip install yfinance pandas numpy
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Callable


# ─── Métriques ────────────────────────────────────────────────────────────────

def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Rendement annualisé géométrique."""
    if len(returns) < 2:
        return np.nan
    total = (1 + returns).prod()
    n_years = len(returns) / periods_per_year
    return float(total ** (1 / n_years) - 1) if n_years > 0 else np.nan


def annualized_vol(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Volatilité annualisée."""
    if len(returns) < 2:
        return np.nan
    return float(returns.std() * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    risk_free: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Ratio de Sharpe annualisé (risk-free = 0 par défaut)."""
    vol = annualized_vol(returns, periods_per_year)
    if vol == 0 or np.isnan(vol):
        return np.nan
    excess = annualized_return(returns, periods_per_year) - risk_free
    return float(excess / vol)


def max_drawdown(returns: pd.Series) -> float:
    """Drawdown maximum (valeur négative, en fraction)."""
    if len(returns) < 2:
        return np.nan
    equity = (1 + returns).cumprod()
    peak   = equity.cummax()
    dd     = (equity - peak) / peak
    return float(dd.min())


def compute_metrics(
    returns: pd.Series,
    label: str = "",
    invested_mask: pd.Series | None = None,
) -> dict:
    """
    Calcule l'ensemble des métriques pour une série de rendements.

    Paramètres
    ----------
    returns       : rendements journaliers nets de la stratégie.
    label         : nom affiché dans les tableaux.
    invested_mask : booléens indiquant les jours où la stratégie est investie.
                    Sert à calculer le win rate uniquement sur les jours actifs
                    (exclut les jours en cash où le rendement est ~0).
                    Si None, tous les jours sont considérés.
    """
    clean = returns.dropna()
    n_days = len(clean)

    # Jours réellement investis (signal ≠ 0)
    if invested_mask is not None:
        inv = invested_mask.reindex(clean.index).fillna(False)
    else:
        inv = pd.Series(True, index=clean.index)

    invested_returns = clean[inv]
    pct_investis = round(inv.sum() / n_days * 100, 1) if n_days > 0 else np.nan
    win_rate = (
        round((invested_returns > 0).mean() * 100, 1)
        if len(invested_returns) > 0
        else np.nan
    )

    return {
        "label":          label,
        "n_jours":        n_days,
        "n_mois_approx":  round(n_days / 21),
        "rendement_ann":  round(annualized_return(clean) * 100, 2),
        "vol_ann":        round(annualized_vol(clean) * 100, 2),
        "sharpe":         round(sharpe_ratio(clean), 3),
        "max_drawdown":   round(max_drawdown(clean) * 100, 2),
        "pct_investis":   pct_investis,
        "win_rate":       win_rate,
    }


# ─── Résultats ────────────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    """
    Contient tous les résultats d'un backtest.

    Attributs
    ---------
    equity_curve  : valeur du portefeuille au fil du temps (base 100)
    returns       : rendements journaliers nets (après coûts de transaction)
    signals       : signaux de position journaliers (-1 short, 0 cash, 1 long)
    regimes       : régime macro pour chaque journée (forward-fill mensuel→journalier)
    strategy_name : nom de la stratégie (pour les graphiques)
    asset_name    : ticker ou nom de l'actif backTesté
    """

    equity_curve:  pd.Series
    returns:       pd.Series
    signals:       pd.Series
    regimes:       pd.Series
    strategy_name: str = "stratégie"
    asset_name:    str = "actif"

    # ── Métriques globales ──────────────────────────────────────────────────

    def global_metrics(self) -> dict:
        """Métriques de performance sur la période complète."""
        invested_mask = self.signals.abs() > 0.01
        return compute_metrics(self.returns, label="Globale", invested_mask=invested_mask)

    # ── Métriques par régime ─────────────────────────────────────────────────

    def by_regime(self) -> pd.DataFrame:
        """
        DataFrame avec les métriques (Sharpe, rendement, drawdown…)
        pour chaque régime macroéconomique observé pendant le backtest.

        C'est LA sortie différenciante du projet.
        """
        rows = []
        for regime in sorted(self.regimes.dropna().unique()):
            mask           = self.regimes == regime
            regime_returns = self.returns[mask]
            regime_signals = self.signals[mask]
            invested_mask  = regime_signals.abs() > 0.01
            m = compute_metrics(regime_returns, label=regime, invested_mask=invested_mask)
            rows.append(m)

        if not rows:
            empty = pd.DataFrame(columns=[
                "n_jours", "n_mois_approx", "rendement_ann",
                "vol_ann", "sharpe", "max_drawdown", "pct_investis", "win_rate",
            ])
            empty.index.name = "Régime"
            return empty

        df = pd.DataFrame(rows).set_index("label")
        df.index.name = "Régime"
        return df.sort_values("sharpe", ascending=False)

    # ── Benchmark ────────────────────────────────────────────────────────────

    def vs_buyhold(self, asset_returns: pd.Series) -> pd.DataFrame:
        """
        Compare les métriques de la stratégie vs un simple buy-and-hold.

        Paramètre
        ---------
        asset_returns : rendements journaliers bruts de l'actif (pd.Series).
        """
        strat_m = self.global_metrics()

        bh_returns = asset_returns.reindex(self.returns.index).dropna()
        # Buy & hold : toujours investi à 100%, win rate sur tous les jours
        bh_invested = pd.Series(True, index=bh_returns.index)
        bh_m = compute_metrics(bh_returns, label="Buy & Hold", invested_mask=bh_invested)

        cols = ["rendement_ann", "vol_ann", "sharpe", "max_drawdown", "pct_investis", "win_rate"]
        df = pd.DataFrame([strat_m, bh_m]).set_index("label")[cols]
        df.index.name = None
        return df

    # ── Résumé terminal ──────────────────────────────────────────────────────

    def summary(self) -> None:
        g = self.global_metrics()

        start = self.equity_curve.index[0].strftime("%b %Y")
        end   = self.equity_curve.index[-1].strftime("%b %Y")

        print(f"\n{'─'*60}")
        print(f"  Backtest · {self.strategy_name} · {self.asset_name}")
        print(f"  Période  : {start} → {end}  ({g['n_jours']} jours)")
        print(f"{'─'*60}")
        print(f"  Rendement annualisé : {g['rendement_ann']:>8.2f} %")
        print(f"  Volatilité ann.     : {g['vol_ann']:>8.2f} %")
        print(f"  Sharpe              : {g['sharpe']:>8.3f}")
        print(f"  Max drawdown        : {g['max_drawdown']:>8.2f} %")
        print(f"  Jours investis      : {g['pct_investis']:>8.1f} %")
        print(f"  Win rate (investis) : {g['win_rate']:>8.1f} %")
        print(f"{'─'*60}")

        print("\n  Performance par régime :\n")
        df = self.by_regime()
        header = (
            f"  {'Régime':<38}  {'Sharpe':>7}  {'Rend.%':>7}"
            f"  {'MDD%':>7}  {'Inv.%':>6}  {'Mois':>5}"
        )
        print(header)
        print("  " + "─" * (len(header) - 2))
        for regime, row in df.iterrows():
            sharpe_str = f"{row['sharpe']:>7.2f}" if not np.isnan(row['sharpe']) else "     —"
            inv_str    = f"{row['pct_investis']:>6.1f}" if not np.isnan(row['pct_investis']) else "    —"
            print(
                f"  {str(regime):<38}  {sharpe_str}  "
                f"{row['rendement_ann']:>7.2f}  "
                f"{row['max_drawdown']:>7.2f}  "
                f"{inv_str}  "
                f"{int(row['n_mois_approx']):>5}"
            )
        print(f"{'─'*60}\n")


# ─── Moteur de backtest ───────────────────────────────────────────────────────

class Backtester:
    """
    Backteste une stratégie sur des données journalières et décompose
    les résultats par régime macroéconomique.

    Paramètres
    ----------
    prices               : pd.Series  Prix de clôture journaliers (yfinance).
    regime_df            : pd.DataFrame  Sortie de RegimeDetector.detect().
    strategy_fn          : callable   Fonction de signal — voir strategy_zoo.py.
                           Signature : f(prices: pd.Series, **params) → pd.Series[float]
                           Les valeurs doivent être dans [-1, 1].
    strategy_params      : dict       Paramètres passés à strategy_fn.
    strategy_name        : str        Nom affiché dans les résultats.
    initial_capital      : float      Capital de départ (pour l'equity curve).
    transaction_cost_bps : float      Coûts de transaction en points de base
                           appliqués à chaque changement de position.
                           Défaut : 5 bps (0.05%).
    """

    def __init__(
        self,
        prices: pd.Series,
        regime_df: pd.DataFrame,
        strategy_fn: Callable,
        strategy_params: dict | None = None,
        strategy_name: str = "stratégie",
        initial_capital: float = 100.0,
        transaction_cost_bps: float = 5.0,
    ) -> None:
        self.prices          = prices.dropna().sort_index()
        self.regime_df       = regime_df
        self.strategy_fn     = strategy_fn
        self.strategy_params = strategy_params or {}
        self.strategy_name   = strategy_name
        self.initial_capital = initial_capital
        self.tc_bps          = transaction_cost_bps
        self._result: BacktestResult | None = None

    # ── Cœur du backtest ─────────────────────────────────────────────────────

    def run(self) -> BacktestResult:
        """
        Exécute le backtest et retourne un objet BacktestResult.

        Pipeline :
          1. Génère les signaux via strategy_fn
          2. Décale d'un jour (évite le look-ahead bias)
          3. Calcule les rendements nets (après coûts de transaction)
          4. Mappe les régimes mensuels → journaliers par forward-fill
          5. Construit l'equity curve
        """
        prices = self.prices

        # 1. Signaux bruts de la stratégie
        signals_raw = self.strategy_fn(prices, **self.strategy_params)
        signals_raw = signals_raw.reindex(prices.index).ffill()

        # 2. Décalage de 1 jour : on exécute à l'ouverture du jour suivant
        #    → la stratégie ne peut jamais "voir" le rendement du jour même
        signals = signals_raw.shift(1).fillna(0)

        # 3. Rendements bruts de l'actif
        asset_returns = prices.pct_change()

        # 4. Coûts de transaction (sur les changements de position)
        position_changes = signals.diff().abs()
        tc = position_changes * (self.tc_bps / 10_000)

        # 5. Rendements nets de la stratégie
        strat_returns = signals * asset_returns - tc

        # 6. Régimes mensuels → journaliers
        #    On prend la colonne "regime" du DataFrame et on forward-fill
        regime_monthly = self.regime_df["regime"]
        regime_daily   = regime_monthly.reindex(
            asset_returns.index, method="ffill"
        )

        # 7. Equity curve (base = initial_capital)
        equity = self.initial_capital * (1 + strat_returns).cumprod()

        # 8. Aligner tous les index sur les jours où on a les données
        common_idx = (
            strat_returns.dropna()
            .index
            .intersection(regime_daily.dropna().index)
        )
        strat_returns = strat_returns.loc[common_idx]
        signals       = signals.loc[common_idx]
        regime_daily  = regime_daily.loc[common_idx]
        equity        = equity.loc[common_idx]

        self._result = BacktestResult(
            equity_curve  = equity,
            returns       = strat_returns,
            signals       = signals,
            regimes       = regime_daily,
            strategy_name = self.strategy_name,
            asset_name    = getattr(prices, "name", "actif") or "actif",
        )
        return self._result

    @property
    def result(self) -> BacktestResult:
        if self._result is None:
            raise RuntimeError("Lance d'abord backtester.run()")
        return self._result



