"""
walk_forward.py
===============
Implémente le walk-forward test pour valider la robustesse out-of-sample
des stratégies du Euro Macro Backtester.

Principe :
  - Expanding window : la calibration commence toujours en start_date et
    s'étend progressivement d'une année à chaque itération.
  - Sur chaque fenêtre de calibration, on identifie la meilleure stratégie
    (par Sharpe in-sample).
  - On applique cette stratégie à la fenêtre de test qui suit immédiatement.
  - On agrège les rendements out-of-sample de toutes les fenêtres pour
    construire une série OOS continue.
  - On compare IS vs OOS pour valider la robustesse.

Usage :
    from walk_forward import WalkForwardTest
    wf = WalkForwardTest(regime_df=rd.detect(), prices_dict=all_prices)
    result = wf.run()
    result.summary()
    result.plot_equity()

Dépendances :
    pip install python-dateutil pandas numpy
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from dateutil.relativedelta import relativedelta

from backtester import Backtester, BacktestResult, compute_metrics
from metrics import run_all_strategies, build_comparison_df


# ─── Génération des fenêtres ──────────────────────────────────────────────────

def generate_windows(
    start_date: str,
    end_date: str,
    calibration_years: int = 8,
    test_years: int = 1,
) -> list[dict]:
    """
    Génère les fenêtres train/test pour le walk-forward.

    Expanding window : la calibration commence toujours à start_date.
    À chaque itération elle s'étend de test_years. Le test suit immédiatement.

    Retourne une liste de dicts :
        [{window, train_start, train_end, test_start, test_end}, ...]
    """
    start = pd.Timestamp(start_date)
    end   = pd.Timestamp(end_date)
    windows = []
    window_num = 1
    train_end = start + relativedelta(years=calibration_years)

    while True:
        test_start = train_end + relativedelta(months=1)
        test_end   = test_start + relativedelta(years=test_years) - relativedelta(months=1)

        if test_end > end:
            break

        windows.append({
            "window":      window_num,
            "train_start": start,
            "train_end":   train_end,
            "test_start":  test_start,
            "test_end":    test_end,
        })
        train_end  += relativedelta(years=test_years)
        window_num += 1

    return windows


# ─── Résultats d'une fenêtre ──────────────────────────────────────────────────

@dataclass
class WindowResult:
    """
    Résultat d'une fenêtre walk-forward pour un actif et une stratégie donnés.

    Attributs
    ---------
    window          : numéro de la fenêtre
    train_start     : début de la calibration
    train_end       : fin de la calibration
    test_start      : début du test
    test_end        : fin du test
    best_strategy   : stratégie choisie sur la calibration (Sharpe IS le plus élevé)
    sharpe_is       : Sharpe de best_strategy sur la calibration
    sharpe_oos      : Sharpe de best_strategy sur le test (valeur cible)
    returns_oos     : série journalière des rendements OOS (pour concatenation)
    regimes_oos     : régimes macro pendant la période de test
    all_is_sharpes  : Sharpe IS de toutes les stratégies (pour analyser la stabilité)
    all_oos_sharpes : Sharpe OOS de toutes les stratégies (comparaison complète)
    """
    window:          int
    train_start:     pd.Timestamp
    train_end:       pd.Timestamp
    test_start:      pd.Timestamp
    test_end:        pd.Timestamp
    best_strategy:   str
    sharpe_is:       float
    sharpe_oos:      float
    returns_oos:     pd.Series
    regimes_oos:     pd.Series
    all_is_sharpes:  dict[str, float]
    all_oos_sharpes: dict[str, float]


# ─── Résultats agrégés ────────────────────────────────────────────────────────

@dataclass
class WalkForwardResult:
    """
    Résultats agrégés du walk-forward pour un actif.

    Attributs
    ---------
    asset_name       : nom de l'actif
    window_results   : liste des WindowResult, un par fenêtre
    returns_oos      : série OOS continue (concaténation de toutes les fenêtres)
    returns_is_full  : série IS du backtest classique sur la même période totale
    regime_df        : DataFrame complet des régimes
    """
    asset_name:      str
    window_results:  list[WindowResult]
    returns_oos:     pd.Series
    returns_is_full: pd.Series
    regime_df:       pd.DataFrame

    # ── Métriques globales IS vs OOS ─────────────────────────────────────────

    def is_oos_comparison(self) -> pd.DataFrame:
        """
        Tableau comparatif IS vs OOS par stratégie sur la période OOS.

        Pour rendre la comparaison équitable, le Sharpe IS est calculé
        sur la même période que le Sharpe OOS (à partir de la première
        fenêtre de test). Ce n'est pas le Sharpe IS "global" — c'est le
        Sharpe moyen IS pondéré par les fenêtres.
        """
        # Sharpe IS moyen pondéré par la durée de chaque fenêtre
        strategies = list(self.window_results[0].all_is_sharpes.keys())
        rows = []
        for strat in strategies:
            is_sharpes  = [w.all_is_sharpes.get(strat, np.nan)  for w in self.window_results]
            oos_sharpes = [w.all_oos_sharpes.get(strat, np.nan) for w in self.window_results]
            durations   = [
                (w.test_end - w.test_start).days for w in self.window_results
            ]

            # Pondération par durée de la fenêtre de test
            def weighted_avg(values, weights):
                pairs = [(v, w) for v, w in zip(values, weights) if not np.isnan(v)]
                if not pairs:
                    return np.nan
                vals, wts = zip(*pairs)
                return np.average(vals, weights=wts)

            rows.append({
                "Stratégie":    strat,
                "Sharpe IS":    round(weighted_avg(is_sharpes,  durations), 3),
                "Sharpe OOS":   round(weighted_avg(oos_sharpes, durations), 3),
            })

        df = pd.DataFrame(rows).set_index("Stratégie")
        # Dégradation = combien le Sharpe se détériore en passant à l'OOS
        df["Dégradation"] = (df["Sharpe OOS"] - df["Sharpe IS"]).round(3)
        df["Robuste ?"] = df["Dégradation"].apply(
            lambda d: "✓ Oui" if d > -0.15 else ("△ Partiel" if d > -0.40 else "✗ Non")
        )
        return df.sort_values("Sharpe OOS", ascending=False)

    # ── Métriques OOS continues ───────────────────────────────────────────────

    def oos_metrics(self) -> dict:
        """Métriques de la série OOS continue (la vraie performance hors-échantillon)."""
        invested_mask = self.returns_oos.abs() > 0.0001
        return compute_metrics(
            self.returns_oos,
            label="Walk-Forward OOS",
            invested_mask=invested_mask,
        )

    # ── Performance OOS par régime ────────────────────────────────────────────

    def oos_by_regime(self) -> pd.DataFrame:
        """
        Décompose la performance OOS par régime macroéconomique.
        Équivalent du by_regime() du Backtester, mais sur données OOS uniquement.
        """
        # Régimes pendant la période OOS
        oos_regimes = pd.concat([w.regimes_oos for w in self.window_results])
        oos_returns = self.returns_oos.reindex(oos_regimes.index)

        rows = []
        for regime in sorted(oos_regimes.dropna().unique()):
            mask = oos_regimes == regime
            regime_returns = oos_returns[mask]
            invested = regime_returns.abs() > 0.0001
            m = compute_metrics(regime_returns, label=regime, invested_mask=invested)
            rows.append(m)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).set_index("label")
        df.index.name = "Régime"
        return df.sort_values("sharpe", ascending=False)

    # ── Stabilité de la sélection de stratégie ───────────────────────────────

    def strategy_stability(self) -> pd.DataFrame:
        """
        Pour chaque fenêtre, quelle stratégie a été choisie en IS ?
        Montre si le modèle change constamment d'avis ou reste cohérent.
        Un bon modèle choisit la même stratégie dans ≥ 70% des fenêtres.
        """
        rows = [{
            "Fenêtre": w.window,
            "Période test": f"{w.test_start.strftime('%Y-%m')} → {w.test_end.strftime('%Y-%m')}",
            "Stratégie choisie": w.best_strategy,
            "Sharpe IS": round(w.sharpe_is, 3),
            "Sharpe OOS": round(w.sharpe_oos, 3),
            "IS → OOS": "✓" if w.sharpe_oos > 0 else "✗",
        } for w in self.window_results]
        return pd.DataFrame(rows).set_index("Fenêtre")

    # ── Résumé terminal ───────────────────────────────────────────────────────

    def summary(self) -> None:
        n = len(self.window_results)
        oos_start = self.window_results[0].test_start.strftime("%b %Y")
        oos_end   = self.window_results[-1].test_end.strftime("%b %Y")

        print(f"\n{'═'*62}")
        print(f"  Walk-Forward Test — {self.asset_name}")
        print(f"  {n} fenêtres  ·  période OOS : {oos_start} → {oos_end}")
        print(f"{'═'*62}")

        # Taux de succès : % de fenêtres où le Sharpe OOS > 0
        win_rate = np.mean([w.sharpe_oos > 0 for w in self.window_results]) * 100
        print(f"\n  Taux de fenêtres OOS positives : {win_rate:.0f}%")
        print(f"  (benchmark naïf : ~50% par hasard)\n")

        print("  IS vs OOS par stratégie :\n")
        comp = self.is_oos_comparison()
        header = f"  {'Stratégie':<28}  {'Sharpe IS':>9}  {'Sharpe OOS':>10}  {'Dégrad.':>8}  {'Robuste ?'}"
        print(header)
        print("  " + "─" * (len(header) - 2))
        for strat, row in comp.iterrows():
            print(
                f"  {strat:<28}  {row['Sharpe IS']:>9.3f}  "
                f"{row['Sharpe OOS']:>10.3f}  {row['Dégradation']:>8.3f}  {row['Robuste ?']}"
            )

        print(f"\n  Stabilité de sélection :")
        stab = self.strategy_stability()
        choices = stab["Stratégie choisie"].value_counts()
        for strat, count in choices.items():
            bar = "█" * count
            print(f"    {strat:<28} {bar} ({count}/{n} fenêtres)")

        print(f"\n{'═'*62}\n")


# ─── Moteur principal ─────────────────────────────────────────────────────────

class WalkForwardTest:
    """
    Orchestre le walk-forward test pour un actif donné.

    Paramètres
    ----------
    regime_df         : pd.DataFrame  Sortie de RegimeDetector.detect().
    prices            : pd.Series     Prix de clôture journaliers de l'actif.
    asset_name        : str           Nom de l'actif (pour les affichages).
    calibration_years : int           Durée de la première fenêtre de calibration.
    test_years        : int           Durée de chaque fenêtre de test.
    tc_bps            : float         Coûts de transaction en bps.
    verbose           : bool          Afficher la progression fenêtre par fenêtre.
    """

    def __init__(
        self,
        regime_df:         pd.DataFrame,
        prices:            pd.Series,
        asset_name:        str = "actif",
        calibration_years: int = 8,
        test_years:        int = 1,
        tc_bps:            float = 5.0,
        verbose:           bool = True,
    ) -> None:
        self.regime_df         = regime_df
        self.prices            = prices.dropna().sort_index()
        self.asset_name        = asset_name
        self.calibration_years = calibration_years
        self.test_years        = test_years
        self.tc_bps            = tc_bps
        self.verbose           = verbose

    def _run_window(self, window: dict) -> WindowResult | None:
        """
        Exécute une fenêtre du walk-forward.

        Étapes internes :
          1. Découper regime_df et prices sur la période de calibration.
          2. Lancer toutes les stratégies sur la calibration → Sharpes IS.
          3. Identifier la meilleure stratégie (Sharpe IS le plus élevé).
          4. L'appliquer sur la période de test → Sharpe OOS.
          5. Retourner les résultats.
        """
        ts, te = window["train_start"], window["train_end"]
        vs, ve = window["test_start"],  window["test_end"]

        # ── Découpage ──
        regime_train = self.regime_df.loc[
            (self.regime_df.index >= ts) & (self.regime_df.index <= te)
        ]
        regime_test = self.regime_df.loc[
            (self.regime_df.index >= vs) & (self.regime_df.index <= ve)
        ]
        prices_train = self.prices.loc[
            (self.prices.index >= ts) & (self.prices.index <= te)
        ]
        prices_test = self.prices.loc[
            (self.prices.index >= vs) & (self.prices.index <= ve)
        ]

        # Vérifications de données suffisantes
        if len(prices_train) < 200 or len(prices_test) < 20:
            return None
        if regime_train.empty or regime_test.empty:
            return None

        # ── Calibration : toutes les stratégies en IS ──
        try:
            results_train = run_all_strategies(
                prices_train, regime_train, tc_bps=self.tc_bps
            )
        except Exception:
            return None

        comp_train = build_comparison_df(results_train)
        if comp_train.empty:
            return None

        all_is_sharpes = comp_train["Sharpe"].to_dict()

        # Meilleure stratégie en IS (Sharpe le plus élevé)
        best_strategy_name = comp_train.index[0]
        best_sharpe_is     = comp_train.iloc[0]["Sharpe"]

        # ── Test OOS : appliquer la même stratégie sur la période de test ──
        try:
            results_test = run_all_strategies(
                prices_test, regime_test, tc_bps=self.tc_bps
            )
        except Exception:
            return None

        comp_test       = build_comparison_df(results_test)
        all_oos_sharpes = comp_test["Sharpe"].to_dict()

        # Sharpe OOS de la stratégie choisie en IS
        if best_strategy_name not in results_test:
            return None

        best_result_oos = results_test[best_strategy_name]
        sharpe_oos      = best_result_oos.global_metrics()["sharpe"]

        if self.verbose:
            sign = "✓" if sharpe_oos > 0 else "✗"
            print(
                f"    [{window['window']:>2}] "
                f"Test {vs.strftime('%Y-%m')}→{ve.strftime('%Y-%m')}  "
                f"Strat. choisie : {best_strategy_name:<25}  "
                f"IS: {best_sharpe_is:>6.3f}  "
                f"OOS: {sharpe_oos:>6.3f}  {sign}"
            )

        return WindowResult(
            window          = window["window"],
            train_start     = ts,
            train_end       = te,
            test_start      = vs,
            test_end        = ve,
            best_strategy   = best_strategy_name,
            sharpe_is       = best_sharpe_is,
            sharpe_oos      = sharpe_oos,
            returns_oos     = best_result_oos.returns,
            regimes_oos     = best_result_oos.regimes,
            all_is_sharpes  = all_is_sharpes,
            all_oos_sharpes = all_oos_sharpes,
        )

    def run(self) -> WalkForwardResult:
        """
        Lance le walk-forward complet.

        1. Génère toutes les fenêtres.
        2. Pour chaque fenêtre : calibration IS + test OOS.
        3. Agrège les rendements OOS en une série continue.
        4. Calcule le backtest IS classique sur la même période totale
           (pour la comparaison équitable).
        """
        start = self.prices.index[0].strftime("%Y-%m-%d")
        end   = self.prices.index[-1].strftime("%Y-%m-%d")

        windows = generate_windows(
            start_date=start,
            end_date=end,
            calibration_years=self.calibration_years,
            test_years=self.test_years,
        )

        if not windows:
            raise ValueError(
                f"Aucune fenêtre générée. Vérifie que la période de données "
                f"({start} → {end}) est suffisante pour {self.calibration_years} ans "
                f"de calibration + {self.test_years} an(s) de test."
            )

        if self.verbose:
            print(f"\n  Walk-forward : {len(windows)} fenêtres à calculer...\n")

        # ── Exécution de toutes les fenêtres ──
        window_results = []
        for w in windows:
            result = self._run_window(w)
            if result is not None:
                window_results.append(result)

        if not window_results:
            raise RuntimeError("Aucune fenêtre n'a produit de résultats valides.")

        # ── Série OOS continue (concaténation de toutes les fenêtres) ──
        returns_oos = pd.concat(
            [w.returns_oos for w in window_results]
        ).sort_index()

        # Dédoublonnage (possible si les fenêtres se chevauchent en jours)
        returns_oos = returns_oos[~returns_oos.index.duplicated(keep="first")]

        # ── Backtest IS de référence sur la période OOS ──
        # On utilise la stratégie la plus souvent choisie en IS pour la comparaison
        most_common_strategy = (
            pd.Series([w.best_strategy for w in window_results])
            .value_counts()
            .index[0]
        )

        oos_start = window_results[0].test_start
        oos_end   = window_results[-1].test_end

        regime_oos_period = self.regime_df.loc[
            (self.regime_df.index >= oos_start) & (self.regime_df.index <= oos_end)
        ]
        prices_oos_period = self.prices.loc[
            (self.prices.index >= oos_start) & (self.prices.index <= oos_end)
        ]

        try:
            is_reference = run_all_strategies(
                prices_oos_period, regime_oos_period, tc_bps=self.tc_bps
            )
            returns_is_full = is_reference[most_common_strategy].returns
        except Exception:
            returns_is_full = pd.Series(dtype=float)

        if self.verbose:
            print(f"\n  Terminé. {len(window_results)} fenêtres valides.")

        return WalkForwardResult(
            asset_name      = self.asset_name,
            window_results  = window_results,
            returns_oos     = returns_oos,
            returns_is_full = returns_is_full,
            regime_df       = self.regime_df,
        )


# ─── Test rapide ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import yfinance as yf
    from regime_detector import RegimeDetector

    print("Chargement régimes...")
    rd = RegimeDetector(start_date="2004-01-01")
    regime_df = rd.detect()

    print("Téléchargement Euro Stoxx 50...")
    prices = yf.download("^STOXX50E", start="2004-01-01", progress=False)["Close"].squeeze()
    prices.name = "Euro Stoxx 50"

    print("\nLancement du walk-forward (cela prend 1-2 minutes)...")
    wf = WalkForwardTest(
        regime_df         = regime_df,
        prices            = prices,
        asset_name        = "Euro Stoxx 50",
        calibration_years = 8,
        test_years        = 1,
        tc_bps            = 5.0,
        verbose           = True,
    )
    result = wf.run()
    result.summary()

    print("Performance OOS par régime :")
    print(result.oos_by_regime()[["sharpe", "rendement_ann", "max_drawdown", "n_mois_approx"]].to_string())
