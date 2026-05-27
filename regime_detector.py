"""
regime_detector.py
==================
Classifie chaque mois historique en un régime macroéconomique (zone euro).

Sources :
  - Inflation  : eurostat  prc_hicp_midx  (HICP index → YoY calculé)
  - Croissance : eurostat  namq_10_gdp    (PIB réel niveau → YoY calculé)
  - Taux long  : FRED      IRLTLT01EZM156N
  - Taux court : FRED      IR3TIB01EZM156N

Dépendances :
    pip install fredapi eurostat pandas python-dotenv
"""

import os
import pandas as pd
from fredapi import Fred
from dotenv import load_dotenv

load_dotenv()

# ─── Régimes : 8 combinaisons (inflation × courbe × croissance) ───────────────

REGIME_LABELS: dict[tuple[bool, bool, bool], str] = {
    (True,  True,  True):  "Surchauffe",
    (True,  True,  False): "Ralentissement inflationniste",
    (True,  False, True):  "Stagflation naissante",
    (True,  False, False): "Stagflation",
    (False, True,  True):  "Expansion saine",
    (False, True,  False): "Désinflation douce",
    (False, False, True):  "Expansion fragile",
    (False, False, False): "Récession-désinflation",
}

REGIME_COLORS: dict[str, str] = {
    "Surchauffe":                    "#D85A30",
    "Ralentissement inflationniste": "#EF9F27",
    "Stagflation naissante":         "#BA7517",
    "Stagflation":                   "#A32D2D",
    "Expansion saine":               "#1D9E75",
    "Désinflation douce":            "#5DCAA5",
    "Expansion fragile":             "#9FE1CB",
    "Récession-désinflation":        "#85B7EB",
}


# ─── Helpers eurostat ─────────────────────────────────────────────────────────

def _eurostat_split(df: pd.DataFrame) -> tuple[str, list[str], list[str]]:
    """
    Sépare les colonnes d'un DataFrame eurostat en :
      - geo_col   : nom de la colonne géographique  (ex: 'geo\\TIME_PERIOD')
      - dim_cols  : toutes les colonnes de dimension (freq, unit, ..., geo)
      - time_cols : colonnes temporelles             (ex: ['1996-01', '1996-02', ...])

    La colonne géo est toujours la dernière colonne de dimension dans la lib eurostat.
    """
    geo_col  = next(c for c in df.columns if "geo" in c.lower())
    geo_idx  = df.columns.tolist().index(geo_col)
    dim_cols = df.columns[:geo_idx + 1].tolist()
    time_cols = df.columns[geo_idx + 1:].tolist()
    return geo_col, dim_cols, time_cols


def _fetch_hicp_eurostat(start: str) -> pd.Series:
    """
    Récupère le HICP zone euro (index 2015=100) depuis eurostat
    et retourne le glissement annuel en %.

    Dataset : prc_hicp_midx
    Filtre  : unit=I15, coicop=CP00, geo=EA
    """
    import eurostat
    df = eurostat.get_data_df("prc_hicp_midx", flags=False)
    geo_col, _, time_cols = _eurostat_split(df)

    row = df[
        (df["unit"]    == "I15")
        & (df["coicop"] == "CP00")
        & (df[geo_col]  == "EA")
    ]
    if row.empty:
        raise RuntimeError(
            "Aucune ligne HICP pour EA/I15/CP00 dans prc_hicp_midx.\n"
            f"Valeurs geo disponibles : {df[geo_col].unique()[:10]}"
        )

    values = pd.to_numeric(row[time_cols].iloc[0], errors="coerce")

    # Format des colonnes : '1996-01' → 1er de chaque mois
    idx = pd.to_datetime([c + "-01" for c in time_cols], format="%Y-%m-%d")
    series = pd.Series(values.values, index=idx, name="hicp_index")

    # Filtrer par date et calculer le YoY
    series = series.sort_index().loc[pd.Timestamp(start):]
    yoy = series.pct_change(12).mul(100)

    print(f"  ✓ Inflation : eurostat prc_hicp_midx  "
          f"({series.index[0].strftime('%Y-%m')} → {series.index[-1].strftime('%Y-%m')})")
    return yoy


def _fetch_gdp_eurostat(start: str) -> pd.Series:
    """
    Récupère le PIB réel zone euro (niveau chaîné) depuis eurostat,
    calcule le glissement annuel en % et retourne une série mensuelle.

    Dataset : namq_10_gdp
    Filtre  : unit=CLV10_MEUR, s_adj=SCA, na_item=B1GQ, geo=EA
    """
    import eurostat
    df = eurostat.get_data_df("namq_10_gdp", flags=False)
    geo_col, _, time_cols = _eurostat_split(df)

    row = df[
        (df["unit"]    == "CLV10_MEUR")
        & (df["s_adj"]  == "SCA")
        & (df["na_item"] == "B1GQ")
        & (df[geo_col]  == "EA")
    ]
    if row.empty:
        raise RuntimeError(
            "Aucune ligne GDP pour EA/CLV10_MEUR/SCA/B1GQ dans namq_10_gdp.\n"
            f"Valeurs geo disponibles : {df[geo_col].unique()[:10]}"
        )

    values = pd.to_numeric(row[time_cols].iloc[0], errors="coerce")

    # Format des colonnes : '1975-Q1' → PeriodIndex trimestriel → début de trimestre
    idx = pd.PeriodIndex(time_cols, freq="Q").to_timestamp()
    series = pd.Series(values.values, index=idx, name="gdp_level")

    # Filtrer et calculer le YoY trimestriel
    series = series.sort_index().loc[pd.Timestamp(start):]
    yoy_quarterly = series.pct_change(4).mul(100)

    # Trimestriel → mensuel par forward-fill (3 mois max)
    yoy_monthly = yoy_quarterly.resample("MS").last().ffill(limit=3)

    print(f"  ✓ Croissance : eurostat namq_10_gdp  "
          f"({yoy_monthly.dropna().index[0].strftime('%Y-%m')} "
          f"→ {yoy_monthly.dropna().index[-1].strftime('%Y-%m')})")
    return yoy_monthly


# ─── Classe principale ────────────────────────────────────────────────────────

class RegimeDetector:
    """
    Détecte les régimes macroéconomiques de la zone euro sur un historique mensuel.

    Les trois dimensions du régime :
      1. Inflation élevée  : HICP YoY > inflation_threshold (défaut 3%)
      2. Courbe normale    : spread 10Y–3M > curve_threshold (défaut 0%)
      3. Croissance forte  : PIB YoY > growth_threshold      (défaut 0.5%)

    Paramètres
    ----------
    start_date          : str   Date de début (YYYY-MM-DD). Défaut : '2000-01-01'.
    inflation_threshold : float Seuil d'inflation YoY en %. Défaut : 3.0.
    curve_threshold     : float Seuil du spread 10Y–3M en %. Défaut : 0.0.
    growth_threshold    : float Seuil PIB YoY en %. Défaut : 0.5.
    smoothing_window    : int   Fenêtre de lissage rolling (mois). Défaut : 3.
    """

    def __init__(
        self,
        start_date: str = "2000-01-01",
        inflation_threshold: float = 3.0,
        curve_threshold: float = 0.0,
        growth_threshold: float = 0.5,
        smoothing_window: int = 3,
    ) -> None:
        api_key = os.getenv("FRED_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "FRED_API_KEY manquante. Crée un fichier .env avec : FRED_API_KEY=ta_clé"
            )
        self.fred = Fred(api_key=api_key)
        self.start_date = start_date
        self.inflation_threshold = inflation_threshold
        self.curve_threshold = curve_threshold
        self.growth_threshold = growth_threshold
        self.smoothing_window = smoothing_window
        self._raw: pd.DataFrame | None = None
        self._regimes: pd.DataFrame | None = None

    # ── 1. Récupération ──────────────────────────────────────────────────────

    def fetch(self) -> pd.DataFrame:
        """
        Télécharge les quatre indicateurs :
          - HICP & PIB   : eurostat  (données actuelles jusqu'à 2025-2026)
          - Taux long/court : FRED   (séries BCE stables)

        Retourne un DataFrame mensuel avec les colonnes :
            inflation, long_rate, short_rate, gdp_yoy, curve
        """
        print("\nRécupération des données...")

        # ── Inflation : eurostat ──
        inflation = _fetch_hicp_eurostat(self.start_date)

        # ── Taux : FRED (séries BCE, stables et à jour) ──
        long_raw  = self.fred.get_series("IRLTLT01EZM156N", observation_start=self.start_date)
        short_raw = self.fred.get_series("IR3TIB01EZM156N", observation_start=self.start_date)
        long_rate  = long_raw.resample("MS").last()
        short_rate = short_raw.resample("MS").last()
        print(f"  ✓ Taux long  : FRED IRLTLT01EZM156N  "
              f"(→ {long_rate.dropna().index[-1].strftime('%Y-%m')})")
        print(f"  ✓ Taux court : FRED IR3TIB01EZM156N  "
              f"(→ {short_rate.dropna().index[-1].strftime('%Y-%m')})")

        # ── Croissance : eurostat ──
        gdp_yoy = _fetch_gdp_eurostat(self.start_date)

        # ── Assemblage ──
        df = pd.DataFrame({
            "inflation":  inflation,
            "long_rate":  long_rate,
            "short_rate": short_rate,
            "gdp_yoy":    gdp_yoy,
        })
        df["curve"] = df["long_rate"] - df["short_rate"]

        self._raw = df
        print("  Données prêtes.\n")
        return df

    # ── 2. Classification ────────────────────────────────────────────────────

    def detect(self) -> pd.DataFrame:
        """
        Applique la classification et retourne le DataFrame enrichi avec :
            regime, regime_color + les trois signaux lissés.
        """
        if self._raw is None:
            self.fetch()

        df = self._raw.copy()
        w  = self.smoothing_window

        # Lissage rolling : réduit les bascules dues au bruit mensuel
        df["inf_smooth"]    = df["inflation"].rolling(w, min_periods=1).mean()
        df["curve_smooth"]  = df["curve"].rolling(w, min_periods=1).mean()
        df["growth_smooth"] = df["gdp_yoy"].rolling(w, min_periods=1).mean()

        # On ne garde que les mois avec les trois signaux disponibles
        df = df.dropna(subset=["inf_smooth", "curve_smooth", "growth_smooth"])

        # Classification booléenne → clé du dictionnaire REGIME_LABELS
        df["inf_high"]        = df["inf_smooth"]    > self.inflation_threshold
        df["curve_normal"]    = df["curve_smooth"]  > self.curve_threshold
        df["growth_positive"] = df["growth_smooth"] > self.growth_threshold

        df["regime"] = df.apply(
            lambda r: REGIME_LABELS[(
                bool(r["inf_high"]),
                bool(r["curve_normal"]),
                bool(r["growth_positive"]),
            )],
            axis=1,
        )
        df["regime_color"] = df["regime"].map(REGIME_COLORS)

        self._regimes = df
        return df[[
            "inflation", "curve", "gdp_yoy",
            "inf_smooth", "curve_smooth", "growth_smooth",
            "inf_high", "curve_normal", "growth_positive",
            "regime", "regime_color",
        ]]

    # ── 3. Utilitaires ───────────────────────────────────────────────────────

    def regime_at(self, date: str) -> str:
        """Régime pour un mois donné. Ex : rd.regime_at('2022-06-01')"""
        if self._regimes is None:
            self.detect()
        loc = self._regimes.index.get_indexer(
            [pd.Timestamp(date)], method="nearest"
        )[0]
        return self._regimes.iloc[loc]["regime"]

    def regime_distribution(self) -> pd.DataFrame:
        """Fréquence de chaque régime sur la période, triée par occurrence."""
        if self._regimes is None:
            self.detect()
        dist = (
            self._regimes["regime"]
            .value_counts()
            .rename_axis("Régime")
            .reset_index(name="Mois")
        )
        dist["% du temps"] = (dist["Mois"] / dist["Mois"].sum() * 100).round(1)
        dist["Couleur"] = dist["Régime"].map(REGIME_COLORS)
        return dist

    def regime_periods(self) -> pd.DataFrame:
        """
        Blocs consécutifs par régime avec début, fin et durée en mois.
        C'est la méthode clé pour le backtester : chaque ligne = un épisode.
        """
        if self._regimes is None:
            self.detect()
        df = self._regimes[["regime"]].copy()
        df["block"] = (df["regime"] != df["regime"].shift()).cumsum()
        periods = (
            df.groupby("block", sort=False)
            .agg(
                regime   = ("regime", "first"),
                start    = ("regime", lambda x: x.index[0]),
                end      = ("regime", lambda x: x.index[-1]),
                duration = ("regime", "count"),
            )
            .reset_index(drop=True)
        )
        periods["color"] = periods["regime"].map(REGIME_COLORS)
        return periods

    def summary(self) -> None:
        """Résumé lisible dans le terminal."""
        if self._regimes is None:
            self.detect()
        n       = len(self._regimes)
        start   = self._regimes.index[0].strftime("%b %Y")
        end     = self._regimes.index[-1].strftime("%b %Y")
        current = self._regimes.iloc[-1]["regime"]

        print(f"\n{'─'*58}")
        print(f"  Euro Macro Regime Detector")
        print(f"  Période  : {start} → {end}  ({n} mois)")
        print(f"  Seuils   : inflation > {self.inflation_threshold}%  |  "
              f"courbe > {self.curve_threshold}%  |  croissance > {self.growth_threshold}%")
        print(f"  Actuel   : {current}")
        print(f"{'─'*58}")
        for _, row in self.regime_distribution().iterrows():
            bar = "█" * int(row["% du temps"] / 2)
            print(f"  {row['Régime']:<38} {bar}  {row['% du temps']}%")
        print(f"{'─'*58}\n")


# ─── Test rapide ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rd = RegimeDetector(start_date="2000-01-01")
    df = rd.detect()
    rd.summary()

    # Quelques vérifications rapides
    print("Régimes clés de mémoire :")
    for date, label in [
        ("2008-09-01", "Crise Lehman"),
        ("2011-07-01", "Crise dettes souveraines"),
        ("2020-04-01", "Covid"),
        ("2022-06-01", "Pic inflation post-Covid"),
    ]:
        try:
            print(f"  {label:<35} → {rd.regime_at(date)}")
        except Exception:
            pass

    print("\nDerniers régimes (12 mois) :")
    print(df[["inflation", "curve", "gdp_yoy", "regime"]].tail(12).to_string())

    print("\nDerniers épisodes :")
    print(rd.regime_periods().tail(8).to_string(index=False))
