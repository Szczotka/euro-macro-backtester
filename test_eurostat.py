"""
test_eurostat.py
────────────────
Lance ce script avant d'intégrer eurostat dans le projet.
Il vérifie que les deux datasets nécessaires sont accessibles
et affiche le format exact des colonnes (utile pour déboguer).

Usage :
    python test_eurostat.py
"""

import pandas as pd

try:
    import eurostat
except ImportError:
    print("✗ Package 'eurostat' introuvable. Lance : pip install eurostat")
    raise SystemExit(1)

print("\n── Test 1 : HICP (prc_hicp_midx) ────────────────────────")
try:
    df_hicp = eurostat.get_data_df("prc_hicp_midx", flags=False)

    # Identifier les colonnes de dimensions vs colonnes temporelles
    geo_col   = next(c for c in df_hicp.columns if "geo" in c.lower())
    time_cols = [c for c in df_hicp.columns if c not in df_hicp.columns[:6].tolist()]

    print(f"  Colonnes dimensions : {[c for c in df_hicp.columns if c not in time_cols]}")
    print(f"  Premières colonnes temporelles : {time_cols[:5]}")
    print(f"  Dernières colonnes temporelles : {time_cols[-5:]}")

    # Chercher une ligne pour EA (toutes variantes possibles)
    for geo in ["EA", "EA20", "EA19", "EA18"]:
        row = df_hicp[
            (df_hicp["unit"] == "I15")
            & (df_hicp["coicop"] == "CP00")
            & (df_hicp[geo_col] == geo)
        ]
        if not row.empty:
            last_val_col = time_cols[-1]
            print(f"  ✓ geo={geo} trouvé  |  dernière colonne : {last_val_col}")
            break
    else:
        print("  ✗ Aucune ligne trouvée pour EA/EA20/EA19")
        print(f"  Valeurs uniques de {geo_col} : {df_hicp[geo_col].unique()[:10]}")

except Exception as e:
    print(f"  ✗ Erreur : {e}")

print("\n── Test 2 : PIB (namq_10_gdp) ────────────────────────────")
try:
    df_gdp = eurostat.get_data_df("namq_10_gdp", flags=False)

    geo_col   = next(c for c in df_gdp.columns if "geo" in c.lower())
    time_cols = [c for c in df_gdp.columns if c not in df_gdp.columns[:6].tolist()]

    print(f"  Colonnes dimensions : {[c for c in df_gdp.columns if c not in time_cols]}")
    print(f"  Premières colonnes temporelles : {time_cols[:3]}")
    print(f"  Dernières colonnes temporelles : {time_cols[-3:]}")

    # Chercher une ligne pour EA, unit PCH_PREVC_PER
    for geo in ["EA", "EA20", "EA19"]:
        for unit in ["PCH_PREVC_PER", "CLV10_MEUR", "CP_MEUR"]:
            row = df_gdp[
                (df_gdp["na_item"] == "B1GQ")
                & (df_gdp["unit"] == unit)
                & (df_gdp["s_adj"] == "SCA")
                & (df_gdp[geo_col] == geo)
            ]
            if not row.empty:
                print(f"  ✓ geo={geo}, unit={unit}  |  dernière colonne : {time_cols[-1]}")
                break
        else:
            continue
        break
    else:
        print("  ✗ Aucune combinaison geo/unit trouvée")
        print(f"  Units disponibles : {df_gdp['unit'].unique()[:8]}")
        print(f"  Valeurs {geo_col} : {df_gdp[geo_col].unique()[:8]}")

except Exception as e:
    print(f"  ✗ Erreur : {e}")

print("\n─────────────────────────────────────────────────────────\n")
