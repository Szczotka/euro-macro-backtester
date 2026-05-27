"""
check_series.py
───────────────
Lance ce script UNE SEULE FOIS pour identifier quelles séries FRED
sont disponibles sur ton compte. Affiche ✓ ou ✗ pour chaque candidat.

Usage :
    python check_series.py
"""

import os
from fredapi import Fred
from dotenv import load_dotenv

load_dotenv()

fred = Fred(api_key=os.getenv("FRED_API_KEY"))

CANDIDATES = {
    "INFLATION (HICP zone euro)": [
        "CP0000EZ17M086NEST",   # ancien (17 pays)
        "CP0000EZ19M086NEST",   # élargi (19 pays)
        "CPALTT01EZM657N",      # OCDE MEI format
        "EA19CPHPTT01GYM",      # OCDE format alternatif
    ],
    "TAUX LONG (10 ans zone euro)": [
        "IRLTLT01EZM156N",      # BCE / OCDE
        "LTGBYEY10DSTM",        # format alternatif
    ],
    "TAUX COURT (3 mois zone euro)": [
        "IR3TIB01EZM156N",      # EURIBOR 3M
        "EURIBOR3MD",           # format alternatif
    ],
    "PIB REEL ZONE EURO": [
        "CLVMNACSCAB1GQEZ19EA", # 19 pays, niveau
        "CLVMNACSCAB1GQEZ17EA", # 17 pays, niveau
        "NAEXKP01EZQ657S",      # OCDE format
        "EUNNGDP",              # format simple
        "EA19GDPC",             # format alternatif
    ],
}

print("\n" + "─" * 58)
print("  Diagnostic séries FRED — Zone Euro")
print("─" * 58)

found: dict[str, str] = {}

for category, series_list in CANDIDATES.items():
    print(f"\n  {category}")
    category_found = False
    for sid in series_list:
        try:
            s = fred.get_series(sid, observation_start="2010-01-01", observation_end="2010-06-01")
            print(f"    ✓  {sid}  ({len(s)} obs. sur test)")
            if not category_found:
                found[category] = sid
                category_found = True
        except Exception:
            print(f"    ✗  {sid}")

print("\n" + "─" * 58)
print("  Séries retenues :")
for cat, sid in found.items():
    print(f"    {cat[:40]:<40} → {sid}")

if len(found) < 4:
    missing = [c for c in CANDIDATES if c not in found]
    print(f"\n  ⚠  Aucune série trouvée pour : {', '.join(missing)}")
    print("  → Copie ce message et partage-le pour qu'on adapte la source.")

print("─" * 58 + "\n")
