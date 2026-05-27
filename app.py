"""
app.py
======
Interface Streamlit du Euro Macro Backtester.

Lancement :
    streamlit run app.py

Dépendances :
    pip install streamlit plotly yfinance fredapi eurostat pandas numpy python-dotenv
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from backtester import compute_metrics
from regime_detector import RegimeDetector, REGIME_COLORS
from metrics import (
    ASSETS, load_prices, run_all_strategies,
    build_heatmap_df, build_comparison_df, get_equity_curves,
    build_correlation_matrix,
)
from walk_forward import WalkForwardTest

# ─── Configuration ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Euro Macro Backtester",
    page_icon="📊",
    layout="wide",
)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def color_sharpe(val):
    if pd.isna(val): return ""
    if val > 0.5:    return "color: #1D9E75; font-weight: 600"
    elif val > 0:    return "color: #5DCAA5"
    else:            return "color: #D85A30"

def color_cell(val):
    if pd.isna(val): return "color: #888888"
    if val > 0.6:    return "background-color: #1a5c38; color: #ffffff; font-weight: 600"
    elif val > 0.3:  return "background-color: #2e7d52; color: #ffffff"
    elif val > 0:    return "background-color: #5a9e78; color: #ffffff"
    elif val > -0.3: return "background-color: #c0392b; color: #ffffff"
    else:            return "background-color: #922b21; color: #ffffff; font-weight: 600"

def bold_best(s):
    is_max = s == s.dropna().max()
    return ["font-weight: 700" if v else "" for v in is_max]

REGIME_ORDER = [
    "Expansion saine", "Expansion fragile", "Désinflation douce",
    "Ralentissement inflationniste", "Stagflation naissante",
    "Surchauffe", "Stagflation", "Récession-désinflation",
]

# ─── Chargement global ────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_all_assets(start_date, inf_thr, curve_thr, growth_thr, tc_bps):
    """
    Unique point de chargement — calcule régimes + tous les backtests.
    L'onglet Analyse réutilise directement ce cache, aucun recalcul.
    """
    rd = RegimeDetector(
        start_date=start_date,
        inflation_threshold=inf_thr,
        curve_threshold=curve_thr,
        growth_threshold=growth_thr,
    )
    regime_df = rd.detect()
    periods   = rd.regime_periods()
    current_regime = regime_df["regime"].iloc[-1]

    summary_rows, all_results, all_prices = [], {}, {}

    for label, ticker in ASSETS.items():
        try:
            prices  = load_prices(ticker, start=start_date)
            results = run_all_strategies(prices, regime_df, tc_bps=tc_bps)
            comp    = build_comparison_df(results)

            bh_ret = prices.pct_change().dropna()
            bh = compute_metrics(
                bh_ret, label="Buy & Hold",
                invested_mask=pd.Series(True, index=bh_ret.index),
            )
            summary_rows.append({
                "Indice":             label,
                "Régime actuel":      current_regime,
                "Meilleure strat.":   comp.index[0],
                "Sharpe (meilleure)": round(comp.iloc[0]["Sharpe"], 3),
                "Rend. ann. (%)":     round(comp.iloc[0]["Rendement ann."], 2),
                "Max drawdown (%)":   round(comp.iloc[0]["Max drawdown"], 2),
                "Investi (%)":        round(comp.iloc[0]["Investi (%)"], 1),
                "BH Sharpe":          round(bh["sharpe"], 3),
                "BH Rend. (%)":       round(bh["rendement_ann"], 2),
                "BH MDD (%)":         round(bh["max_drawdown"], 2),
            })
            all_results[label] = results
            all_prices[label]  = prices

        except Exception as e:
            summary_rows.append({
                "Indice": label, "Régime actuel": current_regime,
                "Meilleure strat.": f"Erreur : {e}",
                **{k: np.nan for k in [
                    "Sharpe (meilleure)", "Rend. ann. (%)", "Max drawdown (%)",
                    "Investi (%)", "BH Sharpe", "BH Rend. (%)", "BH MDD (%)",
                ]},
            })
            all_results[label] = {}
            all_prices[label]  = pd.Series(dtype=float)

    summary_df = pd.DataFrame(summary_rows).set_index("Indice")
    return summary_df, all_results, all_prices, regime_df, periods


# ─── En-tête ──────────────────────────────────────────────────────────────────

st.title("Euro Macro Backtester")
st.caption(
    "Décompose les performances de 5 stratégies par régime macroéconomique (zone euro). "
    "Sources : eurostat · FRED · yfinance"
)

# ─── Onglets ──────────────────────────────────────────────────────────────────

tab_global, tab_analyse, tab_wf = st.tabs([
    "🌍 Vue globale", "📈 Analyse par actif", "🔬 Walk-Forward"
])


# ══════════════════════════════════════════════════════════════════════════════
# ONGLET 1 — Vue globale  (paramètres + chargement ici)
# ══════════════════════════════════════════════════════════════════════════════

with tab_global:

    # ── Paramètres ────────────────────────────────────────────────────────────
    st.markdown("#### Paramètres")
    p1, p2, p3, p4, p5 = st.columns(5)
    with p1:
        inf_threshold    = st.slider("Inflation élevée (% YoY)",      1.0, 6.0,  3.0, 0.5,  key="inf_threshold")
    with p2:
        curve_threshold  = st.slider("Courbe (spread 10Y–3M)",        -1.0, 2.0, 0.0, 0.25, key="curve_threshold")
    with p3:
        growth_threshold = st.slider("Croissance positive (% YoY)",   -1.0, 3.0, 0.5, 0.25, key="growth_threshold")
    with p4:
        tc_bps           = st.slider("Coûts de transaction (bps)",     0, 20,    5,   1,     key="tc_bps")
    with p5:
        start_date       = st.selectbox(
            "Début de période",
            ["2004-01-01", "2007-01-01", "2010-01-01"],
            index=0, key="start_date",
        )

    st.divider()

    # ── Chargement — unique appel pour toute l'app ────────────────────────────
    with st.spinner("Calcul de tous les indices… (30–60 s la première fois, puis mis en cache)"):
        summary_df, all_results, all_prices, regime_df_g, periods_g = load_all_assets(
            start_date, inf_threshold, curve_threshold, growth_threshold, tc_bps
        )

    last_inf    = regime_df_g["inflation"].dropna().iloc[-1]
    last_curve  = regime_df_g["curve"].dropna().iloc[-1]
    last_growth = regime_df_g["gdp_yoy"].dropna().iloc[-1]
    current_regime_g = regime_df_g["regime"].iloc[-1]
    current_date_g   = regime_df_g.index[-1].strftime("%B %Y")

    # ── 1. Tableau de synthèse ────────────────────────────────────────────────
    st.markdown("#### Meilleure stratégie par indice")
    st.caption("Comparaison de la meilleure stratégie (par Sharpe) vs buy-and-hold.")

    display_df = summary_df[[
        "Meilleure strat.", "Sharpe (meilleure)", "Rend. ann. (%)",
        "Max drawdown (%)", "Investi (%)", "BH Sharpe", "BH Rend. (%)", "BH MDD (%)",
    ]].copy()
    st.dataframe(
        display_df.style
        .format({
            "Sharpe (meilleure)": "{:.3f}", "Rend. ann. (%)": "{:.2f}%",
            "Max drawdown (%)": "{:.2f}%",  "Investi (%)": "{:.1f}%",
            "BH Sharpe": "{:.3f}",          "BH Rend. (%)": "{:.2f}%",
            "BH MDD (%)": "{:.2f}%",
        }, na_rep="—")
        .map(color_sharpe, subset=["Sharpe (meilleure)", "BH Sharpe"]),
        use_container_width=True, height=240,
    )

    st.divider()

    # ── 2. Bar chart Sharpe ───────────────────────────────────────────────────
    st.markdown("#### Sharpe : meilleure stratégie vs Buy & Hold")
    bar_data = []
    for indice, row in summary_df.iterrows():
        bar_data.append({"Indice": indice, "Type": "Meilleure stratégie", "Sharpe": row["Sharpe (meilleure)"]})
        bar_data.append({"Indice": indice, "Type": "Buy & Hold",          "Sharpe": row["BH Sharpe"]})

    fig_bar_g = px.bar(
        pd.DataFrame(bar_data), x="Indice", y="Sharpe",
        color="Type", barmode="group", text_auto=".2f",
        color_discrete_map={"Meilleure stratégie": "#1D9E75", "Buy & Hold": "#85B7EB"},
    )
    fig_bar_g.add_hline(y=0, line_dash="dot", line_color="gray", line_width=1)
    fig_bar_g.update_layout(
        height=340, margin=dict(l=0, r=0, t=10, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        yaxis_title="Sharpe", xaxis_title="",
    )
    st.plotly_chart(fig_bar_g, use_container_width=True, key="g_bar_sharpe")

    st.divider()

    # ── 3. Heatmap multi-indices ──────────────────────────────────────────────
    st.markdown("#### Heatmap : Sharpe par régime — tous indices")
    st.caption("Sharpe de la meilleure stratégie de chaque indice, décomposé par régime.")

    heatmap_rows = []
    for label, results_i in all_results.items():
        if not results_i: continue
        comp_i = build_comparison_df(results_i)
        if comp_i.empty: continue
        for _, rm in results_i[comp_i.index[0]].by_regime().reset_index().iterrows():
            heatmap_rows.append({"Indice": label, "Régime": rm["Régime"], "sharpe": rm["sharpe"]})

    if heatmap_rows:
        hm_multi = pd.DataFrame(heatmap_rows).pivot(index="Indice", columns="Régime", values="sharpe")
        ordered = [r for r in REGIME_ORDER if r in hm_multi.columns]
        ordered += [c for c in hm_multi.columns if c not in REGIME_ORDER]
        fig_hm_g = px.imshow(
            hm_multi[ordered], color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0, text_auto=".2f", aspect="auto",
            labels={"x": "Régime", "y": "Indice", "color": "Sharpe"},
        )
        fig_hm_g.update_layout(
            height=280, margin=dict(l=0, r=0, t=10, b=0),
            font=dict(size=12), coloraxis_colorbar=dict(title="Sharpe", thickness=12),
            xaxis_tickangle=-30,
        )
        fig_hm_g.update_traces(textfont_size=11)
        st.plotly_chart(fig_hm_g, use_container_width=True, key="g_hm_multi")

    st.divider()

    # ── 4. Quelle stratégie par régime ────────────────────────────────────────
    st.markdown("#### Quelle stratégie adopter selon le régime ?")
    st.caption("Sharpe moyen pondéré par le nombre de mois, tous indices confondus.")

    all_regime_rows = []
    for label, results_i in all_results.items():
        if not results_i: continue
        for strat_name, result in results_i.items():
            for _, rm in result.by_regime().reset_index().iterrows():
                all_regime_rows.append({
                    "Régime": rm["Régime"], "Stratégie": strat_name,
                    "Sharpe": rm["sharpe"], "Mois": rm["n_mois_approx"],
                })

    if all_regime_rows:
        regime_strat = (
            pd.DataFrame(all_regime_rows).dropna(subset=["Sharpe"])
            .groupby(["Régime", "Stratégie"])
            .apply(lambda g: np.average(g["Sharpe"], weights=g["Mois"]))
            .rename("Sharpe moyen").reset_index()
        )
        best_per_regime = (
            regime_strat.sort_values("Sharpe moyen", ascending=False)
            .groupby("Régime", sort=False).first().reset_index()
        )
        best_per_regime["color"]  = best_per_regime["Régime"].map(REGIME_COLORS)
        best_per_regime["_order"] = best_per_regime["Régime"].map(
            {r: i for i, r in enumerate(REGIME_ORDER)}
        ).fillna(99)
        best_per_regime = best_per_regime.sort_values("_order").drop(columns="_order")

        for row_start in range(0, len(best_per_regime), 4):
            cols = st.columns(4)
            for col_idx, record in enumerate(best_per_regime.iloc[row_start:row_start+4].to_dict("records")):
                sharpe = record["Sharpe moyen"]
                sharpe_str   = f"{sharpe:+.2f}" if not np.isnan(sharpe) else "—"
                sharpe_color = "#1D9E75" if sharpe > 0 else "#D85A30"
                with cols[col_idx]:
                    st.markdown(
                        f"""<div style="border-left:4px solid {record['color']};
                            background:#1e1e1e;border-radius:8px;
                            padding:14px 16px;margin-bottom:10px;">
                            <div style="font-size:12px;color:#aaaaaa;margin-bottom:4px;">
                                {record['Régime']}</div>
                            <div style="font-size:15px;font-weight:600;color:#ffffff;margin-bottom:6px;">
                                {record['Stratégie']}</div>
                            <div style="font-size:13px;color:{sharpe_color};font-weight:500;">
                                Sharpe moy. {sharpe_str}</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )

        st.divider()
        st.markdown("##### Classement complet stratégies × régimes")
        pivot_best = regime_strat.pivot(index="Stratégie", columns="Régime", values="Sharpe moyen")
        cols_ord = [r for r in REGIME_ORDER if r in pivot_best.columns]
        cols_ord += [c for c in pivot_best.columns if c not in REGIME_ORDER]
        st.dataframe(
            pivot_best[cols_ord].round(3).style
            .format("{:.2f}", na_rep="—")
            .map(color_cell)
            .apply(bold_best, axis=0),
            use_container_width=True, height=230,
        )

    st.divider()

    # ── 5. Timeline ───────────────────────────────────────────────────────────
    st.markdown("#### Timeline des régimes macro — zone euro")
    fig_tl_g = px.timeline(
        pd.DataFrame([{
            "Régime": p["regime"], "Start": p["start"],
            "End": p["end"] + pd.DateOffset(months=1), "Durée": f"{p['duration']} mois",
        } for _, p in periods_g.iterrows()]),
        x_start="Start", x_end="End", y="Régime",
        color="Régime", color_discrete_map=REGIME_COLORS, hover_data=["Durée"],
    )
    fig_tl_g.update_yaxes(autorange="reversed")
    fig_tl_g.update_layout(
        height=340, margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False, xaxis_title="", yaxis_title="",
    )
    st.plotly_chart(fig_tl_g, use_container_width=True, key="g_timeline")

    st.divider()

    # ── Conclusion : situation actuelle et recommandations ────────────────────

    regime_color_now = REGIME_COLORS.get(current_regime_g, "#888888")

    st.markdown(
        f"""
        <div style="
            border-left: 5px solid {regime_color_now};
            background: #1e1e1e;
            border-radius: 10px;
            padding: 20px 24px 16px;
            margin-bottom: 20px;
        ">
            <div style="font-size:12px; color:#aaaaaa; margin-bottom:6px; letter-spacing:.05em;">
                SITUATION MACRO — {current_date_g.upper()}
            </div>
            <div style="font-size:22px; font-weight:700; color:#ffffff; margin-bottom:12px;">
                Régime actuel : {current_regime_g}
            </div>
            <div style="display:flex; gap:32px;">
                <div>
                    <div style="font-size:11px;color:#aaaaaa;">Inflation HICP</div>
                    <div style="font-size:18px;font-weight:600;color:#ffffff;">{last_inf:.1f}%</div>
                </div>
                <div>
                    <div style="font-size:11px;color:#aaaaaa;">Courbe 10Y–3M</div>
                    <div style="font-size:18px;font-weight:600;color:#ffffff;">{last_curve:.2f}%</div>
                </div>
                <div>
                    <div style="font-size:11px;color:#aaaaaa;">Croissance PIB</div>
                    <div style="font-size:18px;font-weight:600;color:#ffffff;">{last_growth:.1f}%</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if all_regime_rows:
        current_regime_strat = regime_strat[
            regime_strat["Régime"] == current_regime_g
        ].sort_values("Sharpe moyen", ascending=False)

        index_regime_rows = []
        for label, results_i in all_results.items():
            if not results_i:
                continue
            for strat_name, result in results_i.items():
                regime_breakdown = result.by_regime()
                if current_regime_g in regime_breakdown.index:
                    row = regime_breakdown.loc[current_regime_g]
                    index_regime_rows.append({
                        "Indice":    label,
                        "Stratégie": strat_name,
                        "Sharpe":    row["sharpe"],
                        "Rend. (%)": row["rendement_ann"],
                        "MDD (%)":   row["max_drawdown"],
                        "Mois":      row["n_mois_approx"],
                    })

        index_best_df = (
            pd.DataFrame(index_regime_rows)
            .dropna(subset=["Sharpe"])
            .sort_values("Sharpe", ascending=False)
            .groupby("Indice", sort=False)
            .first()
            .reset_index()
            .sort_values("Sharpe", ascending=False)
        )

        col_strat, col_indices = st.columns([1, 1])

        with col_strat:
            st.markdown("##### Stratégie recommandée")
            st.caption(f"Classement par Sharpe moyen pondéré dans le régime « {current_regime_g} »")

            if not current_regime_strat.empty:
                for rank, (_, srow) in enumerate(current_regime_strat.iterrows(), 1):
                    sharpe  = srow["Sharpe moyen"]
                    is_best = rank == 1
                    bar_w   = max(5, int(min(abs(sharpe) / 1.5 * 100, 100)))
                    bar_col = "#1D9E75" if sharpe > 0 else "#c0392b"
                    border  = f"border: 1.5px solid {regime_color_now};" if is_best else ""
                    badge   = "<span style=\'font-size:10px;background:#D4A017;color:#000;border-radius:4px;padding:1px 6px;margin-left:8px;\'>★ Recommandé</span>" if is_best else ""
                    sharpe_color = "#1D9E75" if sharpe > 0 else "#D85A30"
                    st.markdown(
                        f"""<div style="background:#1e1e1e;border-radius:8px;padding:12px 14px;
                                margin-bottom:8px;{border}">
                            <div style="display:flex;justify-content:space-between;align-items:center;">
                                <span style="font-size:14px;font-weight:{'700' if is_best else '400'};
                                    color:#ffffff;">{rank}. {srow["Stratégie"]}{badge}</span>
                                <span style="font-size:14px;font-weight:600;color:{sharpe_color};">
                                    Sharpe {sharpe:+.2f}</span>
                            </div>
                            <div style="margin-top:8px;background:#333;border-radius:4px;height:5px;">
                                <div style="width:{bar_w}%;background:{bar_col};
                                    border-radius:4px;height:5px;"></div>
                            </div>
                        </div>""",
                        unsafe_allow_html=True,
                    )

        with col_indices:
            st.markdown("##### Classement des indices")
            st.caption(f"Sharpe de la meilleure stratégie par indice dans le régime « {current_regime_g} »")

            if not index_best_df.empty:
                max_sharpe = index_best_df["Sharpe"].abs().max()
                for rank, (_, irow) in enumerate(index_best_df.iterrows(), 1):
                    sharpe  = irow["Sharpe"]
                    is_best = rank == 1
                    bar_w   = max(5, int(abs(sharpe) / max(max_sharpe, 0.01) * 100))
                    bar_col = "#1D9E75" if sharpe > 0 else "#c0392b"
                    border  = f"border: 1.5px solid {regime_color_now};" if is_best else ""
                    badge   = "<span style=\'font-size:10px;background:#D4A017;color:#000;border-radius:4px;padding:1px 6px;margin-left:8px;\'>★ À privilégier</span>" if is_best else ""
                    sharpe_color = "#1D9E75" if sharpe > 0 else "#D85A30"
                    st.markdown(
                        f"""<div style="background:#1e1e1e;border-radius:8px;padding:12px 14px;
                                margin-bottom:8px;{border}">
                            <div style="display:flex;justify-content:space-between;align-items:center;">
                                <div>
                                    <span style="font-size:14px;font-weight:{'700' if is_best else '400'};
                                        color:#ffffff;">{rank}. {irow["Indice"]}{badge}</span>
                                    <div style="font-size:11px;color:#aaaaaa;margin-top:2px;">
                                        via {irow["Stratégie"]}
                                        · Rend. {irow["Rend. (%)"]:+.1f}%
                                        · MDD {irow["MDD (%)"]:+.1f}%
                                        · {int(irow["Mois"])} mois de données
                                    </div>
                                </div>
                                <span style="font-size:14px;font-weight:600;color:{sharpe_color};">
                                    {sharpe:+.2f}</span>
                            </div>
                            <div style="margin-top:8px;background:#333;border-radius:4px;height:5px;">
                                <div style="width:{bar_w}%;background:{bar_col};
                                    border-radius:4px;height:5px;"></div>
                            </div>
                        </div>""",
                        unsafe_allow_html=True,
                    )


# ══════════════════════════════════════════════════════════════════════════════
# ONGLET 2 — Analyse par actif  (aucun recalcul — lecture du cache global)
# ══════════════════════════════════════════════════════════════════════════════

with tab_analyse:

    # ── Sélection de l'actif ──────────────────────────────────────────────────
    _inf    = st.session_state.get("inf_threshold",    3.0)
    _curve  = st.session_state.get("curve_threshold",  0.0)
    _growth = st.session_state.get("growth_threshold", 0.5)
    _tc     = st.session_state.get("tc_bps",           5)
    _start  = st.session_state.get("start_date",       "2004-01-01")

    col_sel, col_info = st.columns([2, 3])
    with col_sel:
        asset_label = st.selectbox("Actif à analyser", list(ASSETS.keys()), index=0)
    with col_info:
        st.caption(
            f"Paramètres actifs : inflation > {_inf}%  |  courbe > {_curve}%  |  "
            f"croissance > {_growth}%  |  coûts : {_tc} bps  |  depuis {_start}  \n"
            f"*(modifiables dans l'onglet Vue globale)*"
        )

    st.divider()

    # ── Lecture directe du cache global — zéro recalcul ──────────────────────
    results  = all_results.get(asset_label, {})
    periods  = periods_g
    regime_df = regime_df_g

    if not results:
        st.error(f"Données indisponibles pour {asset_label}. Vérifie le ticker dans metrics.py.")
        st.stop()

    # Métriques macro
    current_regime = regime_df["regime"].iloc[-1]
    current_date   = regime_df.index[-1].strftime("%B %Y")
    last_inf_a    = regime_df["inflation"].dropna().iloc[-1]
    last_curve_a  = regime_df["curve"].dropna().iloc[-1]
    last_growth_a = regime_df["gdp_yoy"].dropna().iloc[-1]

    col_r, col_i, col_c, col_g = st.columns(4)
    col_r.metric("Régime actuel",  current_regime,          help=f"Au {current_date}")
    col_i.metric("Inflation HICP", f"{last_inf_a:.1f}%",    help="Glissement annuel zone euro")
    col_c.metric("Courbe 10Y–3M",  f"{last_curve_a:.2f}%",  help="Spread taux longs – taux courts")
    col_g.metric("Croissance PIB", f"{last_growth_a:.1f}%", help="PIB réel YoY zone euro")

    st.divider()

    # ── Heatmap ───────────────────────────────────────────────────────────────
    st.subheader("Heatmap : Sharpe par stratégie et par régime")
    st.caption("Chaque cellule = ratio de Sharpe de la stratégie pendant les périodes de ce régime macro.")

    heatmap_metric = st.radio(
        "Métrique",
        ["sharpe", "rendement_ann", "max_drawdown"],
        format_func=lambda x: {
            "sharpe": "Sharpe", "rendement_ann": "Rendement ann. (%)", "max_drawdown": "Max drawdown (%)",
        }[x],
        horizontal=True, key="a_hm_metric",
    )

    heatmap_df = build_heatmap_df(results, metric=heatmap_metric)
    if not heatmap_df.empty:
        colorscale = "RdYlGn" if heatmap_metric != "max_drawdown" else "RdYlGn_r"
        zmid = 0 if heatmap_metric in ("sharpe", "rendement_ann") else None
        fig_heat = px.imshow(
            heatmap_df, color_continuous_scale=colorscale,
            color_continuous_midpoint=zmid, text_auto=".2f", aspect="auto",
            labels={"x": "Régime", "y": "Stratégie", "color": heatmap_metric},
        )
        fig_heat.update_layout(
            height=320, margin=dict(l=0, r=0, t=20, b=0),
            font=dict(size=13), coloraxis_colorbar=dict(title="", thickness=12),
            xaxis_tickangle=-30,
        )
        fig_heat.update_traces(textfont_size=12)
        st.plotly_chart(fig_heat, use_container_width=True, key="a_heatmap")

    st.divider()

    # ── Equity curves ─────────────────────────────────────────────────────────
    st.subheader("Equity curves comparées (base 100)")
    st.caption("Les bandes colorées en arrière-plan = régimes macroéconomiques.")

    equity_df = get_equity_curves(results)
    fig_eq = go.Figure()
    for _, period in periods.iterrows():
        if period["start"] < equity_df.index[-1] and period["end"] > equity_df.index[0]:
            fig_eq.add_vrect(
                x0=max(period["start"], equity_df.index[0]),
                x1=min(period["end"],   equity_df.index[-1]),
                fillcolor=period["color"], opacity=0.12,
                layer="below", line_width=0,
                annotation_text="" if period["duration"] < 6 else period["regime"][:12],
                annotation_position="top left",
                annotation_font_size=9, annotation_font_color="#888",
            )
    colors = px.colors.qualitative.Plotly
    for i, col in enumerate(equity_df.columns):
        fig_eq.add_trace(go.Scatter(
            x=equity_df.index, y=equity_df[col], name=col,
            line=dict(width=1.8, color=colors[i % len(colors)]),
            hovertemplate=f"<b>{col}</b><br>%{{x|%b %Y}}: %{{y:.1f}}<extra></extra>",
        ))
    fig_eq.update_layout(
        height=420, margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        hovermode="x unified", yaxis_title="Valeur (base 100)", xaxis_title="",
    )
    st.plotly_chart(fig_eq, use_container_width=True, key="a_equity")

    st.divider()

    # ── Tableau métriques ─────────────────────────────────────────────────────
    st.subheader("Métriques globales par stratégie")
    comp_df = build_comparison_df(results)
    st.dataframe(
        comp_df.style
        .format({
            "Sharpe": "{:.3f}", "Rendement ann.": "{:.2f}%", "Vol. ann.": "{:.2f}%",
            "Max drawdown": "{:.2f}%", "Investi (%)": "{:.1f}%", "Win rate (%)": "{:.1f}%",
        })
        .map(color_sharpe, subset=["Sharpe"]),
        use_container_width=True, height=240,
    )

    st.divider()

    # ── Matrice de corrélation des stratégies ─────────────────────────────────

    st.subheader("Corrélation des stratégies")
    st.caption(
        "Corrélation des rendements journaliers entre stratégies. "
        "Une valeur proche de **0 ou négative** indique des stratégies complémentaires "
        "— les combiner dans un portefeuille réduit la volatilité globale. "
        "Proche de **1** : elles réagissent de la même façon, pas de bénéfice à les combiner."
    )

    corr_df = build_correlation_matrix(results)

    fig_corr = px.imshow(
        corr_df,
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,
        zmin=-1, zmax=1,
        text_auto=".2f",
        aspect="auto",
        labels={"color": "Corrélation"},
    )
    fig_corr.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=10, b=0),
        font=dict(size=12),
        coloraxis_colorbar=dict(title="", thickness=12, tickvals=[-1, -0.5, 0, 0.5, 1]),
        xaxis_tickangle=-30,
    )
    fig_corr.update_traces(textfont_size=13)
    st.plotly_chart(fig_corr, use_container_width=True, key="a_corr")

    # Identifier les paires les plus décorrélées pour guider l'interprétation
    corr_pairs = []
    strats = corr_df.columns.tolist()
    for i in range(len(strats)):
        for j in range(i + 1, len(strats)):
            corr_pairs.append((strats[i], strats[j], corr_df.iloc[i, j]))
    corr_pairs.sort(key=lambda x: abs(x[2]))

    if corr_pairs:
        best_pair = corr_pairs[0]
        worst_pair = corr_pairs[-1]
        st.caption(
            f"**Paire la plus complémentaire** : {best_pair[0]} × {best_pair[1]} "
            f"(corrélation : {best_pair[2]:+.2f}) — "
            f"les combiner maximise la diversification.   "
            f"**Paire la plus redondante** : {worst_pair[0]} × {worst_pair[1]} "
            f"(corrélation : {worst_pair[2]:+.2f})."
        )

    st.divider()

    # ── Détail stratégie ──────────────────────────────────────────────────────
    st.subheader("Détail par régime — une stratégie")
    selected = st.selectbox("Choisir une stratégie", list(results.keys()), index=2)
    detail   = results[selected].by_regime()

    col_left, col_right = st.columns([3, 2])
    with col_left:
        fig_bar_a = px.bar(
            detail.reset_index(), x="Régime", y="sharpe",
            color="sharpe", color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0, text="sharpe",
            labels={"sharpe": "Sharpe", "Régime": ""},
            title=f"Sharpe par régime — {selected}",
        )
        fig_bar_a.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig_bar_a.update_layout(
            height=360, showlegend=False, margin=dict(l=0, r=0, t=40, b=0),
            coloraxis_showscale=False, xaxis_tickangle=-35,
        )
        fig_bar_a.add_hline(y=0, line_dash="dot", line_color="gray", line_width=1)
        st.plotly_chart(fig_bar_a, use_container_width=True, key="a_bar_regime")

    with col_right:
        st.dataframe(
            detail[["n_mois_approx", "sharpe", "rendement_ann", "max_drawdown"]]
            .rename(columns={
                "n_mois_approx": "Mois", "sharpe": "Sharpe",
                "rendement_ann": "Rend. (%)", "max_drawdown": "MDD (%)",
            })
            .style.format({"Sharpe": "{:.2f}", "Rend. (%)": "{:.1f}", "MDD (%)": "{:.1f}"})
            .map(color_sharpe, subset=["Sharpe"]),
            use_container_width=True, height=340,
        )

    st.divider()

    # ── Timeline ──────────────────────────────────────────────────────────────
    st.subheader("Timeline des régimes macro — zone euro")
    fig_tl_a = px.timeline(
        pd.DataFrame([{
            "Régime": p["regime"], "Start": p["start"],
            "End": p["end"] + pd.DateOffset(months=1), "Durée": f"{p['duration']} mois",
        } for _, p in periods.iterrows()]),
        x_start="Start", x_end="End", y="Régime",
        color="Régime", color_discrete_map=REGIME_COLORS, hover_data=["Durée"],
    )
    fig_tl_a.update_yaxes(autorange="reversed")
    fig_tl_a.update_layout(
        height=340, margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False, xaxis_title="", yaxis_title="",
    )
    st.plotly_chart(fig_tl_a, use_container_width=True, key="a_timeline")
    st.caption("Sources : eurostat · FRED · yfinance")


# ══════════════════════════════════════════════════════════════════════════════
# ONGLET 3 — Walk-Forward
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def run_walk_forward_cached(
    ticker, start_date, inf_thr, curve_thr, growth_thr,
    tc_bps, calibration_years, test_years,
):
    """
    Wrapper cacheable autour de WalkForwardTest.

    On n'expose que des DataFrames et Series à st.cache_data
    (pas d'objets personnalisés imbriqués).
    """
    rd = RegimeDetector(
        start_date=start_date,
        inflation_threshold=inf_thr,
        curve_threshold=curve_thr,
        growth_threshold=growth_thr,
    )
    regime_df = rd.detect()
    prices    = load_prices(ticker, start=start_date)

    wf = WalkForwardTest(
        regime_df=regime_df,
        prices=prices,
        asset_name=ticker,
        calibration_years=calibration_years,
        test_years=test_years,
        tc_bps=tc_bps,
        verbose=False,
    )
    result = wf.run()

    # Extraire uniquement les structures sérialisables
    return {
        "is_oos_comparison": result.is_oos_comparison(),
        "strategy_stability": result.strategy_stability(),
        "oos_by_regime":     result.oos_by_regime(),
        "returns_oos":       result.returns_oos,
        "returns_is_full":   result.returns_is_full,
        "oos_metrics":       result.oos_metrics(),
        "n_windows":         len(result.window_results),
        "oos_start":         result.window_results[0].test_start,
        "oos_end":           result.window_results[-1].test_end,
    }


with tab_wf:

    # ── En-tête explicatif ────────────────────────────────────────────────────

    st.markdown("#### Qu'est-ce que le walk-forward test ?")
    st.caption(
        "Le backtest classique calibre et teste la stratégie sur la **même période**. "
        "C'est comme réviser avec les réponses sous les yeux. "
        "Le walk-forward corrige ça : on calibre sur le passé, on teste sur une période "
        "que le modèle n'a jamais vue. Si les résultats tiennent, le modèle est robuste."
    )

    st.divider()

    # ── Paramètres ────────────────────────────────────────────────────────────

    st.markdown("#### Paramètres")
    wf_c1, wf_c2, wf_c3, wf_c4 = st.columns([2, 1, 1, 1])
    with wf_c1:
        wf_asset_label = st.selectbox(
            "Actif", list(ASSETS.keys()), index=0, key="wf_asset"
        )
        wf_ticker = ASSETS[wf_asset_label]
    with wf_c2:
        wf_cal_years = st.slider(
            "Calibration (ans)", 5, 12, 8, 1, key="wf_cal_years",
            help="Durée de la première fenêtre d'entraînement. Min. 5 ans pour capturer un cycle complet."
        )
    with wf_c3:
        wf_test_years = st.slider(
            "Test (ans)", 1, 3, 1, 1, key="wf_test_years",
            help="Durée de chaque fenêtre de test. 1 an = résultats annuels, faciles à interpréter."
        )
    with wf_c4:
        # Lire les paramètres macro depuis Vue globale
        _inf    = st.session_state.get("inf_threshold",    3.0)
        _curve  = st.session_state.get("curve_threshold",  0.0)
        _growth = st.session_state.get("growth_threshold", 0.5)
        _tc     = st.session_state.get("tc_bps",           5)
        _start  = st.session_state.get("start_date",       "2004-01-01")
        st.caption(
            f"Seuils macro hérités de Vue globale :\n"
            f"inflation > {_inf}%  ·  courbe > {_curve}%  ·  croissance > {_growth}%\n"
            f"Coûts : {_tc} bps"
        )

    # Estimation du nombre de fenêtres
    from walk_forward import generate_windows
    estimated_windows = generate_windows(
        _start, "2026-01-01", wf_cal_years, wf_test_years
    )
    n_est = len(estimated_windows)
    st.caption(
        f"→ {n_est} fenêtres prévues  ·  "
        f"période OOS estimée : "
        f"{estimated_windows[0]['test_start'].strftime('%Y-%m') if estimated_windows else '—'}"
        f" → "
        f"{estimated_windows[-1]['test_end'].strftime('%Y-%m') if estimated_windows else '—'}"
        f"  ·  durée estimée : {n_est * 8 // 60} min {n_est * 8 % 60} s"
    )

    st.divider()

    # ── Bouton de lancement ───────────────────────────────────────────────────

    launch = st.button(
        f"▶  Lancer le walk-forward sur {wf_asset_label}",
        type="primary",
        use_container_width=True,
    )

    if launch or st.session_state.get("wf_ran"):
        st.session_state["wf_ran"] = True

        with st.spinner(f"Calcul en cours… {n_est} fenêtres × 5 stratégies (~{n_est * 8 // 60} min {n_est * 8 % 60} s)"):
            try:
                wf_data = run_walk_forward_cached(
                    wf_ticker, _start, _inf, _curve, _growth,
                    _tc, wf_cal_years, wf_test_years,
                )
            except Exception as e:
                st.error(f"Erreur lors du walk-forward : {e}")
                st.stop()

        comp     = wf_data["is_oos_comparison"]
        stab     = wf_data["strategy_stability"]
        by_reg   = wf_data["oos_by_regime"]
        oos_ret  = wf_data["returns_oos"]
        is_ret   = wf_data["returns_is_full"]
        oos_m    = wf_data["oos_metrics"]
        n_win    = wf_data["n_windows"]
        oos_start_dt = wf_data["oos_start"]
        oos_end_dt   = wf_data["oos_end"]

        # ── Métriques globales ────────────────────────────────────────────────

        st.subheader("Résultats out-of-sample")

        win_rate = (stab["Sharpe OOS"] > 0).mean() * 100
        mean_oos = comp["Sharpe OOS"].max()
        degradation = comp["Dégradation"].mean()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "Fenêtres OOS positives",
            f"{win_rate:.0f}%",
            help="% de fenêtres annuelles où la stratégie choisie gagne de l'argent. Référence : ~50% au hasard.",
        )
        m2.metric(
            "Meilleur Sharpe OOS",
            f"{mean_oos:.3f}",
            help="Sharpe de la stratégie la plus performante sur la période OOS agrégée.",
        )
        m3.metric(
            "Dégradation IS→OOS",
            f"{degradation:+.3f}",
            delta_color="inverse",
            help="Perte de Sharpe moyenne entre in-sample et out-of-sample. Normale entre -0.10 et -0.30.",
        )
        m4.metric(
            "Fenêtres testées",
            n_win,
            help=f"Période OOS couverte : {oos_start_dt.strftime('%Y-%m')} → {oos_end_dt.strftime('%Y-%m')}",
        )

        st.divider()

        # ── Tableau IS vs OOS ─────────────────────────────────────────────────

        st.markdown("#### Sharpe IS vs OOS par stratégie")
        st.caption(
            "Le Sharpe IS est calculé sur les données d'entraînement de chaque fenêtre. "
            "Le Sharpe OOS est le résultat réel sur la période de test. "
            "Une dégradation faible (< 0.20) indique un modèle robuste."
        )

        def color_degradation(val):
            if pd.isna(val): return ""
            if val > -0.15:  return "color: #1D9E75; font-weight: 600"
            elif val > -0.35: return "color: #EF9F27"
            else:             return "color: #D85A30; font-weight: 600"

        def color_robust(val):
            if "Oui" in str(val):     return "color: #1D9E75; font-weight: 600"
            elif "Partiel" in str(val): return "color: #EF9F27"
            else:                       return "color: #D85A30"

        st.dataframe(
            comp.style
            .format({"Sharpe IS": "{:.3f}", "Sharpe OOS": "{:.3f}", "Dégradation": "{:+.3f}"})
            .map(color_sharpe,      subset=["Sharpe IS", "Sharpe OOS"])
            .map(color_degradation, subset=["Dégradation"])
            .map(color_robust,      subset=["Robuste ?"]),
            use_container_width=True,
            height=220,
        )

        st.divider()

        # ── Equity curves IS vs OOS ───────────────────────────────────────────

        st.markdown("#### Equity curves : in-sample vs out-of-sample")
        st.caption(
            "Les deux courbes sont normalisées à 100 au début de la période OOS. "
            "La courbe IS (bleue) représente ce que le backtest classique aurait affiché. "
            "La courbe OOS (verte) représente la performance réelle hors-échantillon."
        )

        if len(oos_ret) > 0 and len(is_ret) > 0:
            # Aligner les deux séries sur la même période OOS et normaliser à 100
            common_start = max(oos_ret.index[0], is_ret.index[0])
            oos_eq = (1 + oos_ret.loc[common_start:]).cumprod() * 100
            is_eq  = (1 + is_ret.loc[common_start:]).cumprod() * 100

            fig_wf_eq = go.Figure()
            fig_wf_eq.add_trace(go.Scatter(
                x=is_eq.index,  y=is_eq,
                name="In-sample (backtest classique)",
                line=dict(width=1.8, color="#85B7EB", dash="dot"),
                hovertemplate="IS %{x|%b %Y}: %{y:.1f}<extra></extra>",
            ))
            fig_wf_eq.add_trace(go.Scatter(
                x=oos_eq.index, y=oos_eq,
                name="Out-of-sample (walk-forward)",
                line=dict(width=2.2, color="#1D9E75"),
                hovertemplate="OOS %{x|%b %Y}: %{y:.1f}<extra></extra>",
                fill="tozeroy", fillcolor="rgba(29,158,117,0.05)",
            ))
            fig_wf_eq.add_hline(y=100, line_dash="dot", line_color="gray", line_width=1)
            fig_wf_eq.update_layout(
                height=380,
                margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
                hovermode="x unified",
                yaxis_title="Valeur (base 100)",
                xaxis_title="",
            )
            st.plotly_chart(fig_wf_eq, use_container_width=True, key="wf_equity")

        st.divider()

        # ── Performance OOS par régime ────────────────────────────────────────

        st.markdown("#### Performance OOS par régime")
        st.caption(
            "Décomposition du Sharpe out-of-sample selon le régime macro en vigueur "
            "pendant la période de test. Valide que la logique macro tient hors-échantillon."
        )

        if not by_reg.empty:
            col_br_left, col_br_right = st.columns([3, 2])

            with col_br_left:
                fig_oos_reg = px.bar(
                    by_reg.reset_index(),
                    x="Régime", y="sharpe",
                    color="sharpe",
                    color_continuous_scale="RdYlGn",
                    color_continuous_midpoint=0,
                    text="sharpe",
                    labels={"sharpe": "Sharpe OOS", "Régime": ""},
                    title="Sharpe OOS par régime",
                )
                fig_oos_reg.update_traces(texttemplate="%{text:.2f}", textposition="outside")
                fig_oos_reg.update_layout(
                    height=340, showlegend=False,
                    margin=dict(l=0, r=0, t=40, b=0),
                    coloraxis_showscale=False, xaxis_tickangle=-35,
                )
                fig_oos_reg.add_hline(y=0, line_dash="dot", line_color="gray", line_width=1)
                st.plotly_chart(fig_oos_reg, use_container_width=True, key="wf_by_regime")

            with col_br_right:
                display_cols = ["n_mois_approx", "sharpe", "rendement_ann", "max_drawdown"]
                available    = [c for c in display_cols if c in by_reg.columns]
                st.dataframe(
                    by_reg[available]
                    .rename(columns={
                        "n_mois_approx": "Mois OOS",
                        "sharpe":        "Sharpe",
                        "rendement_ann": "Rend. (%)",
                        "max_drawdown":  "MDD (%)",
                    })
                    .style.format({
                        "Sharpe": "{:.2f}", "Rend. (%)": "{:.1f}", "MDD (%)": "{:.1f}",
                    })
                    .map(color_sharpe, subset=["Sharpe"]),
                    use_container_width=True,
                    height=320,
                )

        st.divider()

        # ── Stabilité fenêtre par fenêtre ─────────────────────────────────────

        st.markdown("#### Stabilité du choix de stratégie — fenêtre par fenêtre")
        st.caption(
            "Pour chaque année de test, quelle stratégie le modèle a-t-il choisie "
            "sur les données passées, et quel résultat a-t-elle donné ? "
            "Un bon modèle choisit la même stratégie dans ≥ 70% des fenêtres."
        )

        def color_result(val):
            return "color: #1D9E75; font-weight: 600" if val == "✓" else "color: #D85A30"

        st.dataframe(
            stab.style
            .format({"Sharpe IS": "{:.3f}", "Sharpe OOS": "{:.3f}"})
            .map(color_sharpe, subset=["Sharpe IS", "Sharpe OOS"])
            .map(color_result,  subset=["IS → OOS"]),
            use_container_width=True,
            height=min(60 + n_win * 35, 520),
        )

        st.divider()

        # ── Verdict de synthèse ───────────────────────────────────────────────

        st.markdown("#### Verdict — ce projet est-il utilisable en pratique ?")

        # Calcul du score de robustesse sur trois critères
        score = 0
        criteria = []

        # Critère 1 : taux de fenêtres OOS positives
        if win_rate >= 65:
            score += 1
            criteria.append(("✓", f"Taux de fenêtres positives : {win_rate:.0f}% (seuil : 65%)", True))
        elif win_rate >= 50:
            criteria.append(("△", f"Taux de fenêtres positives : {win_rate:.0f}% — acceptable mais limite (seuil : 65%)", None))
        else:
            criteria.append(("✗", f"Taux de fenêtres positives : {win_rate:.0f}% — inférieur au hasard (seuil : 65%)", False))

        # Critère 2 : dégradation IS → OOS
        if degradation > -0.20:
            score += 1
            criteria.append(("✓", f"Dégradation IS→OOS : {degradation:+.3f} — faible (seuil : -0.20)", True))
        elif degradation > -0.40:
            criteria.append(("△", f"Dégradation IS→OOS : {degradation:+.3f} — modérée (seuil : -0.20)", None))
        else:
            criteria.append(("✗", f"Dégradation IS→OOS : {degradation:+.3f} — forte, probable surapprentissage (seuil : -0.20)", False))

        # Critère 3 : stabilité de la sélection de stratégie
        most_common_pct = stab["Stratégie choisie"].value_counts().iloc[0] / n_win * 100
        most_common_strat = stab["Stratégie choisie"].value_counts().index[0]
        if most_common_pct >= 70:
            score += 1
            criteria.append(("✓", f"Stabilité : « {most_common_strat} » choisie dans {most_common_pct:.0f}% des fenêtres (seuil : 70%)", True))
        elif most_common_pct >= 50:
            criteria.append(("△", f"Stabilité : « {most_common_strat} » choisie dans {most_common_pct:.0f}% des fenêtres — modérée (seuil : 70%)", None))
        else:
            criteria.append(("✗", f"Stabilité : aucune stratégie dominante — le modèle change trop souvent d'avis (seuil : 70%)", False))

        # Verdict global
        if score == 3:
            verdict_label  = "Robuste — utilisable en pratique"
            verdict_color  = "#1D9E75"
            verdict_border = "#1D9E75"
            verdict_text   = (
                f"Les trois critères de robustesse sont satisfaits. "
                f"La stratégie « {most_common_strat} » appliquée sur {wf_asset_label} "
                f"produit des résultats cohérents hors-échantillon sur {n_win} fenêtres annuelles. "
                f"Ce projet fournit un signal utilisable — non pas comme oracle, "
                f"mais comme filtre systématique pour orienter une décision d'allocation."
            )
        elif score == 2:
            verdict_label  = "Partiellement robuste — à utiliser avec précaution"
            verdict_color  = "#EF9F27"
            verdict_border = "#EF9F27"
            verdict_text   = (
                f"Deux critères sur trois sont satisfaits. Le modèle montre des signes "
                f"de robustesse mais présente des faiblesses à surveiller. "
                f"Il peut servir d'indicateur de contexte macro, "
                f"mais les signaux de stratégie ne devraient pas être suivis mécaniquement "
                f"sans validation supplémentaire."
            )
        else:
            verdict_label  = "Fragile — surapprentissage probable"
            verdict_color  = "#D85A30"
            verdict_border = "#D85A30"
            verdict_text   = (
                f"Moins de deux critères satisfaits. Les bonnes performances observées "
                f"en backtest classique ne se transfèrent pas hors-échantillon. "
                f"Le modèle a probablement sur-appris les données historiques. "
                f"À réviser avant toute utilisation pratique : ajuster les seuils, "
                f"allonger la calibration, ou simplifier les stratégies."
            )

        # Affichage du bloc verdict
        st.markdown(
            f"""<div style="border-left:5px solid {verdict_border};background:#1e1e1e;
                border-radius:10px;padding:20px 24px 18px;margin-bottom:16px;">
                <div style="font-size:11px;color:#aaaaaa;margin-bottom:6px;letter-spacing:.05em;">
                    VERDICT — {wf_asset_label.upper()} · {n_win} FENÊTRES OOS
                </div>
                <div style="font-size:20px;font-weight:700;color:{verdict_color};margin-bottom:12px;">
                    {verdict_label}
                </div>
                <div style="font-size:13px;color:#cccccc;line-height:1.6;">
                    {verdict_text}
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

        # Détail des trois critères
        for icon, text, passed in criteria:
            color = "#1D9E75" if passed is True else ("#EF9F27" if passed is None else "#D85A30")
            st.markdown(
                f"""<div style="display:flex;align-items:center;gap:10px;
                    padding:8px 14px;margin-bottom:6px;background:#1e1e1e;
                    border-radius:6px;">
                    <span style="font-size:15px;color:{color};font-weight:700;
                        min-width:20px;">{icon}</span>
                    <span style="font-size:13px;color:#cccccc;">{text}</span>
                </div>""",
                unsafe_allow_html=True,
            )

    else:
        # État initial : pas encore lancé
        st.info(
            "Configure les paramètres ci-dessus puis clique sur "
            "**▶ Lancer le walk-forward** pour démarrer l'analyse.\n\n"
            "Le premier calcul prend 1-2 minutes. Les résultats sont ensuite mis en cache "
            "tant que les paramètres ne changent pas."
        )
