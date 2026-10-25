"""
B-Book Risk Lab – risk management of a forex / CFD broker: net exposure of the B-book, client flow toxicity
(markouts), A/B-book routing, B-book P&L, VaR and limits, stress tests with gap and negative balance risk,
ESMA retail protections and IFR own funds (K-factors) for a CySEC-licensed investment firm.

Author: Laurent Motais de Narbonne
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from bbook import book as B
from bbook import clients as CL
from bbook import market as M
from bbook import params as P
from bbook import regulatory as G
from bbook import risk as R
from bbook import toxicity as T

st.set_page_config(page_title="B-Book Risk Lab", page_icon="💱", layout="wide")

C = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#d6a12a", "#d03b3b", "#3fa7c9", "#8a5a44"]
SEG_COLOR = {s: C[i] for i, s in enumerate(P.SEGMENTS.index)}
CLASS_COLOR = {"Benign / uninformed": "#8a8f98", "Insufficient history": "#c9cdd3", "Informed (long horizon)": C[3],
               "Short-term informed (news / scalping)": C[1], "Latency arbitrage": C[5]}


def h(df: pd.DataFrame) -> int:
    return min(35 * (len(df) + 1) + 3, 900)


def usd(x, d=0):
    return f"{x / 1e6:,.{max(d, 1)}f} m" if abs(x) >= 1e6 else f"{x / 1e3:,.{d}f} k"


def text(df, spec):
    out = df.copy()
    for c, fm in spec.items():
        if c in out:
            f_ = fm if callable(fm) else (lambda v, fm=fm: fm.format(v))
            out[c] = [("–" if (v is None or (isinstance(v, float) and np.isnan(v))) else f_(v)) for v in out[c]]
    return out


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
st.sidebar.title("💱 B-Book Risk Lab")
st.sidebar.caption("Forex / CFD broker · exposure · toxicity · VaR · IFR")
n_clients = st.sidebar.select_slider("Active clients", [1000, 2000, 3000], 2000)
seed = int(st.sidebar.number_input("Seed", 1, 9999, 5))
st.sidebar.subheader("A/B-book routing")
thr = st.sidebar.slider("A-book if toxicity score ≥", 0, 101, 60, 5, help="101 = everything in the B-book")
size_rule = st.sidebar.toggle("A-book large professional accounts", True)
size_thr = st.sidebar.number_input("… with equity above (USD)", 10_000, 5_000_000, 250_000, 50_000, disabled=not size_rule)
markup = st.sidebar.slider("Markup kept on A-book flow (share of spread)", 0.0, 0.8, 0.3, 0.05)
st.sidebar.subheader("Hedging of the B-book")
hmode = st.sidebar.radio("Hedge at liquidity providers", ["None", "Excess over instrument limits", "Share of net exposure"])
hratio = st.sidebar.slider("Hedge ratio", 0.0, 1.0, 0.5, 0.1, disabled=hmode != "Share of net exposure")
st.sidebar.caption("Built by [Laurent Motais de Narbonne](https://ldenarbonne.github.io/)")


@st.cache_data(show_spinner="Simulating three years of prices and three months of client flow…", max_entries=6)
def load(n_clients, seed):
    rets = M.history(seed=11)
    flow = CL.generate(n_clients, 3, seed, rets)
    return rets, flow


@st.cache_data(show_spinner="Analysing the client flow…", max_entries=12)
def analyse(n_clients, seed, markup):
    _, flow = load(n_clients, seed)
    return T.client_metrics(flow, markup)


rets, flow = load(n_clients, seed)
met = analyse(n_clients, seed, markup)
books = B.route(met, thr, size_thr if size_rule else None)
exp0 = B.net_exposure(flow, books)
hedge = {} if hmode == "None" else B.hedge_plan(exp0, "excess" if hmode.startswith("Excess") else "ratio",
                                               1.0 if hmode.startswith("Excess") else hratio)
exp = B.net_exposure(flow, books, hedge)
pos = exp["broker net notional (USD)"]
meas = R.measures(pos, rets)
var1 = float(meas.iloc[0]["VaR 1-day"])
pnl = B.bbook_pnl(flow, books, markup)
rp = B.routed_pnl(met, books)

st.title("Forex / CFD Broker: B-Book Risk")
st.caption(f"{flow.clients['client'].isin(flow.trades['client']).sum():,} active clients · {len(flow.trades):,} trades over "
           f"3 months · {len(B.client_positions(flow)):,} open positions at {flow.end:%d %b %Y %H:%M} · reporting in USD "
           "(IFR in EUR) · simulated data")
tabs = st.tabs(["Net exposure", "Client flow & toxicity", "A/B-book routing", "B-book P&L", "VaR & limits", "Stress tests",
                "CySEC / IFR & ESMA", "Methodology"])

# --------------------------------------------------------------------------
# Net exposure
# --------------------------------------------------------------------------
with tabs[0]:
    op = B.client_positions(flow, books)
    k = st.columns(6)
    k[0].metric("Gross of net exposures", usd(pos.abs().sum()))
    k[1].metric("Open B-book positions", f"{(op['book'] == 'B').sum():,}", f"{(op['book'] == 'A').sum():,} A-booked",
                delta_color="off")
    k[2].metric("VaR 99 % 1-day (historical)", usd(var1), f"limit {usd(P.LIMITS['VaR 99 % 1-day'])}", delta_color="off")
    k[3].metric("Instruments above 80 % of limit", f"{int((exp['utilisation'] >= 0.8).sum())}")
    k[4].metric("Broker P&L, 3 months", usd(pnl["total"]))
    k[5].metric("Hedge cost (one-off)", usd(B.hedge_cost(hedge, flow.px)))
    a, b_ = st.columns([3, 2])
    e = exp.reset_index().rename(columns={"index": "instrument"})
    col = np.where(e["utilisation"] >= 1, C[5], np.where(e["utilisation"] >= 0.8, C[4], C[0]))
    fig = go.Figure(go.Bar(y=e["instrument"], x=e["broker net notional (USD)"] / 1e6, orientation="h", marker_color=col,
                           name="net exposure"))
    fig.add_trace(go.Scatter(y=e["instrument"], x=e["limit (USD)"] / 1e6, mode="markers", marker=dict(symbol="line-ns-open", size=14, color="grey"),
                             name="limit"))
    fig.add_trace(go.Scatter(y=e["instrument"], x=-e["limit (USD)"] / 1e6, mode="markers", marker=dict(symbol="line-ns-open", size=14, color="grey"),
                             showlegend=False))
    fig.update_layout(title="Broker net position by instrument (USD m; + long, − short) and limits", height=480,
                      margin=dict(t=50, l=10), yaxis=dict(autorange="reversed"), legend=dict(orientation="h", y=-0.12))
    a.plotly_chart(fig, width="stretch")
    ce = B.currency_exposure(exp, flow.px) / 1e6
    fig = go.Figure(go.Bar(x=ce.index, y=ce.values, marker_color=[C[2] if v > 0 else C[1] for v in ce.values]))
    fig.update_layout(title="Currency decomposition of FX and gold positions (USD m)", height=480, margin=dict(t=50))
    b_.plotly_chart(fig, width="stretch")
    hist = B.exposure_history(flow, books) / 1e6
    sel = st.multiselect("Instruments", list(hist.columns), list(pos.abs().sort_values(ascending=False).index[:5]))
    fig = go.Figure([go.Scatter(x=hist.index, y=hist[i], name=i, mode="lines") for i in sel])
    fig.update_layout(title="End-of-day net B-book exposure over 3 months (USD m, before hedges, at current prices)",
                      height=360, margin=dict(t=50))
    st.plotly_chart(fig, width="stretch")
    a, b_ = st.columns(2)
    show = exp.drop(columns=["clients net units"]).copy()
    a.markdown("**Exposure, hedges and limit use by instrument**")
    a.dataframe(text(show.reset_index(), {"hedge units": "{:,.0f}", "broker net units": "{:,.0f}",
                                         "broker net notional (USD)": "{:,.0f}", "B-book gross client notional (USD)": "{:,.0f}",
                                         "limit (USD)": "{:,.0f}", "utilisation": "{:.0%}"}),
                hide_index=True, width="stretch", height=h(show))
    ob = op[op["book"] == "B"].assign(n=lambda d: -d["side"] * d["notional_usd"])
    cc = ob.groupby("client")["n"].sum()
    conc = (cc.abs().sort_values(ascending=False).head(10)).to_frame("abs. contribution (USD)")
    conc["share of gross net exposure"] = conc["abs. contribution (USD)"] / max(pos.abs().sum(), 1)
    conc["segment"] = conc.index.map(flow.clients.set_index("client")["segment"])
    b_.markdown("**Largest client contributions to the B-book**")
    b_.dataframe(text(conc.reset_index(), {"abs. contribution (USD)": "{:,.0f}", "share of gross net exposure": "{:.1%}"}),
                 hide_index=True, width="stretch", height=h(conc))

# --------------------------------------------------------------------------
# Toxicity
# --------------------------------------------------------------------------
with tabs[1]:
    a, b_ = st.columns(2)
    cur = T.markout_curves(flow)
    fig = go.Figure([go.Scatter(x=list(cur.columns), y=cur.loc[s], name=s, mode="lines+markers",
                                line=dict(color=SEG_COLOR[s], width=2.5)) for s in cur.index])
    fig.add_hline(y=0, line_color="grey")
    fig.add_hline(y=1, line_dash="dot", line_color="grey", annotation_text="1 spread")
    fig.update_layout(title="Average markout by segment (spreads, in the client's favour)", xaxis_title="horizon after the trade",
                      height=420, margin=dict(t=50), legend=dict(orientation="h", y=-0.2))
    a.plotly_chart(fig, width="stretch")
    fig = go.Figure()
    for cl, g in met.groupby("flow class"):
        fig.add_trace(go.Scatter(x=g["median_hold_s"], y=g["mo_1m"], mode="markers", name=cl,
                                 marker=dict(color=CLASS_COLOR.get(cl, "grey"), size=np.clip(np.sqrt(g["notional_usd"]) / 800, 4, 30),
                                             opacity=0.7),
                                 text=g["client"], hovertemplate="%{text}<br>hold %{x:.0f}s<br>1-min markout %{y:.2f}"))
    fig.update_layout(title="Clients: median holding time vs 1-minute markout (size = notional)", xaxis_type="log",
                      xaxis_title="median holding time (s)", yaxis_title="markout 1 min (spreads)", height=420, margin=dict(t=50),
                      legend=dict(orientation="h", y=-0.2))
    b_.plotly_chart(fig, width="stretch")
    ct = pd.crosstab(met["segment (true)"], met["flow class"])
    st.markdown("**Detected flow class vs simulated behaviour** (the simulation knows each client's true segment)")
    st.dataframe(ct, width="stretch", height=h(ct))
    st.caption("Short-horizon edges (latency arbitrage, news, scalping) are detected with near certainty: hundreds of "
               "trades with a markout of 1–3 spreads within a minute. A long-horizon edge (informed / professional flow) is "
               "hard to prove statistically over three months because daily price noise is 50–100 spreads: this is why "
               "brokers also route by account size and monitor realised P&L.")
    cols = ["client", "segment (true)", "flow class", "toxicity score", "trades", "notional_usd", "median_hold_s", "mo_5s",
            "mo_1m", "mo_1h", "mo_1d", "win_rate", "client_pnl", "broker_pnl_B", "broker_pnl_A"]
    top = met.sort_values("toxicity score", ascending=False)[cols].head(40).assign(book=lambda d: d["client"].map(books))
    st.markdown("**Most toxic clients**")
    st.dataframe(text(top, {"notional_usd": "{:,.0f}", "median_hold_s": "{:,.0f}", "mo_5s": "{:.2f}", "mo_1m": "{:.2f}",
                            "mo_1h": "{:.2f}", "mo_1d": "{:.2f}", "win_rate": "{:.0%}", "client_pnl": "{:,.0f}",
                            "broker_pnl_B": "{:,.0f}", "broker_pnl_A": "{:,.0f}", "toxicity score": "{:.0f}"}),
                 hide_index=True, width="stretch", height=420)

# --------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------
with tabs[2]:
    k = st.columns(5)
    k[0].metric("Broker P&L – all B-book", usd(rp["all B-book"]))
    k[1].metric("Broker P&L – all A-book", usd(rp["all A-book"]))
    k[2].metric("Broker P&L – routed", usd(rp["routed"]))
    k[3].metric("Clients A-booked", f"{rp['share of clients A-booked']:.0%}")
    k[4].metric("Notional A-booked", f"{rp['share of notional A-booked']:.0%}")

    @st.cache_data(show_spinner="Computing the routing frontier…", max_entries=12)
    def frontier(n_clients, seed, markup, size_thr):
        _, fl = load(n_clients, seed)
        m_ = analyse(n_clients, seed, markup)
        rows = []
        for t_ in [0, 20, 30, 40, 50, 60, 70, 80, 90, 95, 101]:
            bk = B.route(m_, t_, size_thr)
            ex = B.net_exposure(fl, bk)
            v = R.var_es(R.pnl_vector(ex["broker net notional (USD)"], rets.iloc[-500:]))[0]
            rows.append({"threshold": t_, **B.routed_pnl(m_, bk), "VaR 99 % 1-day": v})
        return pd.DataFrame(rows)

    fr = frontier(n_clients, seed, markup, size_thr if size_rule else None)
    fig = go.Figure(go.Scatter(x=fr["VaR 99 % 1-day"] / 1e6, y=fr["routed"] / 1e6, mode="lines+markers+text",
                               text=[("score rule off" if t_ == 101 else str(t_)) for t_ in fr["threshold"]], textposition="top center",
                               line=dict(color=C[0], width=2.5)))
    cur_v = R.var_es(R.pnl_vector(B.net_exposure(flow, books)["broker net notional (USD)"], rets.iloc[-500:]))[0]
    fig.add_trace(go.Scatter(x=[cur_v / 1e6], y=[rp["routed"] / 1e6], mode="markers", marker=dict(size=14, color=C[5]),
                             name="current setting"))
    fig.update_layout(title="Routing frontier: broker P&L over 3 months vs VaR of the B-book (labels = toxicity threshold; "
                            "the size rule set in the sidebar applies throughout)",
                      xaxis_title="VaR 99 % 1-day of the unhedged B-book (USD m)", yaxis_title="broker P&L (USD m)",
                      height=440, margin=dict(t=50), showlegend=False)
    st.plotly_chart(fig, width="stretch")
    pb = B.pnl_by(met, books)
    st.dataframe(text(pb.reset_index(), {"share A-booked": "{:.0%}", "notional": "{:,.0f}", "broker P&L if all B-book": "{:,.0f}",
                                        "broker P&L if all A-book": "{:,.0f}", "broker P&L (routed)": "{:,.0f}"}),
                 hide_index=True, width="stretch", height=h(pb))
    st.info("B-booking uninformed retail flow earns the spread plus the clients' trading losses; B-booking toxic flow "
            "transfers its edge to the broker's P&L. Routing the most toxic clients to liquidity providers keeps the retail "
            "margin and cuts the market risk of the book. In practice LPs also price toxic flow (wider spreads, last look, "
            "rejection), so the A-book markup on such flow is optimistic.")

# --------------------------------------------------------------------------
# P&L
# --------------------------------------------------------------------------
with tabs[3]:
    comp = pnl["components"]
    fig = go.Figure(go.Waterfall(x=list(comp.index) + ["Total"], y=list(comp.values / 1e6) + [comp.sum() / 1e6],
                                 measure=["relative"] * len(comp) + ["total"],
                                 text=[f"{v / 1e6:,.2f}" for v in list(comp.values) + [comp.sum()]], textposition="outside"))
    fig.update_layout(title="Broker P&L over 3 months by source (USD m, current routing)", height=440, margin=dict(t=50, b=120))
    st.plotly_chart(fig, width="stretch")
    a, b_ = st.columns(2)
    d = pnl["daily"] / 1e6
    fig = go.Figure([go.Bar(x=d.index, y=d.values, name="daily", marker_color=C[0]),
                     go.Scatter(x=d.index, y=d.cumsum().values, name="cumulative", yaxis="y2", line=dict(color=C[1], width=2.5))])
    fig.update_layout(title="Daily P&L (closing date of trades) and cumulative (USD m)", height=380, margin=dict(t=50),
                      yaxis2=dict(overlaying="y", side="right", showgrid=False), legend=dict(orientation="h", y=-0.2))
    a.plotly_chart(fig, width="stretch")
    bi = pnl["by instrument"] / 1e6
    fig = go.Figure([go.Bar(x=bi.index, y=bi[c_], name=f"{c_}-book", marker_color=C[0] if c_ == "B" else C[2]) for c_ in bi.columns])
    fig.update_layout(barmode="relative", title="P&L by instrument and book (USD m)", height=380, margin=dict(t=50))
    b_.plotly_chart(fig, width="stretch")
    st.caption("B-book P&L = spread + commissions + financing − client mid-to-mid trading P&L (realised and unrealised); "
               "A-book P&L = markup + commissions. Hedge P&L is shown in the stress tests (the hedge is set today).")

# --------------------------------------------------------------------------
# VaR & limits
# --------------------------------------------------------------------------
with tabs[4]:
    st.dataframe(text(meas, {c: "{:,.0f}" for c in meas.columns if c != "method"}), hide_index=True, width="stretch", height=h(meas))
    a, b_ = st.columns(2)
    cv = R.components(pos, rets)
    cv = cv[cv["net notional (USD)"] != 0].sort_values("component VaR (parametric)", ascending=False)
    fig = go.Figure([go.Bar(x=cv.index, y=cv["standalone VaR"] / 1e3, name="standalone", marker_color="#c9cdd3"),
                     go.Bar(x=cv.index, y=cv["component VaR (parametric)"] / 1e3, name="component (Euler)", marker_color=C[0])])
    fig.update_layout(barmode="group", title="VaR by instrument (USD k): standalone vs contribution", height=400,
                      margin=dict(t=50), legend=dict(orientation="h", y=-0.2))
    a.plotly_chart(fig, width="stretch")
    bt = R.backtest(pos, rets)
    fig = go.Figure([go.Bar(x=bt["dates"], y=bt["pnl"] / 1e3, name="hypothetical P&L", marker_color=C[0]),
                     go.Scatter(x=bt["dates"], y=-bt["var"] / 1e3, name="− VaR 99 %", line=dict(color=C[5], width=2))])
    exc = -bt["pnl"] > bt["var"]
    fig.add_trace(go.Scatter(x=bt["dates"][exc], y=bt["pnl"][exc] / 1e3, mode="markers", name="exception",
                             marker=dict(color=C[5], size=9, symbol="x")))
    fig.update_layout(title=f"Backtest of the current book over 250 days: {bt['exceptions']} exceptions ({bt['Basel zone']} zone, "
                            f"Kupiec p = {bt['Kupiec p-value']:.2f})", height=400, margin=dict(t=50), legend=dict(orientation="h", y=-0.2))
    b_.plotly_chart(fig, width="stretch")
    op_b = B.client_positions(flow, books)
    ob = op_b[op_b["book"] == "B"].assign(n=lambda d: -d["side"] * d["notional_usd"])
    share = float(ob.groupby("client")["n"].sum().abs().max() / max(pos.abs().sum(), 1)) if len(ob) else 0.0
    last = float(pnl["daily"].iloc[-1]) if len(pnl["daily"]) else 0.0
    lim = R.limits(exp, var1, last, share)
    st.markdown("**Limits**")
    st.dataframe(text(lim, {"value": lambda v: f"{v:.1%}" if v < 1.5 else f"{v:,.0f}", "limit value": lambda v: f"{v:.0%}" if v < 1.5 else f"{v:,.0f}",
                            "utilisation": "{:.0%}"}).style.apply(
        lambda r_: ["background-color: rgba(208,59,59,0.18)" if r_["status"] == "breach" else
                    ("background-color: rgba(214,161,42,0.18)" if r_["status"] == "warning" else "") for _ in r_], axis=1),
        hide_index=True, width="stretch", height=h(lim))
    sv = R.stressed_var(pos, rets)
    st.caption(f"Stressed VaR window: {sv['start']:%d %b %Y} – {sv['end']:%d %b %Y}. VaR is computed on the broker's net "
               "positions after hedges, valued in USD; the 10-day figure scales the 1-day VaR by √10.")

# --------------------------------------------------------------------------
# Stress
# --------------------------------------------------------------------------
with tabs[5]:
    st.markdown("**Scenario shocks** (instantaneous relative price moves, editable; stylised, inspired by historical episodes)")
    sc = P.SCENARIOS.copy()
    sc.index.name = "instrument"
    sc = sc.reindex(P.INSTRUMENTS.index).fillna(0.0)
    ed = st.data_editor((sc * 100).round(1), width="stretch", height=h(sc), key="scen")
    scen = ed / 100
    st_res = R.stress(flow, books, hedge, scen)
    fig = go.Figure()
    for col_, colr in (("B-book P&L", C[0]), ("hedge P&L", C[2]), ("NBP loss on A-book clients", C[5])):
        fig.add_trace(go.Bar(x=st_res["scenario"], y=st_res[col_] / 1e6, name=col_, marker_color=colr))
    fig.add_trace(go.Scatter(x=st_res["scenario"], y=st_res["total broker P&L"] / 1e6, mode="markers", name="total",
                             marker=dict(color="black", size=11, symbol="diamond")))
    fig.update_layout(barmode="relative", title="Broker P&L under each scenario (USD m)", height=440, margin=dict(t=50, b=120),
                      legend=dict(orientation="h", y=-0.45))
    st.plotly_chart(fig, width="stretch")
    st.dataframe(text(st_res, {c: (lambda v: f"{v + 0.0:,.0f}") for c in st_res.columns if c not in ("scenario", "accounts in negative balance")}),
                 hide_index=True, width="stretch", height=h(st_res))
    st.info("A gap leaves no time for the 50 % margin close-out: client losses beyond their equity cannot be collected "
            "(negative balance protection). On A-booked positions the broker still loses the full amount on its LP hedge, "
            "so the uncollectable part is a direct loss; on B-booked positions it reduces the broker's gain. The Swiss franc "
            "de-peg of January 2015 caused losses of this kind at several FX brokers.")

# --------------------------------------------------------------------------
# Regulatory
# --------------------------------------------------------------------------
with tabs[6]:
    c1, c2, c3, c4 = st.columns(4)
    fo = c1.number_input("Fixed overheads, last year (EUR m)", 1.0, 200.0, 10.0, 0.5) * 1e6
    of = c2.number_input("Own funds (EUR m)", 0.5, 500.0, 15.0, 0.5) * 1e6
    lpm = c3.slider("Margin posted at LPs (% of A-book and hedge notional)", 1, 20, 5) / 100
    la = c4.number_input("Liquid assets (EUR m)", 0.0, 200.0, 6.0, 0.5) * 1e6
    ir = G.ifr(flow, books, exp, fo, of, lpm, la)
    k = st.columns(5)
    k[0].metric("K-factor requirement", f"€{ir['K-factor requirement'] / 1e6:,.2f} m")
    k[1].metric("Fixed overheads requirement", f"€{ir['FOR'] / 1e6:,.2f} m")
    k[2].metric("Permanent minimum capital", f"€{ir['PMR'] / 1e6:,.2f} m")
    k[3].metric("Own funds requirement", f"€{ir['own funds requirement'] / 1e6:,.2f} m", f"binding: {ir['binding']}", delta_color="off")
    k[4].metric("Own funds ratio", f"{ir['own funds ratio']:.0%}", "≥ 100 % required", delta_color="off")
    a, b_ = st.columns(2)
    kf = ir["K-factors"] / 1e6
    fig = go.Figure(go.Bar(x=kf.values, y=kf.index, orientation="h", marker_color=C[0], text=[f"{v:,.2f}" for v in kf.values],
                           textposition="outside"))
    fig.add_vline(x=ir["FOR"] / 1e6, line_dash="dot", annotation_text="FOR")
    fig.update_layout(title="K-factors (EUR m)", height=360, margin=dict(t=50, l=10), yaxis=dict(autorange="reversed"))
    a.plotly_chart(fig, width="stretch")
    b_.markdown("**K-NPR detail (simplified standardised approach)**")
    b_.dataframe(text(ir["NPR detail"], {"position (EUR)": "{:,.0f}", "charge": "{:,.0f}"}), hide_index=True, width="stretch",
                 height=h(ir["NPR detail"]))
    b_.caption(f"Client money held €{ir['client money (EUR)'] / 1e6:,.1f} m · average daily trading flow "
               f"€{ir['average daily trading flow (EUR)'] / 1e6:,.0f} m · average daily client orders handled "
               f"€{ir['average daily client orders (EUR)'] / 1e6:,.0f} m · liquidity requirement (1/3 FOR) "
               f"€{ir['liquidity requirement (1/3 FOR)'] / 1e6:,.2f} m vs liquid assets €{la / 1e6:,.1f} m.")
    st.subheader("ESMA retail protections")
    ac = G.esma_accounts(flow, books)
    k = st.columns(4)
    lr = G.losing_retail_share(flow)
    k[0].metric("Retail accounts losing money (3 months)", f"{lr:.0%}")
    k[1].metric("Accounts with open positions", f"{len(ac):,}")
    k[2].metric("In margin call (< 100 %)", f"{int((ac['status'] == 'margin call').sum()):,}")
    k[3].metric("Below the 50 % close-out level", f"{int((ac['status'] != 'ok').sum() - (ac['status'] == 'margin call').sum()):,}")
    a, b_ = st.columns(2)
    fig = go.Figure(go.Histogram(x=np.clip(ac["margin level"] * 100, 0, 500), nbinsx=50, marker_color=C[0]))
    fig.update_xaxes(range=[0, 520])
    fig.add_vline(x=100, line_dash="dot", annotation_text="margin call 100 %", annotation_position="top right")
    fig.add_vline(x=50, line_color=C[5], annotation_text="close-out 50 %", annotation_position="bottom left")
    fig.update_layout(title="Margin level of accounts with open positions (equity / required margin, %)", height=340,
                      margin=dict(t=50))
    a.plotly_chart(fig, width="stretch")
    lev = P.INSTRUMENTS.groupby("class")["leverage"].first().rename("maximum leverage (retail)").to_frame()
    lev["initial margin"] = 1 / lev["maximum leverage (retail)"]
    b_.dataframe(text(lev.reset_index(), {"initial margin": "{:.1%}"}), hide_index=True, width="stretch", height=h(lev))
    b_.caption(f"Risk warning required on the broker's communications: “CFDs are complex instruments and come with a high risk "
               f"of losing money rapidly due to leverage. {lr:.0%} of retail investor accounts lose money when trading CFDs "
               "with this provider…” (ESMA format; the percentage is computed on a rolling 12-month basis in practice).")

# --------------------------------------------------------------------------
# Methodology
# --------------------------------------------------------------------------
with tabs[7]:
    st.markdown(r"""
### Data
* **Market.** Three years of daily returns for 14 CFDs (FX majors and non-majors, gold, WTI, US500, NAS100, GER40,
  BTC, ETH). The simulation uses a factor correlation structure, Student t innovations, a stress quarter and jumps.
  Prices end at current levels.
* **Clients.** Six behavioural segments (directional and active retail, scalpers, news traders, latency arbitrage,
  informed / professional), each with its own frequency, size, holding period and *alpha profile*: the expected price
  move in the client's favour after the trade, $a(\tau) = A\cdot \text{spread}\cdot(1-e^{-\tau/\tau_a})$.
  * The price path of every trade is simulated at the markout horizons, at closing and at the reporting date.
  * Sizes respect the ESMA leverage caps (professionals: 5× the retail leverage).
  * Spreads, commissions on raw accounts and overnight swaps are charged.

### Toxicity and routing
* **Markouts.** For each client, average markouts at 1 s – 1 day and their t-statistics, the return per trade, the
  win rate and the P&L. The **toxicity score** is a logistic transform of the largest t-statistic (50 = t of 2).
* **Classification.** Rules separate latency arbitrage (edge within seconds, holding under 2 minutes), short-term
  informed flow (edge within a minute, holding under an hour) and long-horizon informed flow.
* **Routing.** A-book if the score exceeds the threshold, or for large professional accounts. B-book P&L of a client =
  spread + commission + financing − mid-to-mid P&L; A-book P&L = markup + commission.

### Exposure, VaR and limits
* **Exposure and hedges.** Net position = − B-booked client positions + hedges, in units and USD. FX and gold are
  also decomposed into currencies. Hedges either bring each instrument back within its limit or cover a share of the
  net exposure.
* **VaR methods.**
  * historical simulation (500 days);
  * parametric (equally weighted 250 days, or EWMA 0.94);
  * Monte-Carlo with Student t innovations;
  * stressed VaR on the worst 250-day window;
  * component (Euler) and incremental VaR;
  * backtest of the current book with the Kupiec test and Basel traffic light.
* **Limits.** VaR, net notional by instrument class, gross of net notionals, client concentration, daily stop-loss.

### Stress tests
Instantaneous shocks with no time for the margin close-out. Client losses beyond equity cannot be collected
(negative balance protection). They are a direct loss on A-booked positions, whose LP hedge loses in full, and a
forgone gain on B-booked positions.

### Regulation (CySEC)
* **ESMA product intervention (2018, made permanent nationally).**
  * leverage caps: 30:1 major FX; 20:1 non-major FX, gold and major indices; 10:1 other commodities and
    non-major indices; 5:1 equities; 2:1 crypto;
  * margin close-out at 50 %;
  * negative balance protection;
  * standardised risk warning with the share of losing retail accounts.
* **IFR / IFD.** The own funds requirement is the highest of three amounts:
  * the permanent minimum capital (EUR 750,000 for dealing on own account);
  * a quarter of fixed overheads;
  * the K-factors:
    * K-NPR: simplified standardised approach;
    * K-CMH: 0.4 % of segregated client money;
    * K-DTF and K-COH: 0.01 % of derivatives flows;
    * K-TCD: 1.2 × exposure × 1.6 % for LP exposures.

  There is also a liquidity requirement of a third of the fixed overheads requirement.

### Limitations
* **Simulated data.** Markouts are simulated directly rather than measured on tick data.
* **K-NPR.** Simplified, with a 100 % charge on crypto as a conservative assumption.
* **A-book P&L.** Ignores LP pricing of toxic flow (last look, rejections, wider spreads).
* **Swaps.** Indicative.

### References
* ESMA (2018), *Product intervention measures relating to contracts for differences* (Decision (EU) 2018/796).
* Regulation (EU) 2019/2033 (IFR) and Directive (EU) 2019/2034 (IFD); CySEC Law 165(I)/2021.
* Basel Committee, *Supervisory framework for the use of backtesting* (1996); Kupiec (1995).
* Oomen, R. (2017), *Execution in an aggregator*, Quantitative Finance; Butz, M. and Oomen, R. (2019), *Internalisation
  by electronic FX spot dealers*, Quantitative Finance.
""")
