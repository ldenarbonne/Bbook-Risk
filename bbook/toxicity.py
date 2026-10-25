"""
Client flow analysis ("toxicity").

Markout of a trade at horizon τ = signed mid-price move after the trade, in the client's favour, expressed in
spreads: a client whose markouts are positive at short horizons trades on information the broker's price did
not have yet (latency arbitrage, news). For each client: activity, holding time, P&L, markout curve, win rate,
statistical significance of the markout, and a toxicity score; a rule-based classification of the flow; and the
expected P&L of the client for the broker if the flow is kept in the B-book (− client mid-to-mid P&L + spread
and commission) or passed to the A-book (markup only).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import params as P

HZ = list(P.MARKOUT_HORIZONS)


def client_metrics(flow, a_book_markup: float = 0.3) -> pd.DataFrame:
    t = flow.trades
    g = t.groupby("client")
    m = pd.DataFrame({
        "trades": g.size(), "notional_usd": g["notional_usd"].sum(), "median_hold_s": g["hold_s"].median(),
        "client_pnl": g["client_pnl"].sum(), "client_pnl_mid": g["client_pnl_mid"].sum(),
        "spread_paid": g["spread_paid"].sum(), "commission": g["commission"].sum(), "swap": g["swap"].sum(),
        "win_rate": g["client_pnl_mid"].apply(lambda x: (x > 0).mean()),
    })
    for h in HZ:
        m[f"mo_{h}"] = g[f"mo_{h}"].mean()
    # t-statistic of the 1-minute and 1-hour markouts (is the edge significant?)
    for h in ("5s", "1m", "1h", "1d"):
        sd = g[f"mo_{h}"].std().replace(0, np.nan)
        m[f"t_{h}"] = m[f"mo_{h}"] / (sd / np.sqrt(m["trades"]))
    r = t.assign(r=t["client_pnl_mid"] / t["notional_usd"]).groupby("client")["r"]
    m["t_pnl"] = r.mean() / (r.std().replace(0, np.nan) / np.sqrt(m["trades"]))
    m["pnl_per_million"] = m["client_pnl_mid"] / m["notional_usd"] * 1e6
    # broker P&L if the client is B-booked / A-booked (A-book: the broker keeps a markup share of the spread)
    m["broker_pnl_B"] = -m["client_pnl_mid"] + m["spread_paid"] + m["commission"] - m["swap"]
    m["broker_pnl_A"] = a_book_markup * m["spread_paid"] + m["commission"]
    c = flow.clients.set_index("client")
    m["segment (true)"] = m.index.map(c["segment"])
    m["category"] = m.index.map(c["category"])
    m["equity"] = m.index.map(c["equity"])
    m["toxicity score"] = score(m)
    m["flow class"] = classify(m)
    return m.reset_index()


def score(m: pd.DataFrame) -> pd.Series:
    """Toxicity score in [0, 100]: largest t-statistic among the short-horizon markouts (5 s, 1 min), the
    long-horizon markouts (1 h, 1 day) and the client's return per trade, through a logistic function;
    50 = an edge just significant (t = 2)."""
    t_short = m[["t_5s", "t_1m"]].max(axis=1).fillna(0)
    t_long = m[["t_1h", "t_1d", "t_pnl"]].max(axis=1).fillna(0)
    edge = np.maximum(t_short, t_long)
    return (100 / (1 + np.exp(-(edge - 2.0) * 1.2))).round(1)


def classify(m: pd.DataFrame) -> pd.Series:
    out = np.full(len(m), "Benign / uninformed", dtype=object)
    sig_short = (m["t_5s"] > 2) & (m["mo_5s"] > 0.5)
    sig_min = (m["t_1m"] > 2) & (m["mo_1m"] > 0.3)
    sig_long = ((m["t_1h"] > 2) & (m["mo_1h"] > 0.5)) | ((m["t_1d"] > 2) & (m["mo_1d"] > 1)) | (m["t_pnl"] > 2)
    out[sig_long.to_numpy()] = "Informed (long horizon)"
    out[(sig_min & (m["median_hold_s"] < 3_600)).to_numpy()] = "Short-term informed (news / scalping)"
    out[(sig_short & (m["median_hold_s"] < 120)).to_numpy()] = "Latency arbitrage"
    out[((m["trades"] < 10) & (out == "Benign / uninformed")).to_numpy()] = "Insufficient history"
    return pd.Series(out, index=m.index)


def markout_curves(flow, by: str = "segment") -> pd.DataFrame:
    t = flow.trades
    cur = t.groupby(by)[[f"mo_{h}" for h in HZ]].mean()
    cur.columns = HZ
    return cur
