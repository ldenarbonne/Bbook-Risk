"""
Net exposure of the B-book, routing of clients between the A-book and the B-book, hedging and P&L.

Sign convention: client positions are signed units (+ long); the broker's B-book position is the opposite of
the B-booked clients' positions, plus its hedges at liquidity providers (LPs).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import market as M
from . import params as P

INS = P.INSTRUMENTS


# ------------------------------------------------------------------ routing
def route(metrics: pd.DataFrame, score_threshold: float = 60.0, size_threshold: float | None = 250_000.0,
          a_classes: tuple = ()) -> pd.Series:
    """A-book if the toxicity score reaches the threshold, or the account is a large professional one, or the
    flow class is listed; B-book otherwise. Returns a Series client -> 'A' / 'B'."""
    a = metrics["toxicity score"] >= score_threshold
    if size_threshold is not None:
        a |= (metrics["category"] == "Professional") & (metrics["equity"] >= size_threshold)
    if a_classes:
        a |= metrics["flow class"].isin(a_classes)
    return pd.Series(np.where(a, "A", "B"), index=metrics["client"])


def routed_pnl(metrics: pd.DataFrame, books: pd.Series) -> dict:
    b = metrics["client"].map(books).to_numpy() == "B"
    return {"all B-book": float(metrics["broker_pnl_B"].sum()), "all A-book": float(metrics["broker_pnl_A"].sum()),
            "routed": float(np.where(b, metrics["broker_pnl_B"], metrics["broker_pnl_A"]).sum()),
            "share of clients A-booked": float((~b).mean()),
            "share of notional A-booked": float(metrics.loc[~b, "notional_usd"].sum() / metrics["notional_usd"].sum())}


def pnl_by(metrics: pd.DataFrame, books: pd.Series, by: str = "segment (true)") -> pd.DataFrame:
    m = metrics.assign(book=metrics["client"].map(books))
    m["broker P&L (routed)"] = np.where(m["book"] == "B", m["broker_pnl_B"], m["broker_pnl_A"])
    g = m.groupby(by).agg(clients=("client", "size"), A_booked=("book", lambda x: (x == "A").mean()),
                          notional=("notional_usd", "sum"), all_B=("broker_pnl_B", "sum"), all_A=("broker_pnl_A", "sum"),
                          routed=("broker P&L (routed)", "sum"))
    return g.rename(columns={"A_booked": "share A-booked", "all_B": "broker P&L if all B-book",
                             "all_A": "broker P&L if all A-book", "routed": "broker P&L (routed)"})


# ------------------------------------------------------------------ exposure
def client_positions(flow, books: pd.Series | None = None, at: pd.Timestamp | None = None) -> pd.DataFrame:
    t = flow.trades
    at = flow.end if at is None else at
    op = t[(t["open_time"] <= at) & (t["close_time"] > at)].copy()
    op["book"] = op["client"].map(books) if books is not None else "B"
    op["signed_units"] = op["side"] * op["units"]
    return op


def net_exposure(flow, books: pd.Series | None = None, hedge_units: dict | None = None) -> pd.DataFrame:
    """Broker net position by instrument (units and USD notional): − B-booked client positions + hedges."""
    op = client_positions(flow, books)
    b = op[op["book"] == "B"]
    cl = b.groupby("instrument")["signed_units"].sum().reindex(INS.index).fillna(0.0)
    gross = b.assign(n=b["notional_usd"]).groupby("instrument")["n"].sum().reindex(INS.index).fillna(0.0)
    nb = b.groupby("instrument").size().reindex(INS.index).fillna(0).astype(int)
    hedge = pd.Series(hedge_units or {}, dtype=float).reindex(INS.index).fillna(0.0)
    units = -cl + hedge
    px = flow.px
    usd = pd.Series({i: float(M.notional_usd(units[i], i, px)) for i in INS.index})
    out = pd.DataFrame({"class": INS["class"], "open B-book positions": nb, "clients net units": cl,
                        "hedge units": hedge, "broker net units": units,
                        "broker net notional (USD)": usd, "B-book gross client notional (USD)": gross})
    lim = out["class"].map(P.LIMITS["Net notional by instrument"])
    out["limit (USD)"] = lim
    out["utilisation"] = out["broker net notional (USD)"].abs() / lim
    return out


def currency_exposure(exp: pd.DataFrame, px: dict) -> pd.Series:
    """Decomposition of FX and gold positions into currency positions (USD equivalent)."""
    usdq = M.usd_per_quote(px)
    usdq["XAU"] = px["XAUUSD"]
    usdq["GBP"] = px["GBPUSD"]
    usdq["AUD"] = px["AUDUSD"]
    out = {}
    for i, r in exp.iterrows():
        b, q = INS.loc[i, "base"], INS.loc[i, "quote"]
        if INS.loc[i, "npr"] not in ("FX", "Gold"):
            continue
        u = r["broker net units"]
        out[b] = out.get(b, 0.0) + u * usdq[b]
        out[q] = out.get(q, 0.0) - u * px[i] * usdq[q]
    return pd.Series(out).sort_values()


def exposure_history(flow, books: pd.Series | None = None, days: int = 63) -> pd.DataFrame:
    """End-of-day broker net notional (USD, at current prices) by instrument."""
    t = flow.trades
    if books is not None:
        t = t[t["client"].map(books) == "B"]
    ends = pd.bdate_range(end=flow.end.normalize(), periods=days) + pd.Timedelta(hours=22)
    rows = {}
    su = (t["side"] * t["units"]).to_numpy()
    o, c = t["open_time"].to_numpy(), t["close_time"].to_numpy()
    ins = t["instrument"].to_numpy()
    for d in ends:
        live = (o <= d.to_datetime64()) & (c > d.to_datetime64())
        s = pd.Series(-su[live]).groupby(ins[live]).sum()
        rows[d.normalize()] = {i: float(M.notional_usd(s.get(i, 0.0), i, flow.px)) for i in INS.index}
    return pd.DataFrame(rows).T


def hedge_plan(exp: pd.DataFrame, mode: str = "excess", ratio: float = 1.0) -> dict:
    """Hedge units: 'excess' = bring every instrument back within its limit; 'ratio' = hedge a share of the net."""
    out = {}
    for i, r in exp.iterrows():
        net, lim = r["broker net units"] - r["hedge units"], r["limit (USD)"]
        usd = r["broker net notional (USD)"] / r["broker net units"] if r["broker net units"] else 0.0
        if mode == "ratio":
            out[i] = -ratio * net
        elif usd and abs(net * usd) > lim:
            out[i] = -(net - np.sign(net) * lim / abs(usd)) * ratio
    return out


def hedge_cost(hedge_units: dict, px: dict, lp_spread_share: float = 0.3) -> float:
    """Cost of crossing the LP spread (assumed to be a share of the client spread) once on the hedge."""
    return float(sum(abs(u) * INS.loc[i, "spread"] * lp_spread_share / 2 * M.usd_per_quote(px)[INS.loc[i, "quote"]]
                     for i, u in hedge_units.items()))


# ------------------------------------------------------------------ P&L of the B-book
def bbook_pnl(flow, books: pd.Series | None = None, a_book_markup: float = 0.3) -> dict:
    t = flow.trades.copy()
    t["book"] = t["client"].map(books) if books is not None else "B"
    b, a = t["book"] == "B", t["book"] == "A"
    comp = {
        "Spread captured (B-book)": t.loc[b, "spread_paid"].sum(),
        "Markup on A-book flow": a_book_markup * t.loc[a, "spread_paid"].sum(),
        "Commissions": t["commission"].sum(),
        "Swaps / financing": -t.loc[b, "swap"].sum(),
        "Client trading P&L (realised, opposite)": -t.loc[b & (t["status"] == "closed"), "client_pnl_mid"].sum(),
        "Client trading P&L (unrealised, opposite)": -t.loc[b & (t["status"] == "open"), "client_pnl_mid"].sum(),
    }
    t["broker"] = np.where(b, -t["client_pnl_mid"] + t["spread_paid"] - t["swap"], a_book_markup * t["spread_paid"]) + t["commission"]
    t["day"] = t["close_time"].where(t["status"] == "closed", flow.end).dt.normalize()
    daily = t.groupby("day")["broker"].sum()
    by_ins = t.groupby(["instrument", "book"])["broker"].sum().unstack().fillna(0.0)
    return {"components": pd.Series(comp), "daily": daily, "by instrument": by_ins, "total": float(t["broker"].sum())}
