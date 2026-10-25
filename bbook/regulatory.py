"""
Regulatory views for a CySEC-licensed CFD provider: ESMA retail protections (leverage limits, margin close-out,
negative balance protection, share of losing retail accounts for the risk warning) and the IFR own funds
requirement (permanent minimum capital, fixed overheads requirement, K-factors).

K-NPR is computed with a simplified CRR standardised approach on the broker's net positions: FX and gold 8 % of
the overall net foreign-exchange position (larger of the sum of net long and net short currency positions,
excluding the reporting currency EUR); commodities 15 % of the net position + 3 % of the gross position;
equity index CFDs 8 % general risk (broad indices, no specific risk); crypto-assets 100 % of the net position
(conservative assumption: no dedicated treatment in the CRR market risk framework applied here).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import market as M
from . import params as P
from .book import client_positions, currency_exposure

INS = P.INSTRUMENTS


def esma_accounts(flow, books=None) -> pd.DataFrame:
    op = client_positions(flow, books)
    lev = op["instrument"].map(INS["leverage"]).to_numpy()
    cl = flow.clients.set_index("client")
    pro = op["client"].map(cl["category"]).eq("Professional").to_numpy()
    op["required margin"] = op["notional_usd"] / np.where(pro, lev * 5, lev)
    g = op.groupby("client").agg(positions=("trade", "size"), notional=("notional_usd", "sum"),
                                 required_margin=("required margin", "sum"))
    g["equity"] = g.index.map(cl["equity"])
    g["category"] = g.index.map(cl["category"])
    g["effective leverage"] = g["notional"] / g["equity"]
    g["margin level"] = g["equity"] / g["required_margin"]
    g["status"] = np.select([g["margin level"] < 0.5, g["margin level"] < 1.0], ["close-out (< 50 %)", "margin call"], "ok")
    return g.reset_index()


def losing_retail_share(flow) -> float:
    c = flow.clients
    r = c[(c["category"] == "Retail") & c["client"].isin(flow.trades["client"])]
    return float((r["trading_pnl"] < 0).mean())


def k_npr(exp: pd.DataFrame, px: dict) -> pd.DataFrame:
    eurusd = px["EURUSD"]
    rows = []
    ccy = currency_exposure(exp, px) / eurusd
    ccy = ccy.drop("EUR", errors="ignore")
    longs, shorts = ccy[ccy > 0].sum(), -ccy[ccy < 0].sum()
    rows.append({"risk": "FX and gold (overall net open position)", "position (EUR)": max(longs, shorts),
                 "charge": P.IFR["NPR FX"] * max(longs, shorts)})
    for i, r in exp.iterrows():
        v = abs(r["broker net notional (USD)"]) / eurusd
        kind = INS.loc[i, "npr"]
        if kind == "Commodity":
            rows.append({"risk": f"Commodity {i}", "position (EUR)": v,
                         "charge": (P.IFR["NPR commodity net"] + P.IFR["NPR commodity gross"]) * v})
        elif kind == "Equity index":
            rows.append({"risk": f"Equity index {i}", "position (EUR)": v, "charge": P.IFR["NPR equity index"] * v})
        elif kind == "Crypto":
            rows.append({"risk": f"Crypto {i}", "position (EUR)": v, "charge": P.IFR["NPR crypto (assumption)"] * v})
    return pd.DataFrame(rows)


def ifr(flow, books, exp: pd.DataFrame, fixed_overheads_eur: float = 10e6, own_funds_eur: float = 15e6,
        lp_margin_share: float = 0.05, liquid_assets_eur: float = 6e6) -> dict:
    px = flow.px
    eurusd = px["EURUSD"]
    t = flow.trades
    book = t["client"].map(books)
    n_days = max(t["open_time"].dt.normalize().nunique(), 1)
    # daily trading flow: notional dealt on own account (B-book), opening and closing legs; client orders handled: A-book
    dtf = (t.loc[book == "B", "notional_usd"] * np.where(t.loc[book == "B", "status"] == "closed", 2, 1)).sum() / n_days / eurusd
    coh = (t.loc[book == "A", "notional_usd"] * np.where(t.loc[book == "A", "status"] == "closed", 2, 1)).sum() / n_days / eurusd
    cmh = flow.clients.loc[flow.clients["client"].isin(t["client"]), "equity"].sum() / eurusd
    npr = k_npr(exp, px)
    a_open = client_positions(flow, books)
    lp_notional = (a_open.loc[a_open["book"] == "A", "notional_usd"].sum()
                   + (exp["hedge units"] != 0).astype(float).mul(exp["broker net notional (USD)"].abs()).sum() * 0) / eurusd
    hedge_notional = sum(abs(float(M.notional_usd(u, i, px))) for i, u in
                         exp["hedge units"][exp["hedge units"] != 0].items()) / eurusd
    ev = lp_margin_share * (lp_notional + hedge_notional)
    k = {"K-NPR (net position risk)": float(npr["charge"].sum()),
         "K-CMH (client money held, segregated)": P.IFR["K-CMH segregated"] * cmh,
         "K-DTF (daily trading flow, derivatives)": P.IFR["K-DTF derivatives"] * dtf,
         "K-COH (client orders handled, derivatives)": P.IFR["K-COH derivatives"] * coh,
         "K-TCD (LP counterparty default)": P.IFR["K-TCD alpha"] * ev * P.IFR["K-TCD risk factor (institutions)"]}
    k_total = sum(k.values())
    FOR = fixed_overheads_eur / 4
    req = max(P.IFR["PMR"], FOR, k_total)
    return {"K-factors": pd.Series(k), "K-factor requirement": k_total, "PMR": P.IFR["PMR"], "FOR": FOR,
            "own funds requirement": req, "binding": ["PMR", "FOR", "K-factors"][int(np.argmax([P.IFR["PMR"], FOR, k_total]))],
            "own funds": own_funds_eur, "own funds ratio": own_funds_eur / req, "NPR detail": npr,
            "liquidity requirement (1/3 FOR)": FOR / 3, "liquid assets": liquid_assets_eur,
            "client money (EUR)": cmh, "average daily trading flow (EUR)": dtf, "average daily client orders (EUR)": coh}
