"""Market and flow simulation, toxicity detection, exposure, routing, VaR, stress (negative balances), IFR and the app."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bbook import book as B
from bbook import clients as CL
from bbook import market as M
from bbook import params as P
from bbook import regulatory as G
from bbook import risk as R
from bbook import toxicity as T


@pytest.fixture(scope="module")
def world():
    rets = M.history()
    flow = CL.generate(1000, 3, 7, rets)
    met = T.client_metrics(flow)
    return rets, flow, met


def test_market_history(world):
    rets, flow, _ = world
    assert np.allclose(M.prices(rets).iloc[-1], P.INSTRUMENTS["price"])
    ann = rets.std() * np.sqrt(252)
    assert ann["BTCUSD"] > ann["EURUSD"] * 3
    c = rets.corr()
    assert c.loc["US500", "NAS100"] > 0.7 and c.loc["EURUSD", "USDCHF"] < -0.3
    assert M.notional_usd(1_000_000, "USDJPY", flow.px) == pytest.approx(1_000_000)        # base USD
    assert M.notional_usd(1, "EURUSD", flow.px) == pytest.approx(flow.px["EURUSD"])


def test_flow_consistency(world):
    _, flow, _ = world
    t = flow.trades
    assert (t["units"] > 0).all() and set(t["status"]) == {"open", "closed"}
    assert np.allclose(t["client_pnl"], t["client_pnl_mid"] - t["spread_paid"] - t["commission"] + t["swap"])
    # margin: notional at most 60 % of the deposit times the regulatory leverage (5x for professionals)
    c = flow.clients.set_index("client")
    lev = t["instrument"].map(P.INSTRUMENTS["leverage"]) * np.where(t["client"].map(c["category"]) == "Professional", 5, 1)
    assert (t["notional_usd"] <= 0.6 * t["client"].map(c["deposit"]) * lev * 1.06 + 0.01 * 100_000 * 2).all()
    # retail loses on average, toxic segments win
    seg = t.groupby("segment")["client_pnl"].sum()
    assert seg["Retail – directional"] < 0 and seg["Latency arbitrage"] > 0
    assert 0.6 < G.losing_retail_share(flow) < 0.9


def test_markouts_and_classification(world):
    _, flow, met = world
    cur = T.markout_curves(flow)
    assert cur.loc["Latency arbitrage", "5s"] > 1.5 and cur.loc["Retail – directional", "1d"] < -5
    ct = pd.crosstab(met["segment (true)"], met["flow class"])
    assert ct.loc["Latency arbitrage", "Latency arbitrage"] == ct.loc["Latency arbitrage"].sum()
    assert ct.loc["Scalper", "Short-term informed (news / scalping)"] / ct.loc["Scalper"].sum() > 0.9
    retail = met[met["segment (true)"].str.startswith("Retail")]
    assert (retail["flow class"] == "Benign / uninformed").mean() > 0.85
    assert met.groupby("segment (true)")["toxicity score"].median()["Latency arbitrage"] > 95


def test_routing_improves_broker_pnl(world):
    _, flow, met = world
    allb = B.route(met, 101, None)
    routed = B.route(met, 60, 250_000)
    assert (allb == "B").all()
    p = B.routed_pnl(met, routed)
    assert p["routed"] > p["all B-book"] and p["routed"] > p["all A-book"]
    pb = B.pnl_by(met, routed)
    assert pb.loc["Latency arbitrage", "share A-booked"] == 1.0
    assert pb["broker P&L (routed)"].sum() == pytest.approx(p["routed"])


def test_exposure_and_hedging(world):
    _, flow, met = world
    books = B.route(met)
    e = B.net_exposure(flow, books)
    op = B.client_positions(flow, books)
    b = op[op["book"] == "B"]
    for i in ("EURUSD", "XAUUSD"):
        assert e.loc[i, "broker net units"] == pytest.approx(-(b.loc[b["instrument"] == i, "signed_units"]).sum())
    full = B.net_exposure(flow, books, B.hedge_plan(e, "ratio", 1.0))
    assert np.allclose(full["broker net units"], 0.0)
    exc = B.net_exposure(flow, books, B.hedge_plan(e, "excess"))
    assert (exc["utilisation"] <= 1 + 1e-9).all()
    ce = B.currency_exposure(e, flow.px)
    fx_usd = sum(abs(e.loc[i, "broker net notional (USD)"]) for i in e.index if P.INSTRUMENTS.loc[i, "npr"] in ("FX", "Gold"))
    assert abs(ce).sum() <= 2 * fx_usd + 1


def test_var_methods_and_backtest(world):
    rets, flow, met = world
    pos = B.net_exposure(flow, B.route(met))["broker net notional (USD)"]
    m = R.measures(pos, rets).set_index("method")
    v = m["VaR 1-day"]
    assert (v > 0).all() and v.max() / v.min() < 3
    assert m.loc["Stressed VaR (worst 250-day window)", "VaR 1-day"] >= m.loc["Historical simulation (500 days)", "VaR 1-day"] * 0.9
    cv = R.components(pos, rets)
    par = m.loc["Parametric, 250 days", "VaR 1-day"]
    assert cv["component VaR (parametric)"].sum() == pytest.approx(par, rel=1e-6)
    bt = R.backtest(pos, rets)
    assert 0 <= bt["exceptions"] <= 15 and bt["Basel zone"] in ("green", "amber", "red")
    # a single long EURUSD position of 10 m: parametric VaR = 2.33 × 10 m × daily vol
    one = pd.Series(0.0, index=pos.index)
    one["EURUSD"] = 10e6
    sd = np.sqrt(R.covariance(rets)[0, 0])
    assert R.measures(one, rets).set_index("method").loc["Parametric, 250 days", "VaR 1-day"] == pytest.approx(2.3263 * 10e6 * sd, rel=1e-3)


def test_stress_negative_balance(world):
    _, flow, met = world
    books = B.route(met)
    sc = pd.DataFrame({"none": {"EURUSD": 0.0}, "crash": {i: -0.5 for i in P.INSTRUMENTS.index}}).fillna(0.0)
    res = R.stress(flow, books, {}, sc).set_index("scenario")
    assert res.loc["none", "total broker P&L"] == pytest.approx(0.0)
    assert res.loc["crash", "accounts in negative balance"] > 0
    assert res.loc["crash", "NBP loss on A-book clients"] <= 0
    e = B.net_exposure(flow, books)
    h = B.hedge_plan(e, "ratio", 1.0)
    hedged = R.stress(flow, books, h, sc).set_index("scenario")
    # a full hedge offsets the B-book move except the uncollectable part (gain forgone on clients in deficit)
    assert hedged.loc["crash", "B-book P&L"] + hedged.loc["crash", "hedge P&L"] == pytest.approx(
        hedged.loc["crash", "of which uncollectable on B-book"], rel=1e-6, abs=1.0)


def test_ifr_requirement(world):
    _, flow, met = world
    books = B.route(met)
    e = B.net_exposure(flow, books)
    ir = G.ifr(flow, books, e, fixed_overheads_eur=10e6, own_funds_eur=15e6)
    assert ir["own funds requirement"] == max(750_000, 2.5e6, ir["K-factor requirement"])
    assert ir["K-factors"]["K-CMH (client money held, segregated)"] == pytest.approx(0.004 * ir["client money (EUR)"])
    npr = ir["NPR detail"].set_index("risk")
    v = abs(e.loc["WTI", "broker net notional (USD)"]) / flow.px["EURUSD"]
    assert npr.loc["Commodity WTI", "charge"] == pytest.approx(0.18 * v)
    ac = G.esma_accounts(flow, books)
    assert (ac["margin level"] >= 0.5 - 1e-9).all()


def test_app_runs():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=900).run()
    assert not at.exception and not at.error
    next(r for r in at.sidebar.radio if r.label == "Hedge at liquidity providers").set_value("Excess over instrument limits").run()
    assert not at.exception and not at.error
