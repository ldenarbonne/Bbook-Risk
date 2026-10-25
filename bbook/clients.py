"""
Synthetic client base and trade flow of the broker over the last three months.

Each client belongs to a behavioural segment (directional retail, active retail, scalper, news trader, latency
arbitrage, informed / professional), which sets the trading frequency, size, holding period and the "alpha
profile": the expected price move in the client's favour after the trade, a(τ) = A · spread · (1 − e^(−τ/τ_a)).
Retail flow has a negative A (the client loses on average beyond the spread); toxic flow has a positive A
concentrated at short horizons (latency arbitrage: seconds; news: minutes) or at long horizons (informed).
For every trade the mid-price path is simulated at the markout horizons, at the holding period and at the
reporting date, so that markouts, realised and unrealised P&L are consistent.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import market as M
from . import params as P

HORIZONS = np.array(list(P.MARKOUT_HORIZONS.values()), dtype=float)


@dataclass
class Flow:
    clients: pd.DataFrame
    trades: pd.DataFrame
    end: pd.Timestamp
    px: dict


def generate(n_clients: int = 2000, months: int = 3, seed: int = 5, rets: pd.DataFrame | None = None) -> Flow:
    rng = np.random.default_rng(seed)
    rets = M.history() if rets is None else rets
    prices = M.prices(rets)
    end = prices.index[-1] + pd.Timedelta(hours=22)
    px_end = prices.iloc[-1].to_dict()
    days = prices.index[-(months * 21):]
    ins = P.INSTRUMENTS
    seg = P.SEGMENTS

    # ------------------------------------------------------------- clients
    segs = rng.choice(seg.index, n_clients, p=seg["share"] / seg["share"].sum())
    bal_med = {"Retail – directional": 3_000, "Retail – active": 5_000, "Scalper": 12_000, "News trader": 15_000,
               "Latency arbitrage": 60_000, "Informed / professional": 400_000}
    balance = np.array([bal_med[s] for s in segs]) * rng.lognormal(0, 0.8, n_clients)
    pro = rng.random(n_clients) < seg.loc[segs, "pro_share"].to_numpy()
    activity = rng.lognormal(0, 0.5, n_clients)
    raw = np.isin(segs, ["Scalper", "Latency arbitrage", "Informed / professional"]) | (rng.random(n_clients) < 0.05)
    clients = pd.DataFrame({"client": [f"CL{i:05d}" for i in range(n_clients)], "segment": segs,
                            "category": np.where(pro, "Professional", "Retail"), "account": np.where(raw, "Raw (commission)", "Standard"),
                            "deposit": balance.round(2)})

    # ------------------------------------------------------------- trades
    lam = seg.loc[segs, "trades_pm"].to_numpy() * months * activity
    n_tr = rng.poisson(lam)
    ci = np.repeat(np.arange(n_clients), n_tr)
    n = len(ci)
    s = segs[ci]
    instr = np.empty(n, dtype=object)
    for sg, lst in P.SEGMENT_INSTR.items():
        m = s == sg
        w = np.linspace(1.0, 0.3, len(lst))
        instr[m] = rng.choice(lst, m.sum(), p=w / w.sum())
    side = np.where(rng.random(n) < 0.5, 1, -1)
    # open time: business day of the period + time of day
    day = rng.integers(0, len(days), n)
    t_open = days.values[day] + (rng.uniform(1, 22, n) * 3600e9).astype("timedelta64[ns]")
    hold = np.exp(rng.normal(seg.loc[s, "log_hold"].to_numpy(), 1.0))
    hold = np.minimum(hold, 20 * 86400)
    price0 = prices.values[len(prices) - len(days) + day, [prices.columns.get_loc(i) for i in instr]]
    contract = ins.loc[instr, "contract"].to_numpy()
    lots = seg.loc[s, "lots"].to_numpy() * rng.lognormal(0, 0.6, n)
    # size capped by the margin the account can post at the regulatory leverage (pros get 5x the retail leverage)
    lev = ins.loc[instr, "leverage"].to_numpy() * np.where(pro[ci], 5, 1)
    usdq = M.usd_per_quote(px_end)
    q_usd = np.array([usdq[q] for q in ins.loc[instr, "quote"]])
    max_units = 0.6 * balance[ci] * lev / (price0 * q_usd)
    units = np.minimum(lots * contract, max_units)
    units = np.maximum(np.round(units / (contract * 0.01)) * contract * 0.01, contract * 0.01)
    notional = units * price0 * q_usd

    # ------------------------------------------------------------- price paths at markout horizons, hold, report date
    t_rep = (end - pd.to_datetime(t_open)).total_seconds().to_numpy()
    tau_rep = np.clip(np.minimum(hold, t_rep), 1, None)
    T = np.column_stack([np.tile(HORIZONS, (n, 1)), hold, tau_rep])
    order = np.argsort(T, axis=1)
    Ts = np.take_along_axis(T, order, axis=1)
    dt = np.diff(np.column_stack([np.zeros(n), Ts]), axis=1)
    vol = ins.loc[instr, "vol"].to_numpy()
    W = np.cumsum(rng.standard_normal(Ts.shape) * np.sqrt(dt / 86400), axis=1) * vol[:, None]
    noise = np.empty_like(W)
    np.put_along_axis(noise, order, W, axis=1)
    spread_full = ins.loc[instr, "spread"].to_numpy() * np.where(raw[ci], 0.4, 1.0)
    srel = ins.loc[instr, "spread"].to_numpy() / price0
    A = seg.loc[s, "alpha_spreads"].to_numpy() * rng.lognormal(0, 0.3, n)
    tau_a = seg.loc[s, "alpha_tau"].to_numpy()
    alpha = A[:, None] * srel[:, None] * (1 - np.exp(-T / tau_a[:, None]))
    move = alpha + side[:, None] * noise                       # signed move in the client's favour (relative)
    k = len(HORIZONS)
    markout = move[:, :k] / srel[:, None]                      # in spreads
    closed = hold <= t_rep
    m_end = np.where(closed, move[:, k], move[:, k + 1])
    pnl_mid = m_end * price0 * units * q_usd                   # client mid-to-mid P&L, USD
    spread_cost = spread_full * units * q_usd * np.where(closed, 1.0, 0.5)
    commission = np.where(raw[ci], 6.0 * notional / 100_000, 0.0) * np.where(closed, 1.0, 0.5)
    nights = np.floor(np.where(closed, hold, tau_rep) / 86400)
    swap_rate = np.where(side > 0, ins.loc[instr, "swap_long"].to_numpy(), ins.loc[instr, "swap_short"].to_numpy())
    swap = notional * swap_rate * nights / 360                 # credited to (+) or charged to (−) the client
    trades = pd.DataFrame({
        "trade": np.arange(n), "client": clients["client"].to_numpy()[ci], "segment": s, "instrument": instr,
        "side": side, "units": units, "open_time": pd.to_datetime(t_open), "hold_s": hold,
        "close_time": pd.to_datetime(t_open) + pd.to_timedelta(hold, unit="s"), "open_mid": price0,
        "notional_usd": notional, "status": np.where(closed, "closed", "open"),
        "client_pnl_mid": pnl_mid, "spread_paid": spread_cost, "commission": commission, "swap": swap,
    })
    trades["client_pnl"] = trades["client_pnl_mid"] - trades["spread_paid"] - trades["commission"] + trades["swap"]
    for j, h in enumerate(P.MARKOUT_HORIZONS):
        trades[f"mo_{h}"] = markout[:, j]
    trades = trades.sort_values("open_time").reset_index(drop=True)
    pnl = trades.groupby("client")["client_pnl"].sum()
    clients["trading_pnl"] = clients["client"].map(pnl).fillna(0.0)
    # equity: a client who lost the deposit re-deposits (equity floored at 20 % of the initial deposit)
    eq = np.maximum(clients["deposit"] + clients["trading_pnl"], 0.2 * clients["deposit"])
    # accounts below 50 % of the required margin would have been closed out: margin levels at the reporting date
    # are drawn between 60 % and 400 % of the required margin of the open positions (top-ups by the clients)
    op = trades[trades["status"] == "open"]
    req = (op["notional_usd"] / (op["instrument"].map(ins["leverage"])
                                 * np.where(op["client"].map(clients.set_index("client")["category"]) == "Professional", 5, 1)))
    req = req.groupby(op["client"]).sum().reindex(clients["client"]).fillna(0.0).to_numpy()
    level = rng.uniform(0.6, 4.0, n_clients)
    clients["equity"] = np.maximum(eq, level * req).round(2)
    return Flow(clients, trades, end, px_end)


def open_positions(flow: Flow) -> pd.DataFrame:
    t = flow.trades
    return t[t["status"] == "open"].copy()


def current_mid(flow: Flow, pos: pd.DataFrame) -> np.ndarray:
    """Mark of open positions: the end-of-day price of the reporting date."""
    return pos["instrument"].map(flow.px).to_numpy()
