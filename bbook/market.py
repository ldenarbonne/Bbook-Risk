"""
Simulated market history: three years of daily log-returns with a factor correlation structure (USD, risk-on,
crypto, technology), Student t innovations, a stress regime with higher volatility and correlation, and a few
jumps. Prices end at the current levels of the instrument table. Also: conversion of positions to USD.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import params as P

LOADINGS = {
    "usd": {"EURUSD": 0.80, "GBPUSD": 0.65, "AUDUSD": 0.50, "USDCHF": -0.65, "USDJPY": -0.35, "XAUUSD": 0.35,
            "EURCHF": 0.10, "USDTRY": -0.15},
    "risk": {"AUDUSD": 0.45, "US500": 0.85, "NAS100": 0.80, "GER40": 0.75, "WTI": 0.40, "BTCUSD": 0.40, "ETHUSD": 0.40,
             "USDJPY": 0.30, "EURCHF": 0.30, "XAUUSD": -0.05},
    "crypto": {"BTCUSD": 0.75, "ETHUSD": 0.80},
    "tech": {"NAS100": 0.45, "US500": 0.25},
}


def correlation(names) -> np.ndarray:
    k = len(names)
    Lm = np.array([[LOADINGS[f].get(n, 0.0) for f in LOADINGS] for n in names])
    C = Lm @ Lm.T
    np.fill_diagonal(C, 1.0)
    w, V = np.linalg.eigh(C)
    C = V @ np.diag(np.clip(w, 1e-6, None)) @ V.T
    d = np.sqrt(np.diag(C))
    return C / np.outer(d, d) if k else C


def history(n_days: int = 756, seed: int = 11, end: str = "2026-10-23") -> pd.DataFrame:
    """Daily log-returns (rows = business days, columns = instruments)."""
    rng = np.random.default_rng(seed)
    ins = P.INSTRUMENTS
    names = list(ins.index)
    C = correlation(names)
    Lc = np.linalg.cholesky(C)
    df = 5
    z = rng.standard_normal((n_days, len(names))) @ Lc.T
    z /= np.sqrt(rng.chisquare(df, (n_days, 1)) / df) * np.sqrt(df / (df - 2))
    vol = ins["vol"].to_numpy()
    regime = np.ones(n_days)
    s0 = int(n_days * 0.45)
    regime[s0:s0 + 60] = 2.3                                 # stress period (a quarter of high volatility)
    r = z * vol * regime[:, None]
    # common shock inside the stress period (risk-off day) and a few idiosyncratic jumps
    j = s0 + 12
    r[j] += np.array([{"US500": -0.07, "NAS100": -0.08, "GER40": -0.07, "WTI": -0.12, "BTCUSD": -0.20, "ETHUSD": -0.24,
                       "AUDUSD": -0.03, "USDJPY": -0.025, "XAUUSD": -0.02}.get(n, 0.0) for n in names])
    for n, day, size in (("EURCHF", int(n_days * 0.2), -0.035), ("USDTRY", int(n_days * 0.7), 0.12),
                         ("BTCUSD", int(n_days * 0.85), -0.15), ("GBPUSD", int(n_days * 0.3), -0.04)):
        r[day, names.index(n)] += size
    dates = pd.bdate_range(end=end, periods=n_days)
    return pd.DataFrame(r, index=dates, columns=names)


def prices(rets: pd.DataFrame) -> pd.DataFrame:
    p0 = P.INSTRUMENTS["price"]
    cum = rets.cumsum()
    return np.exp(cum - cum.iloc[-1]) * p0


def usd_per_quote(px: pd.Series | dict) -> dict:
    """USD value of one unit of each quote currency, from current prices."""
    px = dict(px)
    return {"USD": 1.0, "JPY": 1 / px["USDJPY"], "CHF": 1 / px["USDCHF"], "TRY": 1 / px["USDTRY"], "EUR": px["EURUSD"]}


def notional_usd(units, instrument: str, px: dict) -> np.ndarray:
    """USD notional of a position of `units` (signed) in an instrument."""
    q = P.INSTRUMENTS.loc[instrument, "quote"]
    return np.asarray(units) * px[instrument] * usd_per_quote(px)[q]
