"""
Market risk of the broker's net book: historical, parametric (equally weighted or EWMA) and Monte-Carlo
(Student t) VaR, expected shortfall, stressed VaR, component VaR, backtesting of the current book, limits,
stress scenarios with gap risk and negative balance protection.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import market as M
from . import params as P


def pnl_vector(pos_usd: pd.Series, rets: pd.DataFrame) -> np.ndarray:
    """Hypothetical daily P&L of a static book of USD notionals under historical returns (simple returns)."""
    R = np.expm1(rets[pos_usd.index].to_numpy())
    return R @ pos_usd.to_numpy()


def var_es(pnl, alpha=0.99, es_alpha=0.975):
    losses = -np.asarray(pnl)
    v = float(np.quantile(losses, alpha))
    q = np.quantile(losses, es_alpha)
    return v, float(losses[losses >= q].mean())


def covariance(rets: pd.DataFrame, window: int = 250, ewma: float | None = None) -> np.ndarray:
    R = np.expm1(rets.iloc[-window:].to_numpy())
    if ewma:
        w = ewma ** np.arange(len(R))[::-1]
        w /= w.sum()
        Rc = R - (w[:, None] * R).sum(0)
        return (Rc * w[:, None]).T @ Rc
    return np.cov(R.T)


def measures(pos_usd: pd.Series, rets: pd.DataFrame, alpha=0.99, window: int = 500, n_mc: int = 50_000, seed: int = 1):
    hist = rets.iloc[-window:]
    pnl = pnl_vector(pos_usd, hist)
    hs_v, hs_es = var_es(pnl, alpha)
    z = stats.norm.ppf(alpha)
    S = covariance(rets, 250)
    Se = covariance(rets, 250, 0.94)
    x = pos_usd.to_numpy()
    sd, sde = float(np.sqrt(x @ S @ x)), float(np.sqrt(x @ Se @ x))
    rng = np.random.default_rng(seed)
    df = 5
    Lc = np.linalg.cholesky(S + 1e-12 * np.eye(len(x)))
    Z = rng.standard_normal((n_mc, len(x))) @ Lc.T / np.sqrt(rng.chisquare(df, (n_mc, 1)) / df) * np.sqrt((df - 2) / df)
    mc_v, mc_es = var_es(Z @ x, alpha)
    sv = stressed_var(pos_usd, rets, alpha)
    es_z = stats.norm.pdf(stats.norm.ppf(0.975)) / 0.025
    return pd.DataFrame([
        {"method": f"Historical simulation ({window} days)", "VaR 1-day": hs_v, "ES 97.5 %": hs_es},
        {"method": "Parametric, 250 days", "VaR 1-day": z * sd, "ES 97.5 %": es_z * sd},
        {"method": "Parametric, EWMA λ = 0.94", "VaR 1-day": z * sde, "ES 97.5 %": es_z * sde},
        {"method": "Monte-Carlo, Student t (5 df)", "VaR 1-day": mc_v, "ES 97.5 %": mc_es},
        {"method": "Stressed VaR (worst 250-day window)", "VaR 1-day": sv["VaR"], "ES 97.5 %": sv["ES"]},
    ]).assign(**{"VaR 10-day (√10)": lambda d: d["VaR 1-day"] * np.sqrt(10)})


def stressed_var(pos_usd, rets, alpha=0.99, window=250):
    pnl = pnl_vector(pos_usd, rets)
    best = {"VaR": 0.0, "ES": 0.0, "start": None}
    for s in range(0, len(pnl) - window + 1, 5):
        v, e = var_es(pnl[s:s + window], alpha)
        if v > best["VaR"]:
            best = {"VaR": v, "ES": e, "start": rets.index[s], "end": rets.index[s + window - 1]}
    return best


def components(pos_usd: pd.Series, rets: pd.DataFrame, alpha=0.99, window=500) -> pd.DataFrame:
    S = covariance(rets, 250)
    x = pos_usd.to_numpy()
    sd = np.sqrt(x @ S @ x)
    z = stats.norm.ppf(alpha)
    comp = x * (S @ x) / sd * z if sd > 0 else np.zeros_like(x)
    tot, _ = var_es(pnl_vector(pos_usd, rets.iloc[-window:]), alpha)
    incr = []
    for i in range(len(x)):
        p = pos_usd.copy()
        p.iloc[i] = 0.0
        incr.append(tot - var_es(pnl_vector(p, rets.iloc[-window:]), alpha)[0])
    vol = np.sqrt(np.diag(S))
    return pd.DataFrame({"net notional (USD)": x, "standalone VaR": np.abs(x) * vol * z, "component VaR (parametric)": comp,
                         "share of VaR": comp / comp.sum() if comp.sum() else 0.0, "incremental VaR (historical)": incr},
                        index=pos_usd.index)


def backtest(pos_usd: pd.Series, rets: pd.DataFrame, alpha=0.99, test=250, window=500) -> dict:
    """Hypothetical backtest: VaR of the current book from the preceding `window` days vs next-day P&L."""
    pnl = pnl_vector(pos_usd, rets)
    n = len(pnl)
    vs = np.array([var_es(pnl[i - window:i], alpha)[0] for i in range(n - test, n)])
    real = pnl[n - test:]
    exc = int((-real > vs).sum())
    p = 1 - alpha
    x, T = exc, test
    lr = -2 * (np.log((1 - p) ** (T - x) * p ** x) - np.log((1 - x / T) ** (T - x) * (x / T) ** x if 0 < x < T else 1.0))
    zone = "green" if exc <= 4 else ("amber" if exc <= 9 else "red")
    return {"dates": rets.index[n - test:], "pnl": real, "var": vs, "exceptions": exc, "Kupiec LR": float(lr),
            "Kupiec p-value": float(stats.chi2.sf(lr, 1)), "Basel zone": zone}


def limits(exp: pd.DataFrame, var_1d: float, daily_pnl: float, client_share: float) -> pd.DataFrame:
    L = P.LIMITS
    rows = [{"limit": "VaR 99 % 1-day", "value": var_1d, "limit value": L["VaR 99 % 1-day"]},
            {"limit": "Gross of net notionals", "value": float(exp["broker net notional (USD)"].abs().sum()),
             "limit value": L["Gross net notional (all instruments)"]},
            {"limit": "Largest client share of net exposure", "value": client_share,
             "limit value": L["Single client share of net exposure"]},
            {"limit": "Daily loss (last day)", "value": max(-daily_pnl, 0.0), "limit value": L["Daily stop-loss"]}]
    for i, r in exp.iterrows():
        rows.append({"limit": f"Net notional {i}", "value": abs(r["broker net notional (USD)"]), "limit value": r["limit (USD)"]})
    df = pd.DataFrame(rows)
    df["utilisation"] = df["value"] / df["limit value"]
    df["status"] = np.select([df["utilisation"] >= 1, df["utilisation"] >= 0.8], ["breach", "warning"], "ok")
    return df


def stress(flow, books: pd.Series, hedge_units: dict, scenarios: pd.DataFrame) -> pd.DataFrame:
    """Instantaneous gap: each client position re-priced; client losses beyond the account equity are not
    collectable (negative balance protection). For A-booked positions the broker's LP hedge loses the full amount,
    so the uncollectable part is a loss; for B-booked positions it reduces the broker's gain."""
    from .book import client_positions
    op = client_positions(flow, books)
    eq = flow.clients.set_index("client")["equity"]
    px = flow.px
    usdq = M.usd_per_quote(px)
    q_usd = op["instrument"].map(lambda i: usdq[P.INSTRUMENTS.loc[i, "quote"]]).to_numpy()
    rows = []
    for name in scenarios.columns:
        shock = op["instrument"].map(scenarios[name]).fillna(0.0).to_numpy()
        cp = op["signed_units"].to_numpy() * px_arr(op, px) * shock * q_usd     # client P&L, USD
        df = pd.DataFrame({"client": op["client"].to_numpy(), "book": op["book"].to_numpy(), "pnl": cp})
        per = df.groupby(["client", "book"])["pnl"].sum().unstack(fill_value=0.0)
        for c in ("A", "B"):
            if c not in per:
                per[c] = 0.0
        tot = per["A"] + per["B"]
        e = per.index.map(eq).to_numpy()
        short = np.maximum(-tot.to_numpy() - e, 0.0)                       # uncollectable loss
        wa = np.where(tot < 0, np.clip(-per["A"].to_numpy(), 0, None) / np.maximum(-tot.to_numpy(), 1e-9), 0.0)
        wa = np.clip(wa, 0, 1)
        nbp_a, nbp_b = short * wa, short * (1 - wa)
        h = sum(u * px[i] * scenarios[name].get(i, 0.0) * usdq[P.INSTRUMENTS.loc[i, "quote"]] for i, u in hedge_units.items())
        bb = float(-per["B"].sum() - nbp_b.sum())
        rows.append({"scenario": name, "B-book P&L": bb, "hedge P&L": float(h), "NBP loss on A-book clients": 0.0 - float(nbp_a.sum()),
                     "total broker P&L": bb + float(h) - float(nbp_a.sum()),
                     "of which uncollectable on B-book": float(-nbp_b.sum()),
                     "accounts in negative balance": int((short > 0).sum())})
    return pd.DataFrame(rows)


def px_arr(op, px):
    return op["instrument"].map(px).to_numpy()
