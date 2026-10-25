"""
Instruments, client segments, limits and regulatory parameters of the demo broker (a CySEC-licensed CFD
provider reporting in EUR, trading in USD).

Regulatory sources
------------------
* ESMA product intervention on CFDs (2018), made permanent by national measures (CySEC): initial margin
  protection by underlying — 30:1 major currency pairs; 20:1 non-major pairs, gold and major indices;
  10:1 commodities other than gold and non-major indices; 5:1 individual equities and other values;
  2:1 cryptocurrencies — margin close-out at 50 % of the required margin, negative balance protection per account.
* Investment Firms Regulation (EU) 2019/2033 (IFR) and Directive (EU) 2019/2034 (IFD): own funds requirement =
  highest of the permanent minimum capital (EUR 750,000 for dealing on own account), the fixed overheads
  requirement (one quarter of the previous year's fixed overheads) and the K-factor requirement; K-factor
  coefficients (art. 15): K-CMH 0.4 % segregated / 0.5 % non-segregated, K-COH and K-DTF 0.1 % cash trades /
  0.01 % derivatives; K-TCD = 1.2 × exposure value × risk factor (1.6 % credit institutions and investment firms);
  K-NPR under the CRR standardised approach (simplified here).
"""
from __future__ import annotations

import pandas as pd

# name: class, price, daily vol, contract size (units per lot), spread (price units), ESMA leverage (retail),
#       swap long / short (% per year charged to client), K-NPR treatment, base ccy, quote ccy
INSTRUMENTS = pd.DataFrame([
    ("EURUSD", "FX major", 1.085, 0.0050, 100_000, 0.00008, 30, -0.030, 0.008, "FX", "EUR", "USD"),
    ("GBPUSD", "FX major", 1.270, 0.0058, 100_000, 0.00012, 30, -0.025, 0.006, "FX", "GBP", "USD"),
    ("USDJPY", "FX major", 150.0, 0.0060, 100_000, 0.012, 30, 0.020, -0.045, "FX", "USD", "JPY"),
    ("AUDUSD", "FX major", 0.660, 0.0065, 100_000, 0.00012, 30, -0.020, 0.004, "FX", "AUD", "USD"),
    ("USDCHF", "FX major", 0.880, 0.0055, 100_000, 0.00014, 30, 0.015, -0.040, "FX", "USD", "CHF"),
    ("EURCHF", "FX non-major", 0.955, 0.0040, 100_000, 0.00018, 20, 0.005, -0.030, "FX", "EUR", "CHF"),
    ("USDTRY", "FX non-major", 34.0, 0.0090, 100_000, 0.020, 20, 0.250, -0.380, "FX", "USD", "TRY"),
    ("XAUUSD", "Gold", 2400.0, 0.0100, 100, 0.30, 20, -0.040, 0.010, "Gold", "XAU", "USD"),
    ("WTI", "Commodity", 75.0, 0.0220, 1_000, 0.03, 10, -0.050, 0.010, "Commodity", "", "USD"),
    ("US500", "Major index", 5500.0, 0.0105, 1, 0.50, 20, -0.060, 0.020, "Equity index", "", "USD"),
    ("NAS100", "Major index", 19500.0, 0.0135, 1, 1.50, 20, -0.060, 0.020, "Equity index", "", "USD"),
    ("GER40", "Major index", 18500.0, 0.0110, 1, 1.20, 20, -0.050, 0.015, "Equity index", "", "EUR"),
    ("BTCUSD", "Crypto", 65000.0, 0.0330, 1, 40.0, 2, -0.180, -0.180, "Crypto", "", "USD"),
    ("ETHUSD", "Crypto", 3200.0, 0.0420, 1, 3.0, 2, -0.180, -0.180, "Crypto", "", "USD"),
], columns=["instrument", "class", "price", "vol", "contract", "spread", "leverage", "swap_long", "swap_short",
            "npr", "base", "quote"]).set_index("instrument")

# correlation clusters for the simulated market history
CLUSTERS = {"EURUSD": "usd", "GBPUSD": "usd", "AUDUSD": "risk", "USDJPY": "jpy", "USDCHF": "chf", "EURCHF": "chf",
            "USDTRY": "em", "XAUUSD": "gold", "WTI": "oil", "US500": "eq", "NAS100": "eq", "GER40": "eq",
            "BTCUSD": "crypto", "ETHUSD": "crypto"}

# client segments: share of clients, trades per client per month, lot scale, holding period (log-mean seconds),
# alpha profile: amplitude (in spreads) and time constant (seconds), share of professional clients
SEGMENTS = pd.DataFrame([
    ("Retail – directional", 0.62, 8, 1.0, 11.3, -20.0, 86_400, 0.00),
    ("Retail – active", 0.18, 30, 0.8, 8.5, -3.0, 3_600, 0.02),
    ("Scalper", 0.08, 120, 1.5, 4.5, 1.2, 60, 0.10),
    ("News trader", 0.05, 20, 2.0, 6.0, 1.8, 120, 0.15),
    ("Latency arbitrage", 0.02, 250, 3.0, 2.0, 2.5, 2, 0.30),
    ("Informed / professional", 0.05, 15, 8.0, 11.8, 6.0, 14_400, 0.80),
], columns=["segment", "share", "trades_pm", "lots", "log_hold", "alpha_spreads", "alpha_tau", "pro_share"]).set_index("segment")
SEGMENT_INSTR = {   # instrument preferences
    "Retail – directional": ["EURUSD", "GBPUSD", "XAUUSD", "US500", "NAS100", "BTCUSD", "ETHUSD", "WTI", "GER40", "USDJPY"],
    "Retail – active": ["EURUSD", "GBPUSD", "XAUUSD", "NAS100", "GER40", "BTCUSD", "USDJPY"],
    "Scalper": ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "NAS100"],
    "News trader": ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "US500", "WTI"],
    "Latency arbitrage": ["EURUSD", "GBPUSD", "XAUUSD", "GER40", "USDJPY"],
    "Informed / professional": ["EURUSD", "USDJPY", "XAUUSD", "US500", "WTI", "USDCHF", "AUDUSD", "EURCHF", "USDTRY"],
}

MARKOUT_HORIZONS = {"1s": 1, "5s": 5, "30s": 30, "1m": 60, "5m": 300, "30m": 1_800, "1h": 3_600, "4h": 14_400,
                    "1d": 86_400}

# limits of the dealing desk (USD)
LIMITS = {
    "VaR 99 % 1-day": 1_500_000,
    "Net notional by instrument": {"FX major": 40e6, "FX non-major": 8e6, "Gold": 15e6, "Commodity": 6e6,
                                   "Major index": 15e6, "Crypto": 4e6},
    "Gross net notional (all instruments)": 120e6,
    "Single client share of net exposure": 0.25,
    "Daily stop-loss": 1_000_000,
}

# stylised stress scenarios (instantaneous shocks, no time to close out; inspired by historical episodes)
SCENARIOS = pd.DataFrame({
    "CHF de-peg (SNB-type, Jan 2015)": {"USDCHF": -0.18, "EURCHF": -0.20, "EURUSD": -0.02},
    "Brexit referendum-type (Jun 2016)": {"GBPUSD": -0.10, "EURUSD": -0.03, "USDJPY": -0.04, "US500": -0.04,
                                          "NAS100": -0.04, "GER40": -0.07, "XAUUSD": 0.05},
    "Covid crash-type (Mar 2020)": {"US500": -0.12, "NAS100": -0.12, "GER40": -0.12, "WTI": -0.25, "AUDUSD": -0.05,
                                    "XAUUSD": -0.04, "BTCUSD": -0.37, "ETHUSD": -0.42, "USDJPY": -0.02},
    "Oil collapse-type (Apr 2020)": {"WTI": -0.45},
    "Crypto crash-type (Nov 2022)": {"BTCUSD": -0.25, "ETHUSD": -0.30},
    "Turkish lira crisis-type (Dec 2021)": {"USDTRY": 0.30},
    "USD flash rally": {"EURUSD": -0.03, "GBPUSD": -0.035, "AUDUSD": -0.04, "USDJPY": 0.03, "USDCHF": 0.03,
                        "XAUUSD": -0.03},
}).fillna(0.0)

IFR = {"PMR": 750_000, "K-CMH segregated": 0.004, "K-CMH non-segregated": 0.005, "K-COH derivatives": 0.0001,
       "K-DTF derivatives": 0.0001, "K-TCD alpha": 1.2, "K-TCD risk factor (institutions)": 0.016,
       "NPR FX": 0.08, "NPR commodity net": 0.15, "NPR commodity gross": 0.03, "NPR equity index": 0.08,
       "NPR crypto (assumption)": 1.00}
