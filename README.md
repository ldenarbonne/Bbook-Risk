# B-Book Risk Lab – Risk Management of a Forex / CFD Broker

An interactive Streamlit application for the dealing-desk risk management of a retail forex / CFD broker licensed by CySEC. It covers:

* **net exposure of the B-book** by instrument and by currency, with limits, history and hedge proposals;
* **client flow analysis ("toxicity")**: markouts from 1 second to 1 day, a toxicity score and a classification of the flow (latency arbitrage, news / scalping, informed);
* **A/B-book routing**: broker P&L of each policy and the routing frontier (P&L against VaR);
* **B-book P&L** by source: spread, commissions, financing, clients' trading P&L;
* **VaR and limits**: historical, parametric (EWMA), Monte-Carlo t, stressed VaR, component VaR, backtest and limit utilisation;
* **stress tests** with gap risk and **negative balance protection**;
* **regulation**: ESMA retail protections (leverage, margin close-out, risk warning) and **IFR own funds** (PMR, fixed overheads, K-factors).

**Live app:** `https://<your-app-name>.streamlit.app` *(to be completed after deployment)*

![tests](https://github.com/ldenarbonne/bbook-risk-lab/actions/workflows/tests.yml/badge.svg)

---

## 1. Features

| Tab | Content |
|---|---|
| **Net exposure** | KPIs. Net position and limits by instrument. Currency decomposition of FX and gold. End-of-day exposure over 3 months. Hedges and limit use. Largest client contributions |
| **Client flow & toxicity** | Markout curves by segment. Holding time vs 1-minute markout. Detected class vs true behaviour. Most toxic clients with their B-book and A-book P&L |
| **A/B-book routing** | All-B vs all-A vs routed P&L. Routing frontier against the toxicity threshold. P&L by segment |
| **B-book P&L** | Waterfall by source, daily and cumulative P&L, P&L by instrument and book |
| **VaR & limits** | Five VaR / ES methods, component and incremental VaR, backtest (Kupiec, Basel zones), limit dashboard |
| **Stress tests** | Editable stylised scenarios: CHF de-peg, Brexit, Covid, oil, crypto, Turkish lira, USD rally. Results split into B-book, hedges, negative-balance losses and the number of accounts in deficit |
| **CySEC / IFR & ESMA** | K-NPR, K-CMH, K-DTF, K-COH, K-TCD; own funds requirement and ratio; liquidity requirement. Leverage caps, margin levels, share of losing retail accounts for the risk warning |
| **Methodology** | Model, formulas, assumptions, references |

The sidebar holds the routing policy (toxicity threshold, A-booking of large professional accounts, A-book markup) and the hedging policy (none, excess over limits, or a share of the net exposure). Every tab follows these settings.

## 2. Simulation

* **Market.** Three years of daily returns for 14 CFDs: EURUSD, GBPUSD, USDJPY, AUDUSD, USDCHF, EURCHF, USDTRY, XAUUSD, WTI, US500, NAS100, GER40, BTCUSD and ETHUSD. Returns follow a factor correlation structure with Student t innovations, a stress quarter and jumps.
* **Clients.** 2,000 clients over 3 months, about 170,000 trades, in six segments with distinct frequency, size, holding period and alpha profile $a(\tau)=A\cdot\text{spread}\cdot(1-e^{-\tau/\tau_a})$:

| Segment | Alpha profile |
|---|---|
| Directional retail | Loses beyond the spread over a day |
| Active retail | Loses within the hour |
| Scalpers | Edge of about one spread at one minute |
| News traders | Edge at a few minutes |
| Latency arbitrage | Edge of 2–3 spreads within seconds |
| Informed / professional | Edge at hours and days |

* **Price paths and charges.** The path of every trade is simulated at the markout horizons, at closing and at the reporting date. Sizes respect the ESMA leverage caps. Spreads, commissions on raw accounts and swaps are charged.

## 3. Illustrative results (default settings)

| | Value |
|---|---|
| Broker P&L over 3 months – all B-book / all A-book / routed (score ≥ 60 + large professionals) | −1.9 m / +2.0 m / **+4.6 m** USD |
| A-booked under that policy | 18 % of clients, 82 % of notional |
| Net exposure after routing (gross of nets) | 30.9 m USD |
| VaR 99 % 1-day: historical / stressed | 310 k / 363 k USD (limit 1.5 m); backtest: 1 exception in 250 days, green zone |
| Limit breach | Largest client = 28.5 % of the net exposure (limit 25 %) |
| Worst stress | Oil collapse-type −0.8 m; CHF de-peg −0.34 m, with 0.67 m of client losses beyond equity |
| Retail accounts losing money | 73 % |
| IFR | K-factors 2.1 m EUR; fixed overheads requirement 2.5 m (binding); own funds ratio 600 % |

**Reading.**

* **Toxic flow costs the broker money.** Kept entirely in the B-book, latency arbitrage, news traders and informed accounts more than wipe out the margin made on retail. Routing them to liquidity providers turns the result positive and lowers the VaR.
* **Short-horizon edges are easy to detect.** Hundreds of trades with a markout of 1–3 spreads within a minute leave no doubt.
* **Long-horizon informed flow is not.** Its edge is statistically hard to prove over three months, which is why brokers also route by account size and monitor realised P&L.

## 4. Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Tests (9 tests, about 15 seconds) cover:

* simulation consistency (prices, P&L identity, margin caps);
* detection of the true segments;
* routing P&L;
* exposure and hedges;
* VaR (parametric hand calculation, Euler additivity, backtest);
* stress with negative balances (a full hedge leaves only the uncollectable part);
* the IFR identities and the app.

```bash
pip install -r requirements-dev.txt
pytest -q
```

## 5. Repository structure

```
app.py               Streamlit application (8 tabs)
bbook/params.py      Instruments, segments, limits, scenarios, ESMA and IFR parameters
bbook/market.py      Market history, prices, USD conversion
bbook/clients.py     Clients, trades, markouts, P&L, equity
bbook/toxicity.py    Client metrics, toxicity score, classification, markout curves
bbook/book.py        Routing, exposure, currency decomposition, history, hedges, P&L
bbook/risk.py        VaR / ES, stressed VaR, components, backtest, limits, stress with negative balances
bbook/regulatory.py  ESMA accounts and risk warning, IFR K-factors and own funds
tests/               pytest suite
```

## 6. Regulatory references

* ESMA (2018), product intervention measures on CFDs:
  * leverage caps of 30:1, 20:1, 10:1, 5:1 and 2:1;
  * margin close-out at 50 %;
  * negative balance protection;
  * risk warning.

  These measures have since been made permanent by national authorities, including CySEC.
* Regulation (EU) 2019/2033 (IFR), art. 11 and 15 (K-factor coefficients), and Directive (EU) 2019/2034 (IFD), with a permanent minimum capital of EUR 750,000 for dealing on own account.
* Kupiec, P. (1995), *Techniques for verifying the accuracy of risk measurement models*; BCBS (1996), backtesting framework.
* Butz, M. and Oomen, R. (2019), *Internalisation by electronic FX spot dealers*, Quantitative Finance.

## 7. Author

**Laurent Motais de Narbonne** – Risk management and quantitative finance · [ldenarbonne.github.io](https://ldenarbonne.github.io/)

Licence: MIT.
