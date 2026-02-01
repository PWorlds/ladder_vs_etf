# Bond Ladder vs Bond ETF Simulation (v2.1)

A purely Monte Carlo-based simulation framework to compare the financial efficiency of **Bond Ladders** (buying individual bonds held to maturity) versus **Bond ETFs** (constant duration funds) under various market scenarios.

This script is calibrated with real-world parameters (January 2026 research) reflecting the European investor context (taxes, transaction costs, market yields).

## 🚀 Key Features

*   **Monte Carlo Core**: Simulates 30,000 independent market paths over 30 years using a stochastic model for Short Rates, Yield Curve Slope, and Inflation (correlated).
*   **Realistic Friction**:
    *   **Transaction Costs**: Bid-Ask spreads, commissions, and slippage tailored for Retail investors (Ladder) vs Institutional efficiency (ETFs).
    *   **Taxation**: Handles Coupon taxes and Capital Gains taxes (based on Italian *White List* 12.5% vs Standard 26% regimes).
    *   **Credit Risk & Defaults**:
    *   **Stochastic Defaults**: Simulates random issuer defaults based on default probabilities (Normal vs. Crisis regimes).
    *   **Recovery Rates**: Applies variable recovery rates (e.g., 30-60% for Corp) in case of default.
    *   **Concentration Risk**: Differentiates between holding a few issuers (Ladder) where a default is catastrophic, vs holding thousands (ETF) where it is a minor drag.
*   **Behavioral Modeling**:
    *   **"Laziness" Factor**: Simulates the cost of human procrastination. Investors don't reinvest cash instantly; this model applies a stochastic "cash drag" penalty based on a geometric distribution of reaction times.
*   **Strategies Compared**:
    *   **Classic Ladder**: Rolling ladder of individual bonds (e.g., 2-10 years).
    *   **Targeted Ladders**: Short-term (2-4y) and Long-term (6-10y).
    *   **Bond ETFs**: Various constant durations (Short 2y, Medium 6y, Long 10y) and combinations (Barbell).
    *   **"Lazy" Variants**: Measuring how much performance is lost if the investor is slow to execute trades.

## 📊 Scenarios

The script supports switching between different market environments:

1.  **EU_GOV** (Default): European Government Bonds (e.g., BTP/Bunds). Low risk, lower yields, favorable taxation (12.5%).
2.  **EU_CORP**: European Corporate Bonds. Higher credit risk, spreads, higher taxation (26%).
3.  **GLOBAL_GOV**: Global Government Bonds (Hedged).
4.  **GLOBAL_CORP**: Global Corporate Aggregate.

## ⚙️ Unpacking the Algorithm (Step-by-Step)

The simulation proceeds through the following computational phases:

1.  **Market Scenario Generation**:
    *   Generates correlated paths for Short Rates, Yield Curve Slope, and Inflation using a Cholesky-decomposed covariance matrix.
    *   The model assumes a stylized Yield Curve: $Rate(t) = ShortRate + Slope * \log(Maturity)$.

2.  **Portfolio Initialization**:
    *   **Ladder**: Buys individual bonds at market price (Ask) corresponding to specific maturities (e.g., 2, 4, 6... years). Applies transaction costs.
    *   **ETF**: Buys notional units of a constant-duration fund, paying spread and slippage.

3.  **Simulation Loop (Year 1 to 30)**:
    *   **Market Update**: New rates and inflation are applied.
    *   **Credit Events**: Random probability check for defaults.
        *   **Ladder**: Specific bonds might default (Face Value haircut).
        *   **ETF**: Portfolio value drops proportionally to the default rate relative to diversification size.
    *   **Cash Flow Processing**:
        *   **Coupons/Redemptions (Ladder)**: Matured bonds return capital (Taxed on CG). Coupons are collected (Taxed).
        *   **Yield (ETF)**: Fund generates internal yield (income return), price changes due to duration (price return), and roll-down return.
    *   **Behavioral Drag ("Laziness")**:
        *   Calculates a random delay in days before cash is reinvested.
        *   Applies a "Missed Yield" penalty proportional to the delay.
    *   **Reinvestment / Balancing**:
        *   **Accumulation**: New savings + coupons are invested into new Ladder rungs or ETF units.
        *   **Decumulation**: Capital is withdrawn to cover inflation-adjusted spending. If Cash < Spending, assets are sold (Shortest maturity first).

4.  **Result Aggregation**:
    *   Tracks wealth history, real (inflation-adjusted) values, and ruin occurrences across all 30,000 simulations.
    *   Computes CAGR and Efficiency metrics.

## 🛠️ Usage

### Prerequisites
*   Python 3.x
*   Numpy
*   Pandas
*   Matplotlib
*   Scipy (optional, for smoother distribution plots)

### Running the Simulation
Simply run the script directly:

```bash
python ladder_vs_etf_v2.py
```

By default, it runs the **EU_GOV** scenario. You can change `SELECTED_SCENARIO` in the script to test others.

### Output
The script generates:
1.  **Console Summary**: Tables showing Mean Final Wealth, Real Real Returns, Risk of Ruin, and Sharpe-like Efficiency metrics.
2.  **Lazy Penalty Analysis**: Quantifies the % capital lost due to inefficient reinvestment.
3.  **CSV Reports**: `results_ScenarioName.csv` containing detailed metrics.
4.  **Plots**:
    *   **Median Wealth Path**: Evolution of portfolio value over 30 years.
    *   **CAGR Distribution**: Probability density of annualized returns (useful to see tail risks).

## 🧮 How to Interpret Results

*   **Accumulation**: Focus on the **Mean Real (Eu)** and **CAGR**. Usually, higher duration ETFs win if costs are low and rates trend down, but Ladders offer stability.
*   **Decumulation**: Focus on **Ruin %** and **Min Real (Eu)**. This mode simulates withdrawing capital for living expenses. Stability (Sequence of Returns Risk) becomes critical here.
*   **Efficiency**: The ratio of Return/Risk. A strategy might yield less but be much safer (higher intervals of confidence).

## ⚠️ Disclaimer
This simulator uses stylized facts and historical calibrations. It is a research tool, not financial advice. Past correlations used in the Cholesky matrix generation may not hold in the future.
