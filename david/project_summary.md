# Project Summary

This project delivers a local-first AI assistant that turns annual reports into actionable insights, forecasts, and buy/hold/sell guidance. Users upload PDFs (10-K/annual reports), the app extracts narrative and tables, indexes them with embeddings, and answers questions with page-level citations. A free-tier market data pipeline adds stock history and lightweight forecasts; a “docs-only” mode keeps analysis strictly grounded in filings.

# What it is 

    An AI‑driven finance decision‑support and alerting system. It fuses market, macro, fundamental, and alternative data to deliver a single Risk Number (−100 no risk to +100 high risk), a 3‑month forecast, and clear Buy/Hold/Sell guidance. Anomaly detection prioritizes early warnings over naïve predictions.
     

# Who it’s for 

    Investors, analysts, and portfolio managers who want a consolidated signal with traceable drivers and automated alerts.

# Key data inputs 

    Market/technicals: prices, trends, momentum, volatility, chart features.
    Positioning/flow: COT data, “whale/shark” movement.
    Fundamentals: earnings/revenue, cash flow, balance sheet.
    Macroeconomics: central bank releases, economic calendar, exchange rates (e.g., ECB).
    Alternative/sentiment: news/RSS, social media, Google Trends, analyst reports, political monitoring.
    Example APIs/connectors: AlphaVantage, Myfxbook, Yahoo/Quandl‑like feeds.
    All feeds are normalized (time, currency, symbols) and validated.
     

# KPI focus 

    ROE, Profit Margins (gross/operating/net), Debt‑to‑Equity, Free Cash Flow. KPIs are trended and incorporated into decisions.
     

# Processing pipeline (request → decision) 

     Understand request (asset, horizon, risk profile).
     Fetch data via three blocks:
        getFinancialData (prices, fundamentals, COT)
        getEconomicalData (macro, central banks, calendars)
        getNewsAndTrends (news, social, trends)
         
     Feature extraction (technicals, macro surprises, text features).
     Sentiment analysis on news/social (direction, intensity, dispersion).
     Evaluate data and compute risk components and KPIs.
     Anomaly gateway:
        If anomaly = Yes → issue Warning with drivers (what, why).
        If anomaly = No → run forecast and decision engine.
         
     Output packaged and delivered via dashboard/API/alerts.
     

# Models 

    Outlier/Anomaly Detection: robust z‑scores, Isolation Forest, or residual analysis from forecasting models.
    Price Prediction (3‑month): ensemble (Prophet/XGBoost/LSTM) using technical, macro, positioning, sentiment, and KPI features; rolling backtests for weight updates.
    Investment Decision: meta‑model/rules combining expected return, drawdown risk, KPI trend, sentiment, and macro regime.
     

# Risk Number (−100 to +100) 

    Built from weighted factor groups scored on −1 to +1 and mapped to the scale:
        Difficult factors (highest weight): news flow, social sentiment, environmental/geopolitical shocks.
        Medium factors: trading network/COT, local politics, economic calendar.
        Small factors: innovations/inventions, analyst revisions.
         
    Guidance bands: ≤ −40 favorable; −40 to +20 normal; +20 to +60 elevated; ≥ +60 high risk → warnings dominate.
     

# Primary outputs 

    Risk Number with factor attribution.
    3‑month forecast range and confidence.
    Action: Buy/Hold/Sell, optional hedge/position size.
    KPI panel with trend and brief interpretation.
    Realtime anomaly alerts plus daily/weekly summaries.
     

# Quality, explainability, and operations 

    Data validation, deduplication, currency/time alignment, caching and rate‑limit handling.
    Backtesting and live monitoring (hit rate, MAPE, drawdown and alert precision).
    Explainability via feature importance/SHAP and clear rationale bullets.
    Secure logging and audit trails; modular connectors for easy dataset additions.
     

# Example outcome 

    “EURUSD: Risk +28 (elevated due to policy divergence and bearish sentiment). Forecast −1.2% to +0.8% (low confidence). Decision: Hold with mild short bias; watch upcoming central‑bank events.”
     
