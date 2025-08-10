import pandas as pd
import numpy as np
import os

class AnomalyDetectionAgent:
    def __init__(self, price_csv_path, news_data=None, other_data=None, preferred_price_column=None):
        self.price_csv_path = price_csv_path
        self.news_data = news_data
        self.other_data = other_data
        self.preferred_price_column = preferred_price_column
        self.price_df = None
        self.anomaly_score = 0

    def read_csv(self):
        if not os.path.exists(self.price_csv_path):
            raise FileNotFoundError(f"CSV file not found: {self.price_csv_path}")
        self.price_df = pd.read_csv(self.price_csv_path)
        # Try to parse date if present
        if 'Date' in self.price_df.columns:
            self.price_df['Date'] = pd.to_datetime(self.price_df['Date'])
        elif self.price_df.columns[0].lower() in ['date', 'timestamp']:
            self.price_df[self.price_df.columns[0]] = pd.to_datetime(self.price_df[self.price_df.columns[0]])

    def _normalize_name(self, name: str) -> str:
        # Lower-case and strip non-alphanumerics for robust matching
        return "".join(ch for ch in name.lower().strip() if ch.isalnum())

    def _pick_price_column(self) -> str:
        # Choose the most appropriate price column
        assert self.price_df is not None

        columns = list(self.price_df.columns)
        normalized = {self._normalize_name(c): c for c in columns}

        # 1) Use preferred column if provided and present
        if self.preferred_price_column:
            for c in columns:
                if self._normalize_name(c) == self._normalize_name(self.preferred_price_column):
                    return c

        # 2) Try common price columns in order of preference
        candidates = [
            "adjustedclose", "adjclose", "adjclosingprice",
            "close", "closingprice", "lastprice", "price",
        ]
        for key in candidates:
            if key in normalized:
                return normalized[key]

        # 3) Fallback: first numeric column that is not obviously volume
        numeric_cols = []
        for c in columns:
            if self._normalize_name(c) in {"volume", "vol"}:
                continue
            # Check if column can be converted to numeric for most rows
            series = pd.to_numeric(self.price_df[c], errors="coerce")
            if series.notna().sum() >= max(2, int(0.5 * len(series))):
                numeric_cols.append((c, series.notna().sum()))
        if numeric_cols:
            # pick the column with most numeric values
            numeric_cols.sort(key=lambda x: x[1], reverse=True)
            return numeric_cols[0][0]

        # 4) As last resort, use last column
        return columns[-1]

    def detect_price_anomaly(self):
        if self.price_df is None:
            self.read_csv()
        # Use robust selection of price column
        price_col = self._pick_price_column()
        prices = pd.to_numeric(self.price_df[price_col], errors="coerce").dropna()
        if prices.empty or prices.std(ddof=0) == 0:
            # Not enough numeric variation; treat as non-anomalous
            return 0
        # Z-score anomaly detection
        z_scores = (prices - prices.mean()) / prices.std(ddof=0)
        max_z = np.abs(z_scores).max()
        # Map max_z to [-100, 100]
        price_score = min(100, max(0, (max_z - 2) * 25))  # z>2 is unusual, z>6 is extreme
        return price_score

    def detect_news_anomaly(self):
        # Placeholder: if news_data is a list of dicts with 'sentiment' or 'headline'
        if self.news_data is None:
            return 0
        score = 0
        if isinstance(self.news_data, list):
            for item in self.news_data:
                # If sentiment is available, use it
                if 'sentiment' in item:
                    sentiment = item['sentiment']
                    if isinstance(sentiment, (int, float)):
                        score += abs(sentiment) * 10
                # If headline contains "crash", "fraud", "record", etc.
                if 'headline' in item:
                    headline = item['headline'].lower()
                    if any(word in headline for word in ['crash', 'fraud', 'bankruptcy', 'record high', 'record low', 'scandal']):
                        score += 20
        return min(100, score)

    def detect_other_anomaly(self):
        # Placeholder for other data, e.g., volume spikes, financials, etc.
        if self.other_data is None:
            return 0
        score = 0
        # Example: if other_data contains 'volume'
        if isinstance(self.other_data, dict):
            if 'volume' in self.other_data:
                volume = np.array(self.other_data['volume'])
                if len(volume) > 10:
                    z = (volume[-1] - np.mean(volume[:-1])) / (np.std(volume[:-1]) + 1e-6)
                    if abs(z) > 3:
                        score += min(100, abs(z) * 10)
        return min(100, score)

    def compute_anomaly_score(self):
        price_score = self.detect_price_anomaly()
        news_score = self.detect_news_anomaly()
        other_score = self.detect_other_anomaly()
        # Weighted sum, price is most important
        total_score = 0.6 * price_score + 0.3 * news_score + 0.1 * other_score
        # Clamp to [-100, 100]
        self.anomaly_score = max(-100, min(100, int(total_score)))
        return self.anomaly_score

    def explain(self):
        explanation = []
        price_score = self.detect_price_anomaly()
        news_score = self.detect_news_anomaly()
        other_score = self.detect_other_anomaly()
        if price_score > 50:
            explanation.append(f"Unusual price movement detected (score: {price_score:.1f}).")
        if news_score > 20:
            explanation.append(f"Significant news sentiment or headlines detected (score: {news_score:.1f}).")
        if other_score > 10:
            explanation.append(f"Other anomaly detected (score: {other_score:.1f}).")
        if not explanation:
            explanation.append("No significant anomaly detected.")
        return " ".join(explanation)

# Example usage:
agent = AnomalyDetectionAgent("api/cache/yahoo/AAPL_2024-01-01_2024-12-31_1d.csv", news_data=[{"headline": "AAPL hits record high", "sentiment": 0.8}])
score = agent.compute_anomaly_score()
print(f"Anomaly Score: {score}")
print(agent.explain())

