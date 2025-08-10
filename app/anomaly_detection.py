import os
import csv
from openai import OpenAI

class AnomalyDetectionAgent:
    def __init__(self, csv_paths, openai_api_key, model="gpt-4o"):
        self.csv_paths = csv_paths
        self.openai_api_key = openai_api_key
        self.model = model
        self.client = OpenAI(api_key=self.openai_api_key)

    def read_csv_files(self):
        data = {}
        for path in self.csv_paths:
            rows = []
            with open(path, newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    rows.append(row)
            data[os.path.basename(path)] = rows
        return data

    def prepare_prompt(self, data):
        prompt = (
            "You are an expert data analyst. "
            "Given the following CSV data from multiple files, analyze and identify any anomalies or outliers. "
            "Explain your reasoning and highlight which rows or columns are anomalous, if any.\n\n"
        )
        for filename, rows in data.items():
            prompt += f"File: {filename}\n"
            if not rows:
                prompt += "(No data)\n"
                continue
            # Show up to 5 rows for context
            headers = list(rows[0].keys())
            prompt += ",".join(headers) + "\n"
            for row in rows[:5]:
                prompt += ",".join(str(row[h]) for h in headers) + "\n"
            if len(rows) > 5:
                prompt += f"... ({len(rows)} rows total)\n"
            prompt += "\n"
        prompt += "Please provide a detailed anomaly prediction and reasoning."
        return prompt

    def analyze(self):
        data = self.read_csv_files()
        prompt = self.prepare_prompt(data)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant for anomaly detection."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=800,
        )
        return response.choices[0].message.content

# Example usage (set your API key variable before running):
csv_files = [
    "app/data/yahoo/MSFT_financials.csv",
    "app/data/yahoo/MSFT_news.csv",
    "app/data/yahoo/MSFT_options.csv",
    "app/data/yahoo/MSFT_other_financials.csv",
    "app/data/yahoo/MSFT_other.csv",
    "app/data/yahoo/MSFT_overall.csv",
]
openai_key = "<YOUR_OPENAI_API_KEY>"  # set this variable to your API key
agent = AnomalyDetectionAgent(csv_files, openai_key)
result = agent.analyze()
print(result)