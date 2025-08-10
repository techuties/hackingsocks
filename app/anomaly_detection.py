import os
import csv
from openai import OpenAI

# Properly quoted multi-line system prompt
SYSTEM_PROMPT = (
    "You are a multidisciplinary analyst tasked with evaluating the key impact factors and their "
    "evaluation methods for a complex problem. The problem requires a holistic approach that integrates "
    "insights from various domains, including finance, economics, technology, social sciences, and "
    "environmental sciences.\n\n"
    "Your objective is to:\n"
    "  - Identify the critical impact factors that need to be considered.\n"
    "  - Analyze each impact factor in-depth, considering its interdependencies and cascading effects.\n"
    "  - Develop a comprehensive framework for evaluating the identified impact factors, including both quantitative "
    "and qualitative assessment methods.\n"
    "  - Provide recommendations on how to effectively incorporate the evaluation of these impact factors into the "
    "decision-making process.\n\n"
    "To accomplish this task, you will need to:\n"
    "  - Conduct a thorough literature review to understand the existing research and best practices in the relevant disciplines.\n"
    "  - Engage with subject matter experts from diverse backgrounds to gather their insights and perspectives on the problem.\n"
    "  - Analyze historical data, industry trends, and future projections to identify the key impact factors.\n"
    "  - Develop causal models and systems thinking approaches to understand the interconnections between the impact factors.\n"
    "  - Evaluate various quantitative and qualitative evaluation methods, such as financial analysis, cost-benefit analysis, "
    "scenario planning, multi-criteria decision analysis, and stakeholder engagement.\n"
    "  - Propose a comprehensive framework for integrating the evaluation of impact factors into the decision-making process, "
    "considering factors such as data availability, uncertainty, and organizational constraints.\n"
    "  - Provide clear and actionable recommendations on how to implement the proposed framework, including the necessary resources, "
    "stakeholder engagement, and change management strategies.\n\n"
    "Your final report should present a well-researched and structured analysis that demonstrates your ability to synthesize insights "
    "from multiple disciplines, identify critical impact factors, and develop a robust evaluation framework. The report should be tailored "
    "to the specific needs of the decision-makers and provide them with the necessary information to make informed decisions."
)

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
                {"role": "system", "content": SYSTEM_PROMPT},
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
openai_key = os.environ.get("OPENAI_API_KEY", "<YOUR_OPENAI_API_KEY>")  # set this variable to your API key
agent = AnomalyDetectionAgent(csv_files, openai_key)
result = agent.analyze()
print(result)