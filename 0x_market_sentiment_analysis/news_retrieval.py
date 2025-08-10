import requests
from datetime import datetime
from typing import List

NEWSAPI_KEY = "396e086df39c4af4bbe13ce1c8dae26c"

def get_articles(stock_name: str, start_date: str, end_date: str) -> List[dict]:
    """
    Fetch news articles from NewsAPI for a given stock/company name
    between start_date and end_date (YYYY-MM-DD).
    """
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": stock_name,
        "from": start_date,
        "to": end_date,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": 50,
        "apiKey": NEWSAPI_KEY
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    return data.get("articles", [])

# Example usage:
if __name__ == "__main__":
    articles = get_articles("ICICI", "2025-07-09", "2025-08-09")
    print(f"Fetched {len(articles)} articles.\n")
    for art in articles[:5]:  # Show first 5
        print(f"{art['publishedAt']} - {art['title']}")
