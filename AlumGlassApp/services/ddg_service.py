import logging
import requests


class DuckDuckGoService:
    def search(self, query):
        try:
            url = f"https://api.duckduckgo.com/?q={query}&format=json"
            response = requests.get(url)
            response.raise_for_status()
            results = response.json().get('RelatedTopics', [])
            logging.info(f"DuckDuckGo Search Results: {results}")
            return results
        except Exception as e:
            logging.error(f"Error in DuckDuckGo search: {str(e)}")
            return None
