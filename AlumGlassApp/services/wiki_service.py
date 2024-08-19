import logging
import requests


class WikipediaService:
    def search(self, query):
        try:
            url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={
                query}&format=json"
            response = requests.get(url)
            response.raise_for_status()
            results = response.json().get('query', {}).get('search', [])
            logging.info(f"Wikipedia Search Results: {results}")
            return results
        except Exception as e:
            logging.error(f"Error in Wikipedia search: {str(e)}")
            return None
