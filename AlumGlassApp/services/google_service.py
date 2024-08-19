import logging
import requests


class GoogleSearchService:
    def __init__(self, config):
        self.api_key = config.GOOGLE_API_KEY
        self.search_engine_id = config.GOOGLE_SEARCH_ENGINE_ID

    def search(self, query):
        try:
            url = f"https://www.googleapis.com/customsearch/v1?q={
                query}&key={self.api_key}&cx={self.search_engine_id}"
            response = requests.get(url)
            response.raise_for_status()
            results = response.json().get('items', [])
            logging.info(f"Google Search Results: {results}")
            return results
        except Exception as e:
            logging.error(f"Error in Google search: {str(e)}")
            return None
