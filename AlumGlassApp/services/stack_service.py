import logging
import requests


class StackExchangeService:
    def search(self, query):
        try:
            url = f"https://api.stackexchange.com/2.2/search?order=desc&sort=activity&intitle={
                query}&site=stackoverflow"
            response = requests.get(url)
            response.raise_for_status()
            results = response.json().get('items', [])
            logging.info(f"Stack Exchange Search Results: {results}")
            return results
        except Exception as e:
            logging.error(f"Error in Stack Exchange search: {str(e)}")
            return None
