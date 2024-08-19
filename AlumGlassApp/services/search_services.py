from abc import ABC, abstractmethod
from typing import List, Dict, Any
import requests
from googleapiclient.discovery import build
from bs4 import BeautifulSoup
import wikipediaapi
import time
import logging
import json


class SearchService(ABC):
    @abstractmethod
    def search(self, query: str) -> Any:
        pass


class GoogleSearchService(SearchService):
    def __init__(self, config):
        self.api_key = config.GOOGLE_API_KEY
        self.cse_id = config.GOOGLE_CSE_ID

    def search(self, query: str, num_results: int = 5) -> List[Dict[str, str]]:
        try:
            service = build("customsearch", "v1", developerKey=self.api_key)
            res = service.cse().list(q=query, cx=self.cse_id, num=num_results).execute()
            return [{'title': item['title'], 'snippet': item['snippet'], 'link': item['link']} for item in res['items']]
        except Exception as e:
            logging.error(f"Error in Google search: {e}")
            return []


class DuckDuckGoService(SearchService):
    def search(self, query: str) -> str:
        url = f"https://html.duckduckgo.com/html/?q={query}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            zero_click = soup.find('div', {'id': 'zero_click_abstract'})
            if (zero_click):
                return zero_click.text.strip()
            first_result = soup.find('div', {'class': 'result__body'})
            if first_result:
                return first_result.text.strip()
            return "No immediate answer found."
        except Exception as e:
            logging.error(f"Error in DuckDuckGo search: {e}")
            return "Error fetching DuckDuckGo results."


class WikipediaService(SearchService):
    def search(self, query: str, sentences: int = 3) -> str:
        user_agent = "AlumGlass/1.0 (kamankhane@gmail.com)"
        wiki_en = wikipediaapi.Wikipedia(user_agent, 'en')
        wiki_fa = wikipediaapi.Wikipedia(user_agent, 'fa')

        def search_in_language(wiki, query):
            try:
                page = wiki.page(query)
                if page.exists():
                    return ' '.join(page.summary.split('.')[:sentences]) + '.'
                else:
                    return None
            except Exception as e:
                logging.error(f"Error in Wikipedia search: {e}")
                return None

        result_en = search_in_language(wiki_en, query)
        result_fa = search_in_language(wiki_fa, query)

        if result_en and result_fa:
            return f"English: {result_en}\n\nPersian: {result_fa}"
        elif result_en:
            return f"English: {result_en}\n\nNo Persian results found."
        elif result_fa:
            return f"No English results found.\n\nPersian: {result_fa}"
        else:
            return "No Wikipedia page found for this query in English or Persian."


class StackExchangeService(SearchService):
    def search(self, query: str, site: str = 'engineering') -> List[Dict[str, Any]]:
        url = f"https://api.stackexchange.com/2.3/search/advanced"
        params = {
            'order': 'desc',
            'sort': 'relevance',
            'q': query,
            'site': site,
            'pagesize': 5
        }
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if 'items' in data:
                return [{
                    'title': item['title'],
                    'link': item['link'],
                    'score': item['score']
                } for item in data['items']]
            return []
        except Exception as e:
            logging.error(f"Error in Stack Exchange search: {e}")
            return []


class SearxNGService(SearchService):
    def __init__(self, config):
        self.base_url = config.SEARXNG_BASE_URL
        self.search_endpoint = "/search"

    def search(self, query: str, num_results: int = 15) -> str:
        params = {
            "q": query,
            "format": "json"
        }
        url = f"{self.base_url}{self.search_endpoint}"

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            results = response.json()

            if results.get('results'):
                combined_content = " ".join(
                    [result['content']
                        for result in results['results'][:num_results]]
                )
                return combined_content
            else:
                return "No results found."

        except requests.RequestException as e:
            logging.error(f"Error in SearxNG search: {e}")
            return "An error occurred."
        except json.JSONDecodeError:
            logging.error("Error: Received invalid JSON response")
            return "Error: Received invalid JSON response."
