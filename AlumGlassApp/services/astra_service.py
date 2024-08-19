from astrapy.db import AstraDB
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class AstraService:
    def __init__(self, config):
        self.db = AstraDB(
            token=config.ASTRA_DB_TOKEN,
            api_endpoint=config.ASTRA_DB_API_ENDPOINT
        )

    def hybrid_search(self, query_text: str, query_embedding: List[float], collection_name: str, top_k: int) -> List[Dict[str, Any]]:
        collection = self.db.collection(collection_name)
        results = collection.find(
            filter={},
            sort={"$vector": query_embedding},
            options={"limit": top_k * 2},
            projection={"content": 1, "metadata": 1, "_id": 0}
        )
        documents = results.get('data', {}).get('documents', [])
        return self.rank_results(documents, query_text)[:top_k]

    def rank_results(self, results: List[Dict[str, Any]], query_text: str) -> List[Dict[str, Any]]:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(
            [query_text] + [doc['content'] for doc in results])
        cosine_similarities = cosine_similarity(
            tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

        ranked_results = sorted(
            zip(results, cosine_similarities), key=lambda x: x[1], reverse=True)
        return [item[0] for item in ranked_results]
