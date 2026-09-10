"""
Retrieval-Augmented Generation (RAG) module for grounding agent replies
in historical Apple Support resolutions.
Uses BM25 / TF-IDF retrieval over verified historical customer-agent pairs.
"""

import os
import re
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SupportKnowledgeBase:
    def __init__(self, data_path: str = 'data/processed/applesupport_pairs_sample.csv'):
        self.data_path = data_path
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.df: Optional[pd.DataFrame] = None
        self._load_and_index()

    def _load_and_index(self):
        if not os.path.exists(self.data_path):
            alt_path = 'data/processed/applesupport_pairs.parquet'
            if os.path.exists(alt_path):
                self.df = pd.read_parquet(alt_path).head(15000).reset_index(drop=True)
            else:
                raise FileNotFoundError(f"Knowledge base data not found at {self.data_path} or {alt_path}")
        else:
            self.df = pd.read_csv(self.data_path)

        # Ensure strings
        self.df['customer_text'] = self.df['customer_text'].astype(str)
        self.df['agent_text'] = self.df['agent_text'].astype(str)

        # Build TF-IDF index over customer inquiries + agent replies
        # Sublinear TF scaling, n-grams (1, 2)
        corpus = (self.df['customer_text'] + " " + self.df['agent_text']).tolist()
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=25000,
            sublinear_tf=True,
            stop_words='english'
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        print(f"[KnowledgeBase] Successfully indexed {len(self.df):,} historical support interactions.")

    def retrieve_similar_resolutions(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve top_k most similar historical customer queries and Apple's resolutions.
        """
        if self.vectorizer is None or self.tfidf_matrix is None or self.df is None:
            return []

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            row = self.df.iloc[idx]
            results.append({
                "score": round(score, 4),
                "historical_query": row['customer_text'],
                "historical_resolution": row['agent_text'],
                "customer_tweet_id": row.get('customer_tweet_id', ''),
                "agent_tweet_id": row.get('agent_tweet_id', '')
            })
        return results

    def format_grounding_context(self, retrieved: List[Dict[str, Any]]) -> str:
        """
        Formats retrieved historical resolutions into an LLM context block.
        """
        if not retrieved:
            return "No historical resolutions retrieved."
        
        lines = ["HISTORICAL RESOLUTION EVIDENCE (Grounding Knowledge):"]
        for i, item in enumerate(retrieved, start=1):
            lines.append(f"Example #{i} (Relevance Score: {item['score']}):")
            lines.append(f"  Customer Query: \"{item['historical_query']}\"")
            lines.append(f"  Historical Brand Resolution: \"{item['historical_resolution']}\"")
        return "\n".join(lines)

if __name__ == '__main__':
    kb = SupportKnowledgeBase()
    test_queries = [
        "My iPhone 7 battery is draining in 2 hours and shutting off at 20%",
        "Why is my keyboard typing an A and square symbol when I type I?",
        "Someone charged my card $14.99 on iTunes and I didn't authorize it",
        "Dropped my phone in water and screen is flickering green"
    ]
    for q in test_queries:
        print(f"\nQUERY: {q}")
        matches = kb.retrieve_similar_resolutions(q, top_k=2)
        print(kb.format_grounding_context(matches))
