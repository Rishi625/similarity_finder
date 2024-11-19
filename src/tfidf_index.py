
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TFIDFIndex:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
    
    def fit_vectorizer(self, texts):
        self.vectorizer.fit(texts)

    def get_tfidf_vectors(self, texts):
        self.tfidf_vectors = self.vectorizer.transform(texts)


    def get_similarity_scores(self, query):
        query_vector = self.vectorizer.transform([query])
        similarity_scores = cosine_similarity(query_vector, self.tfidf_vectors)
        return similarity_scores

    def get_top_k_indices(self, query, k):
        similarity_scores = self.get_similarity_scores(query)
        top_k_indices = similarity_scores.argsort()[0][-k-1:-1][::-1]
        top_k_scores = similarity_scores[0][top_k_indices]
        return top_k_indices, top_k_scores