"""
recommender.py
Hybrid recommender combining content-based TF-IDF scoring and item-based collaborative filtering
(using cosine similarity on item rating vectors).

API:
    Recommender(products_csv_path, ratings_csv_path)
    .recommend(product_id, top_n=5, mode="hybrid", content_w=0.5)

mode:
    "content" -> content-based only
    "collab"   -> collaborative only
    "hybrid"   -> weighted sum of both (content_w controls weight for content)
"""


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity


class Recommender:
    def __init__(self, products_csv_path, ratings_csv_path=None):
        # load products
        self.df = pd.read_csv(products_csv_path, dtype={"product_id": str})
        self.df["description"] = self.df["description"].fillna("")
        # content model
        self.tfidf = TfidfVectorizer(stop_words="english", max_features=3000)
        # combine title + description for better text signal
        text = (self.df["title"].fillna("") + " " + self.df["description"].fillna("")).values
        self.tfidf_matrix = self.tfidf.fit_transform(text)
        self.content_sim = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)

        # collaborative model (item-item cosine on rating vectors)
        self.ratings_df = None
        self.item_sim = None
        if ratings_csv_path and os.path.exists(ratings_csv_path):
            self._build_collab(ratings_csv_path)
        else:
            # ratings not provided or path invalid; leave item_sim as None
            self.ratings_df = pd.DataFrame(columns=["user_id", "product_id", "rating"])
            self.item_sim = None

        # id-index maps
        self.id_to_idx = {pid: idx for idx, pid in enumerate(self.df["product_id"].astype(str))}
        self.idx_to_id = {idx: pid for pid, idx in self.id_to_idx.items()}

    def _build_collab(self, ratings_csv_path):
        import os
        self.ratings_df = pd.read_csv(ratings_csv_path, dtype={"product_id": str, "user_id": str})
        # pivot to item-user matrix: rows = product_id, columns = user_id
        pivot = self.ratings_df.pivot_table(index="product_id", columns="user_id", values="rating", aggfunc="mean").fillna(0)
        # ensure pivot rows are in same order as products dataframe
        # if some products have no ratings, add them as zero vectors
        all_pids = self.df["product_id"].astype(str).tolist()
        pivot = pivot.reindex(all_pids, fill_value=0)
        # compute cosine similarity between item rating vectors
        mat = pivot.values
        # if there are no ratings at all, set item_sim to None
        if mat.shape[1] == 0:
            self.item_sim = None
            return
        self.item_sim = cosine_similarity(mat, mat)

    def list_products(self):
        return self.df.to_dict(orient="records")

    def recommend(self, product_id, top_n=5, mode="hybrid", content_w=0.5):
        """
        product_id: string or int-like
        mode: 'content', 'collab', or 'hybrid'
        content_w: weight for content (0..1) when mode='hybrid' (collab weight = 1-content_w)
        """
        product_id = str(product_id)
        if product_id not in self.id_to_idx:
            return []

        idx = self.id_to_idx[product_id]
        n = top_n

        # content scores
        content_scores = self.content_sim[idx] if self.content_sim is not None else np.zeros(len(self.df))

        # collab scores
        collab_scores = None
        if self.item_sim is not None:
            collab_scores = self.item_sim[idx]
        else:
            collab_scores = np.zeros(len(self.df))

        # choose mode
        if mode == "content":
            scores = content_scores
        elif mode == "collab":
            scores = collab_scores
        else:  # hybrid
            # normalize both score vectors to 0..1
            def norm(x):
                x = np.array(x, dtype=float)
                if x.max() - x.min() == 0:
                    return np.zeros_like(x)
                return (x - x.min()) / (x.max() - x.min())
            c = norm(content_scores)
            b = norm(collab_scores)
            scores = content_w * c + (1.0 - content_w) * b

        # get top indices excluding the query item
        indices_scores = list(enumerate(scores))
        indices_scores = [t for t in indices_scores if t[0] != idx]
        indices_scores = sorted(indices_scores, key=lambda x: x[1], reverse=True)
        top = [i for i, s in indices_scores[:n]]
        results = self.df.iloc[top].to_dict(orient="records")
        return results


# Make os available at module level for conditional checks
import os
