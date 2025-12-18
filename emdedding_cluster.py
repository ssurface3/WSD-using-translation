from sentence_transformers import SentenceTransformer
import torch
from typing import Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import PowerTransformer, normalize

class emd_clust():
    def __init__(self , embedding_model:str ='all-MiniLM-L6-v2') -> None:
        self.embedding_model = SentenceTransformer(embedding_model) # SOTA 
        pass
    def extract_embeddings_simple(self,sentences):
        """
        sentences: List of translated English strings
        Returns: Numpy array of embeddings (Vectors)
        """
        embeddings = self.embedding_model.encode(
            sentences, 
            batch_size=32, 
            device='cuda' if torch.cuda.is_available() else 'cpu',
            show_progress_bar=True
        )
        
        return embeddings
    def clustering(self, emb): # worked the best although the easiest
        n_clusters = self.amount_of_clusters(emb)
        kmeans = KMeans(n_clusters=n_clusters)
        return  kmeans.fit_predict(emb)
    def clustering_powertrans(self, emb):  # power - trnaformation: thought it would better normalize but not 
        pt = PowerTransformer(method='yeo-johnson', standardize=True) 
        
        try:
            vectors = pt.fit_transform(emb)
        except Exception:
            vectors = emb 
        n_clusters = self.amount_of_clusters(vectors)
        kmeans = KMeans(n_clusters=n_clusters)
        return  kmeans.fit_predict(emb)
    def amount_of_clusters(self, emd): 

        best_score = -1
        best_k = 2

        
        for k_candidate in range(2, 4):
            kmeans = KMeans(n_clusters=k_candidate, random_state=42, n_init=10)
            labels = kmeans.fit_predict(emd)
            score = silhouette_score(emd, labels)
            
            if score > best_score:
                best_score = score
                best_k = k_candidate
        return best_k
    def agglomerative_clustering(self, emb):  
        """
        Resulted in the worse solution than usual clustering
        The problem might lie in the fact thta if i normailize the data it will be no longer 
        applicable to use the Agglomerative Lustering
        """
        pt = PowerTransformer(method='yeo-johnson', standardize=True) 
        
        try:
            vectors = pt.fit_transform(emb)
        except Exception:
            vectors = emb 
            
        vectors = normalize(vectors)

        n_clusters = self.amount_of_clusters(vectors)

        
        model = AgglomerativeClustering(
            n_clusters=n_clusters, 
            metric='cosine', 
            linkage='average'
        )
        
        labels = model.fit_predict(vectors)
        return labels
    def agglomerative_clustering_without_norm(self, emb): 
        """
        In this method  I tried to escape the problem of non-normality of the vectors
        Using only agglomerative clustering
        """
        n_clusters = self.amount_of_clusters(emb)

        
        model = AgglomerativeClustering(
            n_clusters=n_clusters, 
            metric='cosine', 
            linkage='average'
        )
        
        labels = model.fit_predict(emb)
        return labels
    def ARI(self, df,embeddings ,k = 2) -> None: # used to score model at earlier stages
        """
        No automatic clustering right now, made just for checking
        """
        print("Clustering...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        predicted_labels = kmeans.fit_predict(embeddings)

        score = adjusted_rand_score(df['gold_sense_id'], predicted_labels)

        print("-" * 30)
        print(f"Adjusted Rand Index (ARI): {score:.4f}")
        print("-" * 30)

        

