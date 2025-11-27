"""
TODO 
Idea of adding ruBERT:
Translate to Englsih in batches
Extract Embeddings In english BERT
Normalize Embeddings to make euler distnace to cosine similarity 
NLP clusters are not round , they are blobs:  Agglomerative Clustering 

PLUS: 
Extract Embeddings In english ruBERT 
and make the average of the prediction ( threshold guessing)
"""


import time
from  translation import Translator
import pandas as pd 
from emdedding_cluster import emd_clust
from translator_clus import TCluster
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.preprocessing import normalize

class TCluster_plus(TCluster):
    def __init__(self):
        super().__init__()
        print("   [TCluster+] Loading Russian Model...")
        self.model_ru = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    def run_pipeline(self, model_name, df, batch_size: int = 32):
       
        model_WSD = 'Cluster and translation + ruBERT'
        print(f"\n Initializing pipeline with Translation: {model_name} | WSD: {model_WSD}...")
        
        
        self.tr.initialize_model(model_name)
        print("\n... [Phase 1] Translating entire dataset ...")
        
        if 'context_en' not in df.columns:
            all_sentences = df['context'].tolist()
            df['context_en'] = self.tr.translate_batch_verbose(all_sentences, batch_size=batch_size) # we translate all the sentences
        else:
            print("   -> Translations found in columns. Skipping translation.")

        print("\n... [Phase 2] Clustering per word ...")
        
        unique_words = df['word'].unique() 
        results = []
        
        for word in unique_words:
            
            mask = df['word'] == word
            subset_df = df[mask].copy()
            
            print(f"   -> Processing '{word}': {len(subset_df)} rows")
            
            
            vec_en = self.cl.extract_embeddings_simple(subset_df['context_en'].tolist())
            vec_ru = self.model_ru.encode(subset_df['context'].tolist(), show_progress_bar=False)

           
            
            combined_vectors = np.hstack([vec_en, vec_ru])
            
            
            
            subset_df['predict_sense_id'] = self.cl.agglomerative_clustering(combined_vectors)
            
            results.append(subset_df)
        return pd.concat(results)
    