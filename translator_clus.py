"""
TODO 
BASELINE:
Translate to Englsih in batches
Extract Embeddings In english BERT
Normalize Embeddings to make euler distnace to cosine similarity 
NLP clusters are not round , they are blobs:  Agglomerative Clustering 
"""

import time
from  translation import Translator
import pandas as pd 
from emdedding_cluster import emd_clust
class TCluster():
    def __init__(self):
        self.tr = Translator()
        self.cl = emd_clust()
       
    def run_pipeline(self, model_name, df, batch_size: int = 32):
       
        model_WSD = 'Cluster and translation'
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
            
            vectors = self.cl.extract_embeddings_simple(subset_df['context_en'].tolist())
            predictions = self.cl.clustering(vectors)
            subset_df['predict_sense_id'] = predictions
            results.append(subset_df)
            
        final_df = pd.concat(results)
        print("\n Done! Results saved.")
        final_df[['predict_sense_id' , 'gold_sense_id' , 'word']].to_csv()
        return final_df
    