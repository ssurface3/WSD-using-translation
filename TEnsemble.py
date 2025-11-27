"""
TODO 
Idea of making ensembles:
Translate to n_foreign languages in batches ( training time from 3 minutes to 3 min * n_foreign)
Extract Embeddings In n_foreign languages (usually the ones that have pretrained BERT)
Normalize Embeddings to make euler distnace to cosine similarity 
NLP clusters are not round , they are blobs:  Agglomerative Clustering 

Use the idea of ensembles , aggregate through the n_foreign + 1 (russian too) models 
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
from translator_clus import TCluster 
class TEnsemble(TCluster):
    def __init__(self):
        super().__init__()
        
        print("   [TEnsemble] Initializing Multi-Language Logic...")
        
        self.model_en = SentenceTransformer('all-MiniLM-L6-v2')
        
      
        self.model_multi = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    def run_pipeline(self, model_name_translator, df, batch_size=16 , run_dual = True): # do not plat with batch size, collab can not handle it!
        print(f"\n🚀 Starting Ensemble Pipeline with {model_name_translator}...")
        if run_dual:
            self.tr.initialize_model_dual(model_name_translator)
        else:
            self.tr.initialize_model(model_name_translator)
        
        
        all_sentences = df['context'].tolist()
        
        #English
        if 'context_en' not in df.columns:
            print("\n... Translating to English (1/3) ...")
            df['context_en'] = self.tr.translate_batch_verbose_dual(all_sentences, batch_size=batch_size, target_lang="eng_Latn")
            
        #German (Deutsch) 
        if 'context_de' not in df.columns:
            print("\n... Translating to German (2/3) ...")
            df['context_de'] = self.tr.translate_batch_verbose_dual(all_sentences, batch_size=batch_size, target_lang="deu_Latn")
            
        #French (Français)
        if 'context_fr' not in df.columns:
            print("\n... Translating to French (3/3) ...")
            df['context_fr'] = self.tr.translate_batch_verbose_dual(all_sentences, batch_size=batch_size, target_lang="fra_Latn")

       
        print("\n... [Phase 2] Stacking & Clustering ...")
        
        unique_words = df['word'].unique()
        results = []
        
        for word in tqdm(unique_words):
            mask = df['word'] == word
            subset = df[mask].copy()
            
          
            v_en = self.model_en.encode(subset['context_en'].tolist(), show_progress_bar=False)
            
         
            v_de = self.model_multi.encode(subset['context_de'].tolist(), show_progress_bar=False)
            
            v_fr = self.model_multi.encode(subset['context_fr'].tolist(), show_progress_bar=False)
            
            # it is very bad, not reccomend using it!
            # v_ru = self.model_multi.encode(subset['context'].tolist(), show_progress_bar=False)
            
           
            # Result shape: (N, 384 * 4) = (N, 1536) dimensions if russian included
            combined_vectors = np.hstack([v_en,v_de , v_fr])
            
            # Basic Clustering, works ok!
            # # We use your robust clustering function (K-Means + Normalization is inside it)
            # # Note: Make sure your self.cl.clustering handles normalization!
            subset['predict_sense_id'] = self.cl.clustering_powertrans(combined_vectors)
            
            results.append(subset)
            
        return pd.concat(results)