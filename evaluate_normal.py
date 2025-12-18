import pandas as pd
import numpy as np
from sklearn.metrics import adjusted_rand_score

class Evaluator:
    def __init__(self):
        pass

    def calculate_weighted_ari(self, df: pd.DataFrame) -> float:
        """
        Calculates the ARI score for each word and then computes the 
        Weighted Average based on how many sentences each word has.
        """
        
        required_cols = {'word', 'gold_sense_id', 'predict_sense_id'}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"DataFrame missing required columns: {required_cols}")

        word_scores = []
        total_rows = 0
        weighted_sum = 0.0

        print(f"\n{'WORD':<20} | {'COUNT':<6} | {'ARI':<6}")

        
        for word, group in df.groupby('word'):
            
            gold = group['gold_sense_id']
            predict = group['predict_sense_id']
            if len(gold.unique()) < 2:
                ari = 1.0 
            else:
                ari = adjusted_rand_score(gold, predict)
            
            count = len(group)
            
            
            weighted_sum += ari * count
            total_rows += count
            
            
            print(f"{word:<20} | {count:<6} | {ari:.4f}")

        
        final_score = weighted_sum / total_rows
        
        
        print(f"{'TOTAL (Weighted)':<20} | {total_rows:<6} | {final_score:.4f}")
        return final_score