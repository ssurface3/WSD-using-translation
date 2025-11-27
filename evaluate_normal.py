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
        
        # Validation: Check if required columns exist
        required_cols = {'word', 'gold_sense_id', 'predict_sense_id'}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"DataFrame missing required columns: {required_cols}")

        word_scores = []
        total_rows = 0
        weighted_sum = 0.0

        print(f"\n{'WORD':<20} | {'COUNT':<6} | {'ARI':<6}")
        print("-" * 40)

        # 1. Group by word (This replaces the complex dictionary logic)
        for word, group in df.groupby('word'):
            
            # 2. Get the labels for this specific word
            gold = group['gold_sense_id']
            predict = group['predict_sense_id']
            
            # 3. Calculate ARI for this word
            # Handle edge case: if only 1 label exists, ARI is 1.0 (perfect) or 0.0 depending on definition.
            # Usually, if gold has only 1 sense, clustering is trivial.
            if len(gold.unique()) < 2:
                ari = 1.0 # Or 0.0, but usually 1.0 if prediction matches the single group
            else:
                ari = adjusted_rand_score(gold, predict)
            
            count = len(group)
            
            # 4. Add to weighted sum
            weighted_sum += ari * count
            total_rows += count
            
            # Print individual word result
            print(f"{word:<20} | {count:<6} | {ari:.4f}")

        # 5. Calculate Final Weighted Mean
        final_score = weighted_sum / total_rows
        
        print("-" * 40)
        print(f"{'TOTAL (Weighted)':<20} | {total_rows:<6} | {final_score:.4f}")
        print("-" * 40)

        return final_score