import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm 
import sys
import time
from helper import helper
import pandas as pd
def main(input_file):
    try:
        h = helper()
        h.print_header()
        print(f"\n... Loading Data from {input_file} ...")
        try:
            df = pd.read_csv(input_file, sep='\t', 
                             names=['context_id', 'word', 'gold_sense_id', 'dummy', 'positions', 'context'],
                             quoting=3, on_bad_lines='skip')
            if 'dummy' in df.columns: df.drop(columns=['dummy'], inplace=True)
        except Exception as e:
            print(f"Error loading file: {e}")
            sys.exit(1)
        df = df[df['word'] != 'word']
        
        df.reset_index(drop=True, inplace=True)
 

        print(f"Loaded {len(df)} rows.")
        model_name_translator = h.get_user_model_choice_translator()
        model_WSD = h.get_user_model_choice_WSD() 

        confirm = input(f"Start processing? (y/n): ").lower()
        if confirm == 'y':
            df = model_WSD.run_pipeline(model_name_translator, df)
            df.to_csv("checkpoint_translated.tsv", sep='\t', index=False) 
        else:
            print("Aborted.")
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n Pipeline stopped by user.")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = input("Please enter the path to the CSV file: ")
        
    main(file_path)