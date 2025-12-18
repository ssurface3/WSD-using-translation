import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm 


# model_name = "facebook/nllb-200-distilled-600M" # or 1.3B or 3.3B

class Translator():
    def __init__(self):
        pass
    def initialize_model(self,model_name ):
        print("Loading model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.model.to(self.device)
    def initialize_model_dual(self, model_name):
        print(f"   [Translator] Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
       
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16
        )
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if torch.cuda.device_count() > 1:
            print(f"    ACTIVATING DUAL GPU: Found {torch.cuda.device_count()} GPUs!")
            self.model = torch.nn.DataParallel(self.model)
        
        self.model.to(self.device)

    # def translate_nllb_batch(self,sentences, batch_size:int =32, target_lang: str="eng_Latn") -> list:
    #     """
    #     Inputs:
    #     sentences: List of strings (Russian)
    #     batch_size: How many to process at once (32 is standard for laptops)
    #     target_lang: 'eng_Latn' is the NLLB code for English

    #     Outputs: 
    #     List of translated adn tokenized sentences in a batch
    #     """
        
    #     tokenizer = self.tokenizer
    #     device = self.device
    #     model = self.model

    #     target_lang_id  = tokenizer.convert_tokens_to_ids(target_lang)

        
    #     all_translations = []


    #     for i in tqdm(range(0, len(sentences), batch_size), desc="Translating"):
            

    #         batch_texts = sentences[i : i + batch_size]
            
    #         inputs = tokenizer(
    #             batch_texts, 
    #             return_tensors="pt", 
    #             padding=True, 
    #             truncation=True, 
    #             max_length=128
    #         ).to(device)
            

    #         with torch.no_grad(): # Disable gradient calculation to save memory
    #             generated_tokens = model.generate(
    #                 **inputs,
    #                 forced_bos_token_id=target_lang_id, 
    #                 max_length=128
    #             )
                
        
    #         decoded_batch = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            
        
    #         all_translations.extend(decoded_batch)

    #     return all_translations
    def translate_batch_verbose(self, sentences, batch_size=32, target_lang="eng_Latn")-> list:
        """""
        Inputs:
        sentences: List of strings (Russian)
        batch_size: How many to process at once 
        target_lang: 'eng_Latn' is the NLLB code for English

        Outputs: 
        List of translated adn tokenized sentences in a batch
        """
        
        if isinstance(self.model, torch.nn.DataParallel):
            generation_model = self.model.module
        else:
            generation_model = self.model
        # --------------------------------
        tokenizer = self.tokenizer
        model = self.model
        device = self.device
        target_lang_id = tokenizer.convert_tokens_to_ids([target_lang])
        results = []
        tokenizer.src_lang = "rus_Cyrl"  
        with tqdm(total=len(sentences)) as pbar:
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i : i + batch_size]
                
                inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
                
                with torch.no_grad():
                    generated = model.generate(**inputs, forced_bos_token_id=target_lang_id, max_length=128)
                
                decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
                results.extend(decoded) # translations
                
        return results
    def translate_batch_verbose_dual(self, sentences, batch_size=32, target_lang="eng_Latn") -> list:
        """
        Inputs:
        sentences: List of strings (Russian)
        batch_size: How many to process at once 
        target_lang: 'eng_Latn' is the NLLB code for English

        Outputs: 
        List of translated and tokenized sentences in a batch
        """
        
       
        if isinstance(self.model, torch.nn.DataParallel):
            generation_model = self.model.module
        else:
            generation_model = self.model
       

        tokenizer = self.tokenizer
        device = self.device
        
       
        target_lang_id = tokenizer.convert_tokens_to_ids(target_lang)
        
        results = []
        tokenizer.src_lang = "rus_Cyrl" 
        
        with tqdm(total=len(sentences)) as pbar:
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i : i + batch_size]
                
                inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
                
                with torch.no_grad():
       
                    generated = generation_model.generate(
                        **inputs, 
                        forced_bos_token_id=target_lang_id, 
                        max_length=128
                    )
                
                decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
                results.extend(decoded) 
                
                pbar.update(len(batch))
                
        return results