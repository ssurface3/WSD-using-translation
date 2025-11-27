
from TEnsemble import TEnsemble 
from Tcluster_plus import TCluster_plus
from translator_clus import TCluster
class helper():
    def __init__(self):
        self.MODEL_translator_OPTIONS = {
            "1": ("facebook/nllb-200-distilled-600M", "Lightweight (Fastest)"),
            "2": ("facebook/nllb-200-distilled-1.3B", "Balanced (Standard)"),
            "3": ("facebook/nllb-200-3.3B", "Heavy (Best Quality, requires strong GPU)")
        }
        self.class_registry = {
                    "Tcluster": TCluster,
                    "Tcluster_plus": TCluster_plus,
                    "TEnsemble": TEnsemble
                }

        self.MODEL_WSD_OPTIONS = {
            "1": ("Tcluster", "Translation to english + clustering"),
            "2": ("Tcluster_plus", "Translation to englsih + ruBERT"),
            "3": ("TEnsemble", "Translation to n laguages + ruBERT")
        }
    def print_header(self):
        print("\033[95m" + "="*50)
        print("   WSD PIPELINE: TRANSLATION & CLUSTERING")
        print("="*50 + "\033[0m")

    def get_user_model_choice_translator(self):
        """
        Handles the interactive menu logic
        """
        print("\n\033[94mPlease select the NLLB Translation Model:\033[0m")
        
        for key, (name, desc) in self.MODEL_translator_OPTIONS.items():
            print(f"  [{key}] {name:<35} | {desc}")

        while True:
            choice = input("\n\033[92mEnter choice (1-3): \033[0m").strip()
            
            if choice in self.MODEL_translator_OPTIONS:
                selected_model, _ = self.MODEL_translator_OPTIONS[choice]
                print(f"\nYou selected: \033[1m{selected_model}\033[0m")
                return selected_model
            else:
                print("Invalid selection. Please type 1, 2, or 3.")
    def get_user_model_choice_WSD(self):
        print("\n\033[94mPlease select the WSD Logic:\033[0m")
        
        for key, (name, desc) in self.MODEL_WSD_OPTIONS.items():
            print(f"  [{key}] {name:<35} | {desc}")

        while True:
            choice = input("\n\033[92mEnter choice (1-3): \033[0m").strip()
            
            if choice in self.MODEL_WSD_OPTIONS:
                model_name, _ = self.MODEL_WSD_OPTIONS[choice]
                
                print(f"\nYou selected: \033[1m{model_name}\033[0m")
                

                model_class = self.class_registry[model_name]
                
                return model_class() 
            else:
                print("Invalid selection.")
