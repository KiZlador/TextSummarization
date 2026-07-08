import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from app.core.config import settings

class SummarizationModel:
    def __init__(self):
        self.device = settings.DEVICE if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        
    def load(self):
        if self.model is not None:
            return
            
        print(f"Загрузка модели...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_NAME)
        
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            settings.MODEL_NAME,
            dtype=dtype
        )
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Модель загружена на {self.device}")
        
    def generate(self, text: str, max_input_tokens: int = None, max_output_tokens: int = None) -> str:
        if max_input_tokens is None:
            max_input_tokens = settings.MAX_INPUT_TOKENS
        if max_output_tokens is None:
            max_output_tokens = settings.MAX_OUTPUT_TOKENS
            
        input_text = f"summarize: {text}"
        
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=max_input_tokens,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_output_tokens,
                num_beams=settings.NUM_BEAMS,
                early_stopping=True,
                repetition_penalty=settings.REPETITION_PENALTY,
                no_repeat_ngram_size=settings.NO_REPEAT_NGRAM_SIZE,
                length_penalty=settings.LENGTH_PENALTY
            )
            
        summary = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return summary

model_instance = SummarizationModel()