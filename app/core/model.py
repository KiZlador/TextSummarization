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
            
        print(f"Загрузка модели из: {settings.MODEL_NAME}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_NAME)
        
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            settings.MODEL_NAME,
            dtype=dtype,
            use_safetensors=True
        )
        
        self.model.generation_config.max_length = settings.MAX_OUTPUT_TOKENS + 50
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Модель успешно загружена на {self.device}")
        
    def generate(self, text: str, max_input_tokens: int = None, max_output_tokens: int = None) -> str:
        if max_input_tokens is None:
            max_input_tokens = settings.MAX_INPUT_TOKENS
        if max_output_tokens is None:
            max_output_tokens = settings.MAX_OUTPUT_TOKENS
            
        text = text.strip()
        if len(text) < 50:
            return text 
            
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
                max_new_tokens=max(50, max_output_tokens),
                min_length=15,
                num_beams=2, 
                early_stopping=True,
                repetition_penalty=settings.REPETITION_PENALTY,
                no_repeat_ngram_size=2  
            )
            
        summary = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        if summary.lower().startswith("summarize"):
            summary = summary.replace("summarize", "", 1).strip(": ")
            
        if len(summary.split()) < 5:
            sentences = text.split('. ')
            return '. '.join(sentences[:2]) + '.'

        return summary

model_instance = SummarizationModel()