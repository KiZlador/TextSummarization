import re
from app.core.model import model_instance
from app.core.config import settings

def split_into_paragraphs(text: str) -> list[str]: 
    paragraphs = re.split(r'\n\s*\n', text)
    if len(paragraphs) <= 1:
        paragraphs = text.split('\n')
    return [p.strip() for p in paragraphs if p.strip()]

def chunk_text_hard(text: str, max_tokens: int) -> list[str]:
    tokens = model_instance.tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return [text]
    
    chunks = []
    stride = max_tokens - 50
    for i in range(0, len(tokens), stride):
        chunk_tokens = tokens[i:i + max_tokens]
        chunks.append(model_instance.tokenizer.decode(chunk_tokens, skip_special_tokens=True))
        if i + max_tokens >= len(tokens):
            break
    return chunks

def chunk_text(text: str, max_tokens: int = None) -> list[str]:
    if max_tokens is None:
        max_tokens = settings.MAX_CHUNK_TOKENS

    paragraphs = split_into_paragraphs(text)
    chunks = []
    current_chunk = ""
    current_chunk_tokens = 0

    for para in paragraphs:
        para_tokens = len(model_instance.tokenizer.encode(para, add_special_tokens=False))

        if para_tokens > max_tokens:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
                current_chunk_tokens = 0
            hard_chunks = chunk_text_hard(para, max_tokens)
            chunks.extend(hard_chunks)
            continue

        if current_chunk_tokens + para_tokens > max_tokens and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = para
            current_chunk_tokens = para_tokens
        else:
            current_chunk = f"{current_chunk}\n\n{para}" if current_chunk else para
            current_chunk_tokens += para_tokens

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks if chunks else [text]

def summarize_long_text(text: str) -> str:
    chunks = chunk_text(text)
    
    if len(chunks) == 1:
        return model_instance.generate(text)
    
    chunk_summaries = []
    for chunk in chunks:
        summary = model_instance.generate(chunk)
        chunk_summaries.append(summary)
    
    combined = " ".join(chunk_summaries)
    
    combined_tokens = len(model_instance.tokenizer.encode(combined, add_special_tokens=False))
    if combined_tokens > settings.MAX_CHUNK_TOKENS:
        final_summary = model_instance.generate(
            combined,
            max_input_tokens=settings.MAX_CHUNK_TOKENS,
            max_output_tokens=350
        )
        return final_summary
    else:
        return combined