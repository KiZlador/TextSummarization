import pdfplumber
from docx import Document as DocxDocument
from pathlib import Path

def parse_pdf(file_path: str) -> str:
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

def parse_docx(file_path: str) -> str:
    doc = DocxDocument(file_path)
    text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    return text.strip()

def parse_txt(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def remove_duplicate_paragraphs(text: str) -> str:
    paragraphs = text.split('\n\n')
    seen = set()
    unique_paragraphs = []
    
    for p in paragraphs:
        p_clean = p.strip()
        if p_clean and p_clean not in seen:
            seen.add(p_clean)
            unique_paragraphs.append(p_clean)
            
    return '\n\n'.join(unique_paragraphs)

def parse_document(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    
    if ext == '.pdf':
        text = parse_pdf(file_path)
    elif ext == '.docx':
        text = parse_docx(file_path)
    elif ext == '.txt':
        text = parse_txt(file_path)
    else:
        raise ValueError(f"Неподдерживаемый формат: {ext}")
        
    text = remove_duplicate_paragraphs(text)
    return text.strip()

def is_scan(text: str, min_length: int = 100) -> bool:
    return len(text) < min_length