import streamlit as st
import tempfile
from pathlib import Path

from app.services.document_parser import parse_document, is_scan
from app.services.chunking import summarize_long_text
from app.core.model import model_instance

st.set_page_config(
    page_title="Text Summarizer",
    page_icon="",
    layout="wide"
)

MAX_CHARS = 100000

st.title("📄 Text Summarizer")
st.markdown("Загрузите документ (PDF, DOCX или TXT) и получите краткое содержание")

with st.sidebar:
    st.header("О модели")
    st.markdown("**Модель:** `rut5-base-absum`")
    st.markdown("**Параметры:** ~250М")
    st.markdown("**Архитектура:** T5-base (Seq2Seq)")
    st.markdown("**Обучена на:** новостном датасете Gazeta")
    
    st.markdown("---")
    st.markdown("**Ограничения:**")
    st.info(f"Макс. размер документа: {MAX_CHARS:,} символов")
    
    st.markdown("---")
    st.markdown("**Поддерживаемые форматы:**")
    st.markdown("- PDF (текстовый)")
    st.markdown("- DOCX (Word)")
    st.markdown("- TXT (обычный текст)")

uploaded_file = st.file_uploader(
    "Выберите файл для суммаризации",
    type=["pdf", "docx", "txt"],
    help="Максимальный размер файла: 50MB"
)

if uploaded_file is not None:
    st.markdown("### Информация о файле")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Имя файла", uploaded_file.name)
    with col2:
        st.metric("Размер", f"{uploaded_file.size / 1024:.1f} KB")
    with col3:
        file_ext = Path(uploaded_file.name).suffix.upper()
        st.metric("Тип", file_ext)
    
    st.markdown("---")
    
    if st.button("Получить краткое содержание", type="primary", use_container_width=True):
        with st.spinner("Загрузка модели..."):
            if model_instance.model is None:
                model_instance.load()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            with st.spinner("Извлечение текста из документа..."):
                text = parse_document(tmp_file_path)
            
            file_ext = Path(uploaded_file.name).suffix.lower()
            if file_ext == '.pdf' and is_scan(text):
                st.error("Похоже, это скан документа.\n\nПожалуйста, загрузите текстовый PDF, Word или TXT документ.")
                st.stop()
            
            if len(text) > MAX_CHARS:
                st.warning(f"Документ слишком большой ({len(text):,} символов).")
                st.info(f"Будет обработана только первая часть ({MAX_CHARS:,} символов).")
                text = text[:MAX_CHARS]
            
            st.markdown("### 📝 Саммари")
            with st.spinner("Создание краткого содержания... "):
                summary = summarize_long_text(text)
            
            st.success("✅ Готово!")
            st.markdown(summary)
            
            st.download_button(
                label="Скачать саммари",
                data=summary,
                file_name=f"summary_{Path(uploaded_file.name).stem}.txt",
                mime="text/plain"
            )
        
        except Exception as e:
            st.error(f"Произошла ошибка: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
        
        finally:
            import os
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)

else:
    st.info("Загрузите файл выше, чтобы начать")
    
    st.markdown("""
    1. **Загрузите документ** (PDF, DOCX или TXT)
    2. **Нажмите кнопку 'Суммаризировать'**
    3. **Получите краткое саммари**
    """)
    
    st.markdown("### Рекомендации")
    st.markdown("""
    **Хорошо подходят:**
    - Новостные статьи (1-5 страниц)
    - Научные статьи (5-15 страниц)
    - Отчеты и документы (10-30 страниц)
    
    **Не подходят:**
    - Книги и романы (слишком большие)
    - Сканы документов (нет текста)
    """)
