import re
import streamlit as st
from PyPDF2 import PdfReader
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


import streamlit as st
from PyPDF2 import PdfReader

uploaded_file = st.file_uploader("📄 Upload a PDF", type="pdf")

if uploaded_file is not None:
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    st.success("✅ PDF successfully loaded.")
    st.text_area("Extracted Text Preview", text[:1500])
else:
    st.warning("Please upload a PDF to continue.")

cleaned_text = re.sub(r'\s+', ' ', text)
cleaned_text = re.sub(r'(\w)- (\w)', r'\1\2', cleaned_text)

print(f"cleaned text: {len(cleaned_text)}")


context = cleaned_text


from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

model_name = "valhalla/longformer-base-4096-finetuned-squadv1"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)


#ex
# questions = [
#     "What is high-quality primary care?",
#     "What are the essential elements of primary care?",
#     "What role does the government play in strengthening it?"
# ]

# for q in questions:
#     result = qa_pipeline({"question": q, "context": context})
#     print(f"\n Question: {q}")
#     print(f" Answer: {result['answer']}")
#     print(f" Score: {round(result['score'], 3)}")
