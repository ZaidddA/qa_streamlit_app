import streamlit as st
import re
from PyPDF2 import PdfReader
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
from docx import Document

st.title("📄 Document Question Answering App")

uploaded_file = st.file_uploader("📤 Upload a document", type=["pdf", "docx", "txt"])

text = ""

if uploaded_file is not None:
    file_type = uploaded_file.name.split(".")[-1]

    if file_type == "pdf":
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text

    elif file_type == "docx":
        doc = Document(uploaded_file)
        for para in doc.paragraphs:
            text += para.text + "\n"

    elif file_type == "txt":
        text = uploaded_file.read().decode("utf-8")


    raw_text = text
    cleaned_text = re.sub(r"\s+", " ", raw_text)
    cleaned_text = re.sub(r"(\w)- (\w)", r"\1\2", cleaned_text)
    context = cleaned_text

    st.success("✅ Document successfully loaded.")
    st.text_area("📑 Extracted Text Preview", text[:1500])

    model_name = "valhalla/longformer-base-4096-finetuned-squadv1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

    user_question = st.text_input("💬 Ask a question about the document:")
    if st.button("Get Answer") and user_question.strip():
        result = qa_pipeline({"question": user_question, "context": context})
        st.write(f"🔹 **Question**: {user_question}")
        st.write(f"🧠 **Answer**: {result['answer']}")
        st.write(f"📊 **Confidence Score**: {round(result['score'], 3)}")

else:
    st.info("Please upload a PDF, DOCX, or TXT file to begin.")
