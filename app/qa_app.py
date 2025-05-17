!pip install faiss-cpu sentence-transformers transformers PyPDF2


import re
from PyPDF2 import PdfReader

reader = PdfReader("white_paper.pdf")
raw_text = ""
for page in reader.pages:
    text = page.extract_text()
    if text:
        raw_text += text + " "

cleaned_text = re.sub(r'\s+', ' ', raw_text)
cleaned_text = re.sub(r'(?<=\w)- (?=\w)', '', cleaned_text)
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
