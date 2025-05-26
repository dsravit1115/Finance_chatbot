import os, re, spacy
import streamlit as st
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from evaluate_metrics import calculate_context_recall, calculate_exact_match

load_dotenv()
nlp = spacy.load("en_core_web_sm")
api_key = os.getenv("OPENAI_API_KEY")

def mask_pii(text):
    doc = nlp(text)
    masked = text
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "EMAIL", "PHONE", "GPE", "ORG", "MONEY"]:
            masked = masked.replace(ent.text, f"[MASKED_{ent.label_}]")
    masked = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '[MASKED_EMAIL]', masked)
    return masked

def semantic_chunk(text, max_tokens=200):
    doc = nlp(text)
    chunks, chunk = [], ""
    for sent in doc.sents:
        if len(chunk) + len(sent.text) <= max_tokens:
            chunk += sent.text + " "
        else:
            chunks.append(chunk.strip())
            chunk = sent.text + " "
    if chunk:
        chunks.append(chunk.strip())
    return chunks

# Streamlit UI
st.title(" Secure Finance RAG Bot")
file = st.file_uploader("Upload financial knowledge file (.txt)", type="txt")
mask = st.checkbox("Mask PII")

if file:
    raw = file.read().decode("utf-8")
    if mask: raw = mask_pii(raw)
    chunks = semantic_chunk(raw)
    docs = [Document(page_content=c) for c in chunks]

    embed = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(docs, embed)
    llm = ChatOpenAI(openai_api_key=api_key, model_name="gpt-3.5-turbo")
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())

    query = st.text_input("Ask a finance-related question")
    if query:
        answer = qa.run(query)
        st.success(answer)

        # Accuracy Evaluation (optional)
        retrieved_docs = vectordb.similarity_search(query, k=2)
        source_texts = [d.page_content for d in retrieved_docs]
        recall = calculate_context_recall(source_texts, answer)
        exact = calculate_exact_match(answer, query)
        st.write(f" Context Recall: {recall:.2f} |  Exact Match: {exact}")
