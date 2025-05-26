# CLI version for quick testing
from app import semantic_chunk, mask_pii
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

with open("sample_finance.txt", "r", encoding="utf-8") as f:
    text = f.read()

text = mask_pii(text)
chunks = semantic_chunk(text)
docs = [Document(page_content=chunk) for chunk in chunks]

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embedding)
llm = ChatOpenAI(openai_api_key=api_key, model_name="gpt-3.5-turbo")
qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

query = "What is a mutual fund?"
print("Q:", query)
print("A:", qa.run(query))
