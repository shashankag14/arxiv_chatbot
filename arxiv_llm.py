from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.schema import Document
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.llms import HuggingFaceHub
import os
import faiss
import pathlib
import json
import streamlit as st


os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

DATA_PATH = "./data/arxiv_papers.json"

with open(DATA_PATH, "r") as f:
    arxiv_papers = json.load(f)

docs = []
for paper in arxiv_papers:
    title = paper["title"]
    abstract = paper["summary"]
    link = paper["link"]
    paper_content = f"Title: {title}\nAbstract: {abstract}"
    paper_content = paper_content.lower()

    docs.append(Document(page_content=paper_content,
                         metadata={"link": link}))


# Load a pre-trained embedding model
embedding_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

index = faiss.IndexFlatL2(len(embedding_model.embed_query("Hello LLM")))

vector_store = FAISS(
    embedding_function=embedding_model,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

vector_store.add_documents(docs)

retriever = vector_store.as_retriever()

llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct")

qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

st.title("Arxiv Papers Q&A")
# User input for query
query = st.text_input("Ask a question about Arxiv papers:")

# Show the answer when the user submits a question
if query:
    with st.spinner("Thinking..."):
        response = qa.invoke(query)
        st.write("Answer:")
        st.write(response)
