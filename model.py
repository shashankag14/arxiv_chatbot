from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_cohere import ChatCohere, CohereEmbeddings
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate

from dotenv import load_dotenv
import feedparser
import faiss
import json
import os
import streamlit as st


class ArxivModel:
    def __init__(self, api_key=None):

        # Use .env file to load keys if not passed explicitly
        if api_key is None:
            self._set_api_keys()
        else:
            os.environ["COHERE_API_KEY"] = api_key

    def _set_api_keys(self):
        # load all env vars from .env file
        load_dotenv()

        # Add all such vars in OS env vars
        for key, value in os.environ.items():
            if key in os.getenv(key):  # Check if it exists in the .env file
                os.environ[key] = value

        print("All environment variables loaded successfully!")

    def load_json(self, file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
        return data

    def create_documents(self, data):
        docs = []
        for paper in data:
            title = paper["title"]
            abstract = paper["summary"]
            link = paper["link"]
            paper_content = f"Title: {title}\nAbstract: {abstract}"
            paper_content = paper_content.lower()

            docs.append(Document(page_content=paper_content,
                                 metadata={"link": link}))

        return docs

    def create_retriever(self, docs):
        # Load a pre-trained embedding model
        embedding_model = CohereEmbeddings(model="embed-english-light-v3.0")

        index = faiss.IndexFlatL2(
            len(embedding_model.embed_query("Hello LLM")))

        vector_db = FAISS(
            embedding_function=embedding_model,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

        vector_db.add_documents(docs)

        self.retriever = vector_db.as_retriever()

    def create_history_aware_retreiver(self):
        system_prompt_to_reformulate_input = (
            "Given a chat history and the latest user question "
            "which might reference the context in the chat history "
            "formulate a standalone question which can be understood "
            "without chat history. Do NOT answer the question, "
            "just reformulate it is needed and otherwise return it as is."
        )

        prompt_to_reformulate_input = ChatPromptTemplate.from_messages([
            ("system", system_prompt_to_reformulate_input),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        history_aware_retriever_chain = create_history_aware_retriever(
            self.llm, self.retriever, prompt_to_reformulate_input
        )
        return history_aware_retriever_chain

    def get_prompt(self):
        system_prompt = ("You are an AI assistant called 'ArXiv Assist' that has a conversation with the user and answers to questions based on the provided research papers. "
                         "There could be cases when user does not ask a question, but it is just a statement. In such cases, do not use knowledge from the papers. Just reply back normally and accordingly to have a good conversation (e.g. 'You're welcome' if the input is 'Thanks'). "
                         "If no relevant information is found in the papers, respond with: 'Sorry, I do not have much information related to this. "
                         "If you reference a paper, provide a hyperlink to it. "
                         "Be polite, friendly, and format your response well (e.g., use bullet points, bold text, etc.). "
                         "Below are relevant excerpts from the research papers:\n{context}\n\n"
                         )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "Answer the following question: {input}")
        ])

        return prompt

    def create_qa_chain(self):
        # Generate main prompt for the final chain
        prompt = self.get_prompt()

        # Subchain 1: Create ``history aware´´ retriever chain that uses conversation history tp update docs
        history_aware_retriever_chain = self.create_history_aware_retreiver()

        # Subchain 2: Create chain to send docs to LLM
        qa_chain = create_stuff_documents_chain(llm=self.llm, prompt=prompt)

        # Main chain: Create a chain that connects the two subchains
        rag_chain = create_retrieval_chain(
            retriever=history_aware_retriever_chain,
            combine_docs_chain=qa_chain)
        return rag_chain

    def get_model(self, data):
        docs = self.create_documents(data)
        self.create_retriever(docs)
        self.llm = ChatCohere(
            model="command-r-plus-08-2024", max_tokens=256, temperature=0.5)
        rag_chain = self.create_qa_chain()
        return rag_chain


if __name__ == "__main__":
    USE_PRELOADED_DATA = False
    USE_STREAMLIT = False

    arxiv_model = ArxivModel()

    # Use preloaded arxiv data in JSON format for RAG
    if USE_PRELOADED_DATA:
        DATA_PATH = "./data/arxiv_papers.json"
        data = arxiv_model.load_json(DATA_PATH)
    # Use direct API URL to parse the data from arxiv using keyword based search
    else:
        KEYWORD = "RAG"
        # ArXiv API URL
        url = f'http://export.arxiv.org/api/query?search_query=abs:{KEYWORD}&start=0&max_results=10&sortBy=lastUpdatedDate&sortOrder=descending'
        # Parse the API response
        feed = feedparser.parse(url)
        data = feed.entries

    llm_chain = arxiv_model.get_model(data)

    response = llm_chain.invoke({"input": "Hello!"})
    print(response["answer"])

    for chunk in llm_chain.stream("What is LLM?"):
        print(chunk, end="", flush=True)

    if USE_STREAMLIT:
        # User input for query
        query = st.text_input(
            "Ask a question about papers related to RAG:")

        # Show the answer when the user submits a question
        if query:
            with st.spinner("Thinking..."):
                response = llm_chain.invoke(query)
                st.write("Answer:")
                st.write(response)
