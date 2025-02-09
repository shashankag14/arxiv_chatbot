from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document
from langchain.memory import ConversationSummaryMemory
from langchain_core.prompts import PromptTemplate
from langchain_cohere import ChatCohere, CohereEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import feedparser
import faiss
import json
import os
import yaml
import streamlit as st


class ArxivModel:
    def __init__(self, api_key=None):

        # Use .env file to load keys if not passed explicitly
        if api_key is None:
            self._set_api_keys()
        else:
            os.environ["COHERE_API_KEY"] = api_key

        # Load prompts from YAMl
        with open('./prompts/prompt_templates.yaml', 'r') as file:
            self.prompts = yaml.safe_load(file)

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

    def get_llm(self):
        return ChatCohere(model="command-r-plus-08-2024",
                          max_tokens=256,
                          temperature=0.5
                          )

    def create_vector_db(self, docs):
        # Load a pre-trained embedding model
        embedding_model = CohereEmbeddings(model="embed-english-light-v3.0")

        index = faiss.IndexFlatL2(len(embedding_model.embed_query("Hello LLM")))

        vector_db = FAISS(
            embedding_function=embedding_model,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

        vector_db.add_documents(docs)

        return vector_db

    def create_conversational_memory(self, llm):
        # Get prompt template from YAML for the chat summarization
        summary_prompt_template = self.prompts["chat_summary"]["v_1_0"]

        # Create prompt
        summary_prompt = PromptTemplate(
            template=summary_prompt_template,
            input_variables=["summary", "new_lines"]    # keep the input variable names the same!
        )

        # Create object to hold chat memory
        memory = ConversationSummaryMemory(llm=llm,
                                           prompt=summary_prompt,
                                           memory_key="chat_history",
                                           return_messages=True,
                                           verbose=True)
        return memory

    def get_prompt_for_qa(self):
        # Get prompt template from YAML for the QA Chain
        main_prompt_template = self.prompts["qa_chain"]["v_1_0"]

        # Create prompt
        main_prompt = PromptTemplate(
            template=main_prompt_template,
            input_variables=["context", "chat_history", "question"])

        return main_prompt

    def create_qa_chain(self, vector_db, llm):
        retriever = vector_db.as_retriever()
        memory = self.create_conversational_memory(llm)
        prompt = self.get_prompt_for_qa()

        input_dict = {
            "context": retriever,
            "chat_history": lambda x: memory.load_memory_variables(x)["chat_history"],
            "question": RunnablePassthrough(verbose=True)
        }

        qa_chain = (
            input_dict
            | prompt
            | llm
            | StrOutputParser()
            )

        return qa_chain

    def get_model(self, data):
        docs = self.create_documents(data)
        vector_db = self.create_vector_db(docs)
        llm = self.get_llm()
        qa_chain = self.create_qa_chain(vector_db, llm)
        return qa_chain


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

    qa_chain = arxiv_model.get_model(data)

    for chunk in qa_chain.stream("What is LLM?"):
        print(chunk, end="", flush=True)

    if USE_STREAMLIT:
        # User input for query
        query = st.text_input(
            "Ask a question about papers related to RAG:")

        # Show the answer when the user submits a question
        if query:
            with st.spinner("Thinking..."):
                response = qa_chain.invoke(query)
                st.write("Answer:")
                st.write(response)
