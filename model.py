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
import streamlit as st


class ArxivModel:
    def __init__(self, api_key=None):

        if api_key is None:
            self._set_api_keys()
        else:
            os.environ["COHERE_API_KEY"] = api_key

    def _set_api_keys(self):
        # Load environment variables from the .env file
        load_dotenv()

        # Check if the API keys are present
        if not os.getenv('HUGGINGFACE_API_KEY'):
            raise ValueError(
                "Error: HUGGINGFACE_API_KEY is not set. Please add it to your .env file.")
        else:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv('HUGGINGFACE_API_KEY')

        if not os.getenv('OPENAI_API_KEY'):
            raise ValueError(
                "Error: OPENAI_API_KEY is not set. Please add it to your .env file.")
        else:
            os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

        if not os.getenv('COHERE_API_KEY'):
            raise ValueError(
                "Error: OPENAI_API_KEY is not set. Please add it to your .env file.")
        else:
            os.environ["COHERE_API_KEY"] = os.getenv('COHERE_API_KEY')

        print("API keys loaded successfully!")

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
        summary_prompt_template = """Summarize the conversation and update with new lines.
        Current summary: {summary}
        New lines: {new_lines}
        Updated summary:
        """
        summary_prompt = PromptTemplate(
            template=summary_prompt_template,
            input_variables=["summary", "new_lines"]
        )

        memory = ConversationSummaryMemory(llm=llm,
                                           prompt=summary_prompt,
                                           memory_key="chat_history")
        return memory

    def get_prompt_for_qa(self):

        main_prompt_template = """You are an AI assistant called 'ArXiv Assist' that has a conversation with the user and answers to questions strictly based on the provided research papers or based on the conversational history passed to you.
        Below are relevant excerpts from the papers:

        {context}

        Below is the conversation history:
        {chat_history}

        Based on the above information, answer the following question:
        {question}

        Note:
        - There could be cases when its not a question but just a statement. In such cases, do not use knowledge from the papers. Just reply back with what you have learned before (e.g. 'You're welcome' if the input is 'Thanks').
        - If you feel its a question and you do not find relevant information in the given papers or the conversational memory, respond with 'Sorry, I do not have much information related to this. But this could be something close to what you are looking for...' and then respond back with the knowledge you have apart from the papers.
        - Be polite and friendly in your responses.
        """
        main_prompt = PromptTemplate(
            template=main_prompt_template, input_variables=["context", "question", "chat_history"])

        return main_prompt

    def create_qa_chain(self, vector_db, llm):
        retriever = vector_db.as_retriever()
        memory = self.create_conversational_memory(llm)
        prompt = self.get_prompt_for_qa()

        qa_chain = (
            {"context": retriever, "chat_history": lambda x: memory.load_memory_variables(
                x)["chat_history"], "question": RunnablePassthrough()}
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
