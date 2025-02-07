import streamlit as st
import feedparser
from model import ArxivModel
from urllib.parse import quote


class ArxivChatbot:
    def __init__(self):
        self.initialize_states()
        self.render_ui()

    def initialize_states(self):
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "qa_chain" not in st.session_state:
            st.session_state.qa_chain = None
        if "keywords" not in st.session_state:
            st.session_state.keywords = []
        if "cohere_api_key" not in st.session_state:
            st.session_state.cohere_api_key = None

    def add_keyword(self):
        new_keyword = st.session_state.new_keyword.strip()
        if new_keyword and new_keyword not in st.session_state.keywords:
            st.session_state.keywords.append(new_keyword)
        st.session_state.new_keyword = ""

    def remove_keyword(self, keyword_to_remove):
        st.session_state.keywords.remove(keyword_to_remove)
        st.rerun()

    def get_input_keywords(self):
        st.sidebar.header("🔍 ArXiv Paper Search")
        st.sidebar.text_input(
            "Enter keywords:",
            key="new_keyword",
            on_change=self.add_keyword,
            placeholder='E.g., visual question answering'
        )

    def display_input_keywords(self):
        if st.session_state.keywords:
            st.sidebar.write("#### Keyword(s) Entered:")
            for kw in st.session_state.keywords:
                with st.sidebar.container():
                    col1, col2 = st.columns([4, 1])  # Reduce spacing
                    with col1:
                        st.markdown(f"- {kw}")
                    with col2:
                        if st.button("✖", key=f"remove_{kw}", type="secondary"):
                            self.remove_keyword(kw)

    def fetch_arxiv_papers(self, keywords):
        quoted_keywords = [quote(kw) for kw in keywords]
        query = "+AND+".join(
            [f"abs:{quote(keyword)}" for keyword in quoted_keywords])
        url = (f'http://export.arxiv.org/api/query?search_query={query}'
               f'&start=0&max_results=10&sortBy=lastUpdatedDate&sortOrder=descending')
        print(url)  # debug
        return feedparser.parse(url)

    def prepare_llm(self, keywords, feed):
        arxiv_instance = ArxivModel(st.session_state.cohere_api_key)
        if feed.entries:
            st.session_state.qa_chain = arxiv_instance.get_model(feed.entries)
            st.sidebar.write(f"Fetched {len(feed.entries)} papers for:")
            for kw in keywords:
                st.sidebar.write(f"- {kw}")
        else:
            st.sidebar.warning(
                "Failed to fetch papers. Try different keywords.")

    def process_chat(self, prompt):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Thinking..."):
            response = st.session_state.qa_chain.invoke(prompt)

        st.session_state.messages.append(
            {"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

    def render_ui(self):
        st.title("ArXiv Q&A")
        st.write(
            "Retrieve recent papers from **ArXiv** and ask AI questions about them.")

        cohere_key = st.text_input("Cohere API Key", type="password")
        if not cohere_key:
            st.info("Please add your Cohere API key to continue.", icon="🗝️")
            return

        st.session_state.cohere_api_key = cohere_key
        self.get_input_keywords()
        self.display_input_keywords()

        with st.sidebar.container():
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Fetch Papers", type="primary"):
                    if not st.session_state.keywords:
                        st.sidebar.warning("Enter at least one keyword!")
                    else:
                        feed = self.fetch_arxiv_papers(
                            st.session_state.keywords)
                        self.prepare_llm(st.session_state.keywords, feed)
            with col2:
                if st.button("Reset", type="secondary"):
                    st.session_state.keywords = []
                    st.session_state.qa_chain = None
                    st.rerun()

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask a question about the papers..."):
            if not st.session_state.qa_chain:
                st.warning(
                    "Please provide keywords related to a topic you want to discuss about.")
            else:
                self.process_chat(prompt)


if __name__ == "__main__":
    ArxivChatbot()
