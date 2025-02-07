import streamlit as st
import feedparser
from model import ArxivModel
from urllib.parse import quote


def add_keyword():
    new_keyword = st.session_state.new_keyword.strip()
    # add new keywords
    if new_keyword and new_keyword not in st.session_state.keywords:
        st.session_state.keywords.append(new_keyword)
    st.session_state.new_keyword = ""


def remove_keyword(keyword_to_remove):
    st.session_state.keywords.remove(keyword_to_remove)
    st.rerun()


def display_entered_keywords():
    for kw in st.session_state.keywords:
        st.sidebar.write("#### Keyword(s) Entered:")
        # two columns: one for the keyword, one for the close button
        col1, col2 = st.sidebar.columns([1, 1])
        with col1:
            st.sidebar.markdown(f"- {kw}")  # Display the keyword
        with col2:
            # cross icon button
            if st.sidebar.button("x", key=f"remove_{kw}", type="secondary"):
                remove_keyword(kw)


def initialize_states():
    # Initialize session state for messages and model
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None

    if "keywords" not in st.session_state:
        st.session_state.keywords = []


def process_chat(prompt):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Store user message
    st.session_state.messages.append(
        {"role": "user", "content": prompt})

    # Generate response
    with st.spinner("Thinking..."):
        response = st.session_state.qa_chain.invoke(prompt)

    # Display assistant message
    with st.chat_message("assistant"):
        st.markdown(response)

    # Store assistant message
    st.session_state.messages.append(
        {"role": "assistant", "content": response})


# Title of the app
st.title("ArXiv Q&A")
st.write(
    "This chatbot retrieves recent papers from **ArXiv** based on a keyword and uses Cohere's 'command-r-plus-08-2024' model to answer questions. "
    "To use this app, provide your Cohere API key below. "
    "You can get an API key from [Cohere's platform](https://dashboard.cohere.com/api-keys)."
)

if "cohere_api_key" not in st.session_state:
    st.session_state.cohere_api_key = None

cohere_key = st.text_input("Cohere API Key", type="password")
if not cohere_key:
    st.info("Please add your Cohere API key to continue.", icon="🗝️")
else:
    st.session_state.cohere_api_key = cohere_key

    initialize_states()

    st.sidebar.header("🔍 ArXiv Paper Search")
    st.sidebar.text_input(
        "Enter keywords to fetch papers:",
        key="new_keyword",
        on_change=add_keyword,
        placeholder='E.g., visual question answering')

    display_entered_keywords()

    col1, col2 = st.sidebar.columns([1, 1])
    with col2:
        if st.sidebar.button("Reset", type="secondary"):
            st.session_state.keywords = []
            st.rerun()
    with col1:
        if st.sidebar.button("Fetch Papers", type="primary"):
            keywords = st.session_state.keywords

            if not keywords:
                st.warning("Please enter at least one keyword!")
                st.rerun()

            # refine the keyword by removing extra spaces etc.
            quoted_keywords = [quote(kw) for kw in keywords]

            # Construct ArXiv query using AND condition
            query = "+AND+".join([f"abs:{quote(keyword)}" for keyword in quoted_keywords])
            url = f'http://export.arxiv.org/api/query?search_query={query}&start=0&max_results=10&sortBy=lastUpdatedDate&sortOrder=descending'
            print(url)

            feed = feedparser.parse(url)
            arxiv_instance = ArxivModel(st.session_state.cohere_api_key)
            if len(feed.entries):
                st.session_state.qa_chain = arxiv_instance.get_model(feed.entries)
                # display the used keywords
                st.sidebar.write(
                    f"Fetched {len(feed.entries)} papers related to:\n")
                for kw in keywords:
                    st.sidebar.write(f"- {kw}")
            else:
                st.sidebar.warning(
                    "Failed to fetch papers. Please try changing the keywords.")

    # Display existing chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input field
    if prompt := st.chat_input("Ask a question about the papers..."):
        if not st.session_state.qa_chain:
            st.warning("Please provide a keyword to fetch papers first.")
        else:
            process_chat(prompt)
