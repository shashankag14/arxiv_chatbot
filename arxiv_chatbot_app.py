import streamlit as st
import feedparser
from model import ArxivModel
from urllib.parse import quote

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

    # Initialize session state for messages and model
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None

    if "keywords" not in st.session_state:
        st.session_state.keywords = []

    st.sidebar.header("🔍 ArXiv Paper Search")
    new_keyword = st.sidebar.text_input(
        "Enter a keyword to fetch papers:")
    if new_keyword and new_keyword not in st.session_state.keywords:
        st.session_state.keywords.append(new_keyword)
        # # Reset the field after submission
        # st.sidebar.text_input(
        #     "Enter a keyword to fetch papers:", value="", key="new_keyword")
        st.rerun()

    st.write("### Keywords Entered:")
    cols = st.columns(len(st.session_state["keywords"]) + 1)

    for i, kw in enumerate(st.session_state["keywords"]):
        with cols[i]:
            if st.button(f"❌ {kw}", key=f"remove_{kw}"):
                st.session_state["keywords"].remove(kw)
                st.rerun()   # Refresh page to reflect changes

    cols = st.columns(len(st.session_state["keywords"]) + 1)

    if st.sidebar.button("Fetch Papers"):
        keywords = st.session_state["keywords"]

        if not keywords:
            st.warning("Please enter at least one keyword!")

        quoted_keywords = [quote(kw) for kw in keywords]

        # Construct ArXiv query using AND condition
        query = "+AND+".join([f"abs:{quote(keyword)}" for keyword in quoted_keywords])
        url = f'http://export.arxiv.org/api/query?search_query={query}&start=0&max_results=10&sortBy=lastUpdatedDate&sortOrder=descending'
        print(url)
        feed = feedparser.parse(url)

        arxiv_instance = ArxivModel(st.session_state.cohere_api_key)
        st.session_state.qa_chain = arxiv_instance.get_model(feed.entries)
        st.success(f"Fetched {len(feed.entries)} papers related to '{keyword}'!")
    else:
        st.warning("Please enter a keyword before fetching papers.")

    # Display existing chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input field
    if prompt := st.chat_input("Ask a question about the papers..."):
        if not st.session_state.qa_chain:
            st.warning("Please provide a keyword to fetch papers first.")
        else:
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
