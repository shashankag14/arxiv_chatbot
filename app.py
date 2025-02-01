import streamlit as st
import feedparser
import model
from urllib.parse import quote

# Title of the app
st.title("ArXiv Q&A")

# Sidebar for keyword input
keyword = st.sidebar.text_input("Enter the keyword for ArXiv search:")

query = st.sidebar.text_input(
    f"Ask a question about papers related to {keyword}:")

# Button to submit the keyword
submit_button = st.sidebar.button("Submit")

# Display query input after submit_button is clicked
if submit_button and keyword and query:
    with st.spinner("Thinking..."):
        # Encode the keyword for the URL
        encoded_keyword = quote(keyword)

        # ArXiv API URL
        url = f'http://export.arxiv.org/api/query?search_query=abs:{encoded_keyword}&start=0&max_results=1000&sortBy=lastUpdatedDate&sortOrder=descending'

        # Parse the API response
        feed = feedparser.parse(url)

        docs = model.create_documents(feed.entries)
        vector_db = model.create_vector_db(docs)
        qa = model.create_qa_chain(vector_db)

        # Show the answer when the user submits a question
        response = qa.invoke(query)
        st.write("Answer:")
        st.write(response)

else:
    st.sidebar.write("Please enter a keyword and query to submit.")
