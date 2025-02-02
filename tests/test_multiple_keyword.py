from model import ArxivModel
import feedparser
from urllib.parse import quote


arxiv_model = ArxivModel()
keywords = ['RAG', 'LLM']

quoted_keywords = [quote(kw) for kw in keywords]

# Construct ArXiv query using AND condition
query = "+AND+".join([f"abs:{quote(keyword)}" for keyword in quoted_keywords])

# ArXiv API URL
url = f'http://export.arxiv.org/api/query?search_query={query}&start=0&max_results=10&sortBy=lastUpdatedDate&sortOrder=descending'
print(f"URL created: {url}")

# Parse the API response
feed = feedparser.parse(url)
print("Papers fetched.")

data = feed.entries

qa_chain = arxiv_model.get_model(data)

for chunk in qa_chain.stream("What is LLM?"):
    print(chunk, end="", flush=True)
