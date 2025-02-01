
# ArXiv Chatbot

ArXiv Chatbot is a simple web application, designed to interact with research papers using the [ArXiv API](https://info.arxiv.org/help/api/index.html). The app allows users to enter a keyword, fetch related research papers, and ask questions about them. The aim of the app is to use concepts from Retrieval Augmented Generation (RAG) to answer questions about the papers. Following are the main features supported by the app:

- **Search Papers**: Users can input a keyword and get a list of research papers from ArXiv.
- **Ask Questions**: Users can ask questions related to the research papers, and the app will provide answers using a trained model.
- **Interactive UI**: A simple user interface built with Streamlit.

## Installation

Follow these steps to get your project set up locally.

  ```bash
  git clone https://github.com/shashankag14/arxiv_chatbot.git
  cd arxiv_chatbot
  conda env create -f environment.yaml
  conda activate arxiv-chatbot-env
  ```

##### Add API Key

Make sure to securely store your API key (e.g., Hugging Face or others if you wish to change the embedding or LLM model) in a `.env` file to avoid exposing sensitive information:

1. Create a `.env` file in the project directory.
2. Add your API keys in this file:
   ```plaintext
   API_KEY=your_api_key_here
   ```

### Usage

To run the app locally, execute the following command from the root directory of the project:

```bash
streamlit run arxiv_chatbot_app.py
```

This will launch the Streamlit web interface in your browser.

## How it works?

1. **Search**: The user inputs a keyword in the sidebar.
2. **Fetch Papers**: The app queries the ArXiv API and fetches research papers related to the keyword.
3. **Q&A**: After fetching the papers, users can enter a question. The app will process the query and provide an answer based on the fetched papers.

## Frameworks Used

- **LangChain**: For LLM based Q&A chain.
- **FAISS**: For similarity search and retrieval.
- **Streamlit**: For building the web-based UI.
- **ArXiv API**: For fetching research papers from the ArXiv database.
- **Feedparser**: For parsing the XML feed from the ArXiv API.

Feel free to fork the repository and submit pull requests. If you encounter any issues, please open an issue in the repository.
