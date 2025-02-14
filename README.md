
# ArXiv Chatbot

#### **Try it out:** [![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://arxiv-qna.streamlit.app/)

ArXiv Chatbot is a simple web application, designed to interact with research papers using the [ArXiv API](https://info.arxiv.org/help/api/index.html). The app allows users to enter multiple keyword, fetch related research papers, and ask questions about them. The aim of the app is to use concepts from Retrieval Augmented Generation (RAG) to answer questions about the papers using PyTorch and Streamlit. Following are the main features supported by the app:

- **Search Papers**: Users can input multiple keywords and get a list of research papers from ArXiv.
- **Ask Questions**: Users can ask questions related to the multiple research papers retrieved, and the app will provide answers using a trained model.
- **Interactive UI**: A simple user interface built with Streamlit.
- **Memory Management:** The chatbot now incorporates a memory mechanism, allowing it to retain context from previous interactions within a single session. This improves the accuracy of responses over multiple questions.

> **⚠️ Note:** The app uses Cohere for LLM based responses. Without a valid Cohere API key, the bot will not work. You can get your API key from [Cohere's API Key Page](https://dashboard.cohere.com/api-keys).

## Installation

Follow these steps to get your project set up locally.

  ```bash
  git clone https://github.com/shashankag14/arxiv_chatbot.git
  cd arxiv_chatbot
  conda create --name arxiv-chatbot-env python=3.9
  conda activate arxiv-chatbot-env
  pip install -r requirements.txt
  ```

##### Add API Key

In order to test the app locally without using the streamlit app, make sure to securely store your Cohere API key (or others if you wish to change the embedding or LLM model) in a `.env` file to avoid exposing sensitive information:

1. Create a `.env` file in the project directory.
2. Add your API keys in this file:
   ```plaintext
   API_KEY=your_api_key_here
   ```

### Usage

To run the app locally, execute the following command from the root directory of the project:

```bash
streamlit run ./app.py
```

This will launch the Streamlit web interface in your browser.

## How it works?

1. **Search**: The user inputs multiple keywords in the sidebar.
2. **Fetch Papers**: The app queries the ArXiv API and fetches research papers related to the entered keywords.
3. **Q&A**: After fetching the papers, users can enter a question. The app will process the query and provide an answer based on the fetched papers.

## Frameworks Used

- **LangChain**: For LLM based Q&A chain.
- **FAISS**: For similarity search and retrieval.
- **Streamlit**: For building the web-based UI.
- **ArXiv API**: For fetching research papers from the ArXiv database.

Feel free to fork the repository and submit pull requests. If you encounter any issues, please open an issue in the repository.
