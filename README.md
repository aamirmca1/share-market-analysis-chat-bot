# AI Research Chatbot 📈

An AI-powered Research Chatbot that processes news articles and websites, stores their embeddings in a FAISS vector database, and answers user questions using Retrieval-Augmented Generation (RAG).

## Features

- Load and process multiple website URLs
- Extract and chunk website content
- Generate embeddings using Hugging Face Sentence Transformers
- Store embeddings locally using FAISS
- Perform semantic similarity search
- Generate context-aware answers using Groq Llama 3.1
- Interactive Streamlit web interface

## Tech Stack

- Python
- Streamlit
- LangChain
- Hugging Face Embeddings
- FAISS Vector Store
- Groq LLM
- Sentence Transformers

## Architecture

```text
Website URLs
      ↓
Document Loader
      ↓
Text Splitter
      ↓
Embeddings
      ↓
FAISS Vector Store
      ↓
Similarity Search
      ↓
Groq LLM
      ↓
Answer Generation
```

## Installation

### Clone Repository

```bash
git clone https://github.com/aamirmca1/share-market-analysis-chat-bot.git
cd share-market-analysis-chat-bot
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Virtual Environment

Windows:

```bash
venv\Scripts\activate
```

Mac/Linux:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
```

## Run Application

```bash
streamlit run app.py
```

## Usage

1. Enter one or more article URLs.
2. Click **Process URLs**.
3. Wait for embeddings and vector database creation.
4. Ask questions about the loaded articles.
5. Receive AI-generated answers based on retrieved context.

## Sample Questions

- What is the company's revenue growth?
- What are the key risks mentioned in the article?
- What are the latest business updates?
- Summarize the article in 5 points.

## Project Structure

```text
share-market-analysis-chat-bot/
│
├── app.py
├── requirements.txt
├── .env
├── README.md
├── faiss_index/
└── venv/
```

## Future Enhancements

- Chat history support
- Multi-document search
- PDF support
- Source citation links
- Conversation memory
- Deployment on Streamlit Cloud

## Author

Mohammed Aamir
