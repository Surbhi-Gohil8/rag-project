# RAG Assistant

A sophisticated Retrieval-Augmented Generation (RAG) application that allows you to upload documents, scrape web content, and chat with your data using AI-powered conversational interfaces.

---

## Features

* **Document Processing**: Support for PDF, TXT, and CSV files
* **Web Content Scraping**: Extract and process content from any URL
* **Intelligent Chat**: Natural language conversation with your uploaded content
* **Vector Search**: Advanced semantic search using embeddings
* **Cloud Storage**: Qdrant Cloud integration for scalable vector storage
* **Modern UI**: Clean, responsive Streamlit interface
* **Session Management**: Isolated sessions for different conversations

---

## Quick Start

### Prerequisites

* Python 3.8 or higher
* Ollama installed and running (`ollama serve`)
* Qdrant Cloud account (or self-hosted Qdrant)

### Installation

1. **Clone the repository**

```bash
git clone <repository-url>
cd rag-assistant
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Set up environment variables**

```bash
cp .env.example .env
```

Edit `.env` with your configuration:

```env
# Qdrant Cloud Configuration
QDRANT_URL=your-qdrant-cloud-url
QDRANT_API_KEY=your-qdrant-api-key
COLLECTION_NAME=rag_documents

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral
EMBEDDING_MODEL=nomic-embed-text
```

4. **Start Ollama**

```bash
ollama serve
ollama pull mistral
ollama pull nomic-embed-text
```

5. **Run the application**

```bash
streamlit run app.py
```

---

## Usage

### Adding Content

1. **Upload Documents**:

   * Click "Choose files" to upload PDF, TXT, or CSV files
   * Multiple files can be uploaded simultaneously
   * Files are automatically processed and chunked

2. **Add Web Content**:

   * Enter a URL in the "Add Web Content" section
   * Click "Add URL" to scrape and process the content
   * Supports most web pages with accessible content

### Chatting with Your Data

1. Once content is added, the status indicator will show "Ready to chat"
2. Type your question in the chat input field
3. Press Enter or click "Send" to get AI-powered responses
4. Responses include source citations for transparency

### Session Management

* Each session has a unique ID for data isolation
* Use "Clear All" to reset everything and start fresh
* Sessions are automatically cleaned up when cleared

---

## Architecture

### Core Components

* **app.py**: Main Streamlit application with UI and session management
* **src/rag_pipeline.py**: Central orchestrator for RAG operations
* **src/document_processor.py**: Document parsing and chunking
* **src/web_scraper.py**: Web content extraction
* **src/vector_store.py**: Qdrant integration for vector storage
* **src/embeddings.py**: Text embedding generation
* **src/llm_client.py**: Ollama client for LLM interactions

### Data Flow

1. **Ingestion**: Documents/web content → Text chunks → Embeddings → Vector store
2. **Query**: User question → Embedding → Vector search → Context retrieval
3. **Generation**: Context + question → LLM → Response with sources

---

## Configuration

### Environment Variables

| Variable          | Description            | Default                                          |
| ----------------- | ---------------------- | ------------------------------------------------ |
| `QDRANT_URL`      | Qdrant Cloud endpoint  | Required                                         |
| `QDRANT_API_KEY`  | Qdrant Cloud API key   | Required                                         |
| `COLLECTION_NAME` | Vector collection name | rag_documents                                    |
| `OLLAMA_BASE_URL` | Ollama server URL      | [http://localhost:11434](http://localhost:11434) |
| `OLLAMA_MODEL`    | Chat model             | mistral                                          |
| `EMBEDDING_MODEL` | Embedding model        | nomic-embed-text                                 |

### Supported Models

The application works with any Ollama-compatible models. Recommended combinations:

* **Chat**: mistral, llama2, codellama
* **Embeddings**: nomic-embed-text, all-minilm

---

## Project Structure

```bash
rag-assistant/
├── app.py
├── requirements.txt
├── .env
├── README.md
└── src/
    ├── __init__.py
    ├── rag_pipeline.py
    ├── document_processor.py
    ├── web_scraper.py
    ├── vector_store.py
    ├── embeddings.py
    └── llm_client.py
```

---

## Development

### Adding New Document Types

To support additional file formats, extend the `DocumentProcessor` class:

```python
def process_new_format(self, file_path: str) -> List[str]:
    # Implementation for new format
    pass
```

### Custom Embedding Models

To use different embedding models, modify the `Embeddings` class:

```python
class CustomEmbeddings:
    def __init__(self, model_name: str):
        # Custom embedding logic
        pass
```

---

## Troubleshooting

### Common Issues

1. **Ollama Connection Failed**

   * Ensure Ollama is running: `ollama serve`
   * Check if the model is installed: `ollama list`
   * Verify the `OLLAMA_BASE_URL` in `.env`

2. **Qdrant Connection Issues**

   * Verify your Qdrant Cloud URL and API key
   * Check network connectivity
   * Ensure the collection exists

3. **Document Processing Errors**

   * Check file format support
   * Verify file permissions
   * Check console logs for detailed error messages

### Debug Mode

Enable debug logging by setting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Acknowledgments

* **Streamlit** for the web app framework
* **Ollama** for local LLM capabilities
* **Qdrant** for vector storage
* **LangChain** for RAG framework components
* **Sentence Transformers** for embeddings

---

## Support

For support and questions:

* Create an issue in the GitHub repository
* Check the troubleshooting section above
* Review the console logs for detailed error messages

---

**Built with love using Python, Streamlit, and modern AI technologies**
