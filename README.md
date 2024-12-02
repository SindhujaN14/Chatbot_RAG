# Web Scraping and Similarity Search API

## Overview
This project provides a Python-based web application for scraping data from specified websites, storing the scraped data in a vector database (Milvus), and performing semantic similarity searches using a pre-trained language model (`sentence-transformers/all-MiniLM-L6-v2`). It also supports OpenAI-powered responses based on retrieved results.

The application is built using Flask, integrates with Milvus for vector storage, and uses OpenAI's GPT-3.5 for generating context-aware responses.

---

## Features
1. **Web Scraping:**
   - Scrapes text (headers, paragraphs, and other textual content) from specified websites.
   - Extracted data is stored as text embeddings in Milvus.

2. **Embedding Generation:**
   - Converts textual data into embeddings using the `sentence-transformers/all-MiniLM-L6-v2` model.

3. **Vector Storage:**
   - Utilizes Milvus for storing and managing vector embeddings with metadata such as text content and source website.

4. **Semantic Similarity Search:**
   - Enables similarity searches within stored embeddings using cosine similarity.

5. **OpenAI Integration:**
   - Generates context-aware responses based on similarity search results and user-provided prompts.

---

## Technologies Used
- **Flask**: API framework.
- **Milvus**: Vector database for managing text embeddings.
- **PyTorch**: Used for loading and running the transformer model.
- **BeautifulSoup**: For web scraping.
- **OpenAI API**: For generating responses.
- **Transformers Library**: To generate embeddings from text.

---

## Setup Instructions

### Prerequisites
1. Python 3.8 or later.
2. Milvus installed and running (host and port configurable).
3. Install the required Python libraries:
   ```bash
   pip install flask flask-cors pymilvus requests beautifulsoup4 transformers torch openai
