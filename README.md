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
4. Obtain an OpenAI API key and set it as OPENAI_API_KEY in the script.
   
# Configuration
Milvus Configuration:

Update MILVUS_HOST and MILVUS_PORT in the script with your Milvus server's address and port.
OpenAI API Key:

Replace the placeholder in OPENAI_API_KEY with your actual OpenAI API key.
Websites to Scrape:

Update the websites dictionary with website names and URLs.
# Running the Application
Run the Flask application:

bash
Copy code
python app.py
The application will start at http://0.0.0.0:5006.

# API Endpoints
1. Scrape Websites
Endpoint: GET /scrape
Description: Scrapes data from predefined websites, generates embeddings, and stores them in Milvus.
Response:
Success: {"message": "Data scraping completed successfully."}
Error: {"error": "Error message"}
2. Query Language Model
Endpoint: POST /query_lm
Description: Performs similarity search and generates a response using OpenAI API.
Request Body:
json
Copy code
{
  "query": "Your query here",
  "website_name": "Optional website filter"
}
Response:
json
Copy code
{
  "response": "Generated response based on query and similarity search"
}

# Key Functions
1. Web Scraping
scrape_website(url): Fetches text content (headers, paragraphs, etc.) from the given URL.
scrape_all_websites(websites): Iterates over a dictionary of website names and URLs, scraping and storing data in Milvus.
2. Embedding Creation
create_embeddings(texts): Converts text into embeddings using a transformer model.
mean_pooling(model_output, attention_mask): Pools embeddings for each text input.
3. Milvus Integration
insert_data(collection_name, texts, website_name): Inserts text and embeddings into a Milvus collection.
similarity_search(collection_name, query_embedding, top_k, website_name_filter): Performs similarity search on stored embeddings.
4. OpenAI Integration
generate_response(prompt): Generates a contextual response based on a user prompt and retrieved data.

# Milvus Collection Configuration
Name: collection_Banking
Schema:
ID: Auto-incremented integer.
Text: Scraped text (max length 500).
Embedding: 384-dimensional vector.
Website Name: Source website (max length 100).
Indexing:

Index type: IVF_FLAT.
Metric: COSINE.
Future Improvements
Add support for dynamic website input.
Enhance scraping logic to handle JavaScript-heavy websites.
Enable batch operations for large-scale scraping and embedding storage.
Improve error handling and logging mechanisms.
Contributing
Feel free to fork this repository and contribute by:

Adding new features.
Improving scraping mechanisms.
Optimizing embedding storage and search.
For questions or contributions, reach out via GitHub Issues.

License
This project is open-source and available under the MIT License.
