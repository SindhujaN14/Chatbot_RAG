import os
import requests
from bs4 import BeautifulSoup
from flask import Flask, jsonify, request, Response
from flask_cors import CORS
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections, utility
import numpy as np
import logging
import openai
import torch
from transformers import AutoTokenizer, AutoModel
import traceback

# Flask app initialization
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.DEBUG)

# Milvus connection settings
MILVUS_HOST = "10.100.7.20"
MILVUS_PORT = 19530

# OpenAI API key configuration
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
openai.api_key = OPENAI_API_KEY

# Load the tokenizer and model for embedding generation
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def create_embeddings(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        model_output = model(**inputs)
    embeddings = mean_pooling(model_output, inputs['attention_mask'])
    return embeddings.numpy()

def scrape_website(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching URL {url}: {e}")
        return {'error': f'Failed to fetch URL {url}'}
    soup = BeautifulSoup(response.text, 'html.parser')
    headers_content = [header.get_text().strip() for header in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])]
    paragraphs = [p.get_text().strip() for p in soup.find_all('p')]
    other_texts = [div.get_text(strip=True) for div in soup.find_all(['div', 'span'])]
    data = {
        'headers': headers_content,
        'paragraphs': paragraphs,
        'other_texts': other_texts
    }
    return data

def scrape_all_websites(websites):
    for website_name, url in websites.items():
        try:
            # Scrape data from the website
            data = scrape_website(url)
            print(f"Scraped data for {website_name}: ", data)
            # Combine headers, paragraphs, and other texts if they exist
            headers = data.get('headers', [])
            paragraphs = data.get('paragraphs', [])
            other_texts = data.get('other_texts', [])
            texts = headers + paragraphs + other_texts
            # Insert data into Milvus collection as embeddings with metadata
            insert_data(collection_name, texts, website_name)
            print(f"Data for {website_name} inserted into Milvus collection.")
        except Exception as e:
            # Logging error if scraping or storing fails for any website
            print(f"Error processing {website_name}: {e}")
    return {"message": "Data scraped and inserted for all websites."}

# Connect to Milvus server
try:
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
    logging.info("Successfully connected to Milvus!")
except Exception as e:
    logging.error(f"Failed to connect to Milvus: {e}")
    raise

# Define the collection name and schema with metadata
collection_name = "collection_Banking"
# Drop the existing collection if it exists
if utility.has_collection(collection_name):
    logging.info(f"Collection '{collection_name}' already exists. Dropping it for re-creation.")
    utility.drop_collection(collection_name)
    
# Recreate the collection with COSINE metric
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=500),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384, metric_type="COSINE"),
    FieldSchema(name="website_name", dtype=DataType.VARCHAR, max_length=100)
]
schema = CollectionSchema(fields=fields, description="Scraped data collection with metadata")
collection = Collection(name=collection_name, schema=schema)
logging.info(f"Collection '{collection_name}' created with COSINE metric.")

# Create an index for the embedding field
index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "COSINE",  # Set the metric type to COSINE
    "params": {"nlist": 128}
}
collection.create_index("embedding", index_params, index_name="embedding_index")
logging.info("Created Milvus collection and index for embeddings with COSINE metric.")

def insert_data(collection_name, texts, website_name):
    collection = Collection(collection_name)
    embeddings = create_embeddings(texts)
    if embeddings is not None:
        data = [texts, embeddings.tolist(), [website_name] * len(texts)]
        collection.insert(data)
        collection.load()
        logging.info(f"Inserted {len(texts)} records from '{website_name}' into '{collection_name}'.")
    else:
        logging.error("Embeddings list is empty. Data not inserted.")
        
def similarity_search(collection_name, query_embedding, top_k=3, website_name_filter=None):
    try:
        if not utility.has_collection(collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist.")
        collection = Collection(collection_name)
        collection.load()
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        if website_name_filter:
            results = collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["text", "website_name"],
                expr=f"website_name == '{website_name_filter}'"
            )
        else:
            results = collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["text", "website_name"]
            )
        return results
    except Exception as e:
        logging.error(f"Error during similarity search: {e}")
        logging.error(traceback.format_exc())
        raise
    
def generate_response(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message['content']
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        return "Error generating response."
    
@app.route('/scrape', methods=['GET'])
def scrape_endpoint():
    websites = {
        "Bank of America": "https://www.bankofamerica.com/",
        "Chase": "https://www.chase.com/",
        "Citi": "https://www.citi.com/"
    }
    try:
        print("scrape_data")
        Response = scrape_all_websites(websites)
        print("scrape_test")
        return jsonify({"message": "Data scraping completed successfully."}), 200
    except Exception as e:
        print("execption")
        return jsonify({"error": str(e)}), 500
    
@app.route('/query_lm', methods=['POST'])
def query_lm():
    if not request.is_json:
        return jsonify({"error": "JSON data required"}), 400
    data = request.get_json()
    prompt = data.get("query", "")
    website_name_filter = data.get("website_name", None)
    if not prompt:
        return jsonify({"error": "Query parameter is missing"}), 400
    query_embedding = create_embeddings([prompt])[0]
    results = similarity_search(collection_name, query_embedding, top_k=3, website_name_filter=website_name_filter)
    relevant_texts = "\n".join([result.entity.get("text") for result in results[0]])
    response = generate_response(f"{prompt}\n\n{relevant_texts}")
    return jsonify({"response": response})

# Run Flask application
if __name__ == "__main__":
    websites = {
        "Bank of America": "https://www.bankofamerica.com/",
        "Chase": "https://www.chase.com/",
        "Citi": "https://www.citi.com/"
    }
    app.run(host="0.0.0.0", port=5006, debug=True)
