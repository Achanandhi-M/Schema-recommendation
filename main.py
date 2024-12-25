import os
import json
import numpy as np
import faiss
import torch
from transformers import BertTokenizer, BertModel
from elasticsearch import Elasticsearch
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Groq client
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# Load dataset (Cleaned_Schemes.json)
with open("Cleaned_Schemes.json", "r") as f:
    schemes_data = json.load(f)

# Initialize Elasticsearch
es = Elasticsearch(
    ["http://localhost:9200"],
    basic_auth=("elastic", "#A6s3h8*+Mr")
)

# Create Elasticsearch index
def create_es_index():
    if not es.indices.exists(index="schemes"):
        es.indices.create(index="schemes")
        for idx, scheme in enumerate(schemes_data):
            es.index(index="schemes", id=idx, body=scheme)

# Search Elasticsearch
def search_es(query):
    search_results = es.search(index="schemes", body={
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["Scheme Name^3", "Description^2", "Eligibility", "Benefits", "Required Documents"]
            }
        }
    })
    return [
        hit["_source"] for hit in search_results["hits"]["hits"]
    ] if search_results.get("hits", {}).get("hits") else []

# Load BERT model and tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
bert_model = BertModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Encode text using BERT
def get_bert_embedding(text):
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

def create_faiss_index():
    combined_texts = [
        f"{scheme['Scheme Name']}. {scheme['Description']}" for scheme in schemes_data
    ]
    embeddings = np.vstack([get_bert_embedding(text) for text in combined_texts]).astype("float32")

    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings)

    return faiss_index, schemes_data

def search_faiss(query, faiss_index, schemes_data, top_k=5):
    query_embedding = get_bert_embedding(query).astype("float32").reshape(1, -1)
    distances, indices = faiss_index.search(query_embedding, top_k)
    results = [schemes_data[idx] for idx in indices[0]]
    return results

# Generate response with LLaMA-3
def generate_llama_response(scheme, query):
    scheme_name = scheme.get('Scheme Name', 'Unknown Scheme')
    description = scheme.get('Description', 'Description not available.')
    eligibility = scheme.get('Eligibility', 'Eligibility criteria not specified.')
    benefits = scheme.get('Benefits', 'Benefits not mentioned.')
    required_documents = scheme.get('Required Documents', 'Required documents not specified.')

    prompt = (
        f"Query: {query}\n"
        f"Scheme Name: {scheme_name}\n"
        f"Description: {description}\n"
        f"Eligibility: {eligibility}\n"
        f"Benefits: {benefits}\n"
        f"Required Documents: {required_documents}\n\n"
        "Provide a focused and accurate response to the user's query about this scheme."
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt}
        ],
        model="llama-3.1-70b-versatile",
    )

    return chat_completion.choices[0].message.content

# Combine FAISS and Elasticsearch results
def combine_results(es_results, faiss_results, query_lower):
    faiss_weight = 3
    exact_match_weight = 5
    partial_match_weight = 2

    scores = {}

    for result in faiss_results:
        scheme_name = result["Scheme Name"].lower()
        scores[scheme_name] = scores.get(scheme_name, 0) + faiss_weight

    for result in es_results:
        scheme_name = result["Scheme Name"].lower()
        if query_lower in scheme_name:
            scores[scheme_name] = scores.get(scheme_name, 0) + exact_match_weight
        elif any(word in scheme_name for word in query_lower.split()):
            scores[scheme_name] = scores.get(scheme_name, 0) + partial_match_weight
        else:
            scores[scheme_name] = scores.get(scheme_name, 0) + 1

    sorted_schemes = sorted(scores.items(), key=lambda x: -x[1])
    combined_results = [
        res
        for res in faiss_results + es_results
        if res["Scheme Name"].lower() in dict(sorted_schemes)
    ]

    return combined_results

# Chatbot response
def chatbot_response(query):
    query_lower = query.lower()
    es_results = search_es(query)
    faiss_index, data = create_faiss_index()
    faiss_results = search_faiss(query, faiss_index, data)

    combined_results = combine_results(es_results, faiss_results, query_lower)

    if combined_results:
        best_match = combined_results[0]
        return generate_llama_response(best_match, query)

    return "Sorry, I couldn't find any scheme matching your query. Please try again with a different query."

# Main Function
if __name__ == "__main__":
    create_es_index()
    print("Chatbot initialized. Type your query below:")
    while True:
        user_query = input("You: ")
        if user_query.lower() in ["exit", "quit"]:
            print("Exiting chatbot. Goodbye!")
            break
        response = chatbot_response(user_query)
        print("Chatbot:", response)
