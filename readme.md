# **Scheme Recommendation and Chatbot Assistant using  Bert and LLama3**

## **Overview**
The Scheme Recommendation and Chatbot Assistant is an intelligent system that combines advanced search and natural language processing capabilities to help users find government or other relevant schemes based on their queries. It integrates Elasticsearch, FAISS for vector-based search, and a pre-trained BERT model to deliver accurate search results and uses the LLaMA model to provide human-like responses.

---

## **Features**
- **Elasticsearch Integration:** Provides text-based search functionality across scheme data.
- **FAISS Integration:** Delivers semantic search using vector embeddings for improved relevance.
- **BERT Model:** Encodes text into embeddings for FAISS search.
- **LLaMA-Powered Chatbot:** Generates detailed and context-aware responses based on search results.
- **Combined Search Logic:** Merges results from FAISS and Elasticsearch for comprehensive responses.
- **Scalable Design:** Easily extendable to accommodate more schemes or new AI models.

---

## **Project Structure**

```plaintext
.
├── Cleaned_Schemes.json   # Dataset containing information about various schemes
├── main.py                # Main entry point for the chatbot application
├── requirements.txt       # List of Python dependencies for the project
├── venv/                  # Virtual environment directory (optional)
```

---

## **Setup Instructions**

### **Prerequisites**
1. Install Python 3.8+.
2. Install Elasticsearch and start the service locally.
3. Create an account on **Groq** and obtain an API key.

### **Environment Variables**
Create a `.env` file in the project directory and add the following variables:
```env
GROQ_API_KEY=your_groq_api_key
```

### **Installation**
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/scheme-chatbot.git
   cd scheme-chatbot
   ```

2. **Set Up a Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run Elasticsearch:**
   Ensure Elasticsearch is running on `http://localhost:9200`. Adjust the connection details in `main.py` if necessary.

5. **Load Scheme Data:**
   The `Cleaned_Schemes.json` file should be in the project root directory. The script will load this data into Elasticsearch on the first run.

---

## **How to Run**

1. **Initialize the Application:**
   Run the chatbot application:
   ```bash
   python main.py
   ```

2. **Interact with the Chatbot:**
   - Type your queries (e.g., "What are the benefits of scheme X?").
   - To exit the chatbot, type `exit` or `quit`.

---

## **How It Works**
1. **Data Indexing:**
   - The `Cleaned_Schemes.json` file is loaded into Elasticsearch for traditional text-based search.
   - BERT embeddings are generated for each scheme and stored in a FAISS index for semantic search.

2. **Query Handling:**
   - A user query triggers both FAISS and Elasticsearch searches.
   - Results from both searches are combined and scored based on predefined weights.

3. **Response Generation:**
   - The top result is processed using LLaMA, which generates a detailed response.

---

## **Extending the Project**

### Add New Schemes
Update the `Cleaned_Schemes.json` file with additional schemes, then restart the application to re-index the data.

### Customize Search Weights
Modify the `combine_results` function in `main.py` to adjust the scoring logic for Elasticsearch and FAISS results.

### Use a Different LLM
Replace the LLaMA integration with another model by modifying the `generate_llama_response` function.

---

## **Known Issues**
- Ensure Elasticsearch is properly configured and running; otherwise, search functionality will fail.
- BERT embedding generation might be slow for large datasets. Consider optimizing preprocessing or using a GPU.

## Author

Develped by Ashik

---

