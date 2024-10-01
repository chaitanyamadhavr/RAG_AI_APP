# InsightBot: Your Personal AI Insights Hub

Documentation Name: RAG_Documentation.pdf

## What is InsightBot?
InsightBot 🤖 is a specialized RAG chatbot designed to provide personalized insights about yourself. Leveraging a user's natural language queries, it delivers precise responses on your qualifications, experiences, and achievements, helping you navigate through your personal data with ease.

### Instructions on How to Setup and Run
### Step 1: Place your txt/pdf files in "data" directory.

### Step 2: Install Required Python Libraries
Install the necessary libraries from the requirements.txt file
```
pip install -r requirements.txt
```

NOTE: If there is issue with Pysqlite3 installation, follow below steps

For pysqlite3 installation keep the wheel file (available in Github repo) in your project directory and run below command
```
pip install pysqlite3_wheels-0.5.0-cp310-cp310-win_amd64.whl
```
### Step 3: Generate and Store Embeddings

Set Up API Keys: Ensure your HuggingFace API Token is in the .env file (Create a seperate file called ".env" and copy paste the content from env.txt given in repository)
```
HUGGINGFACEHUB_API_TOKEN = "<HUGGINGFACEHUB_API_TOKEN>"
```
Generate Embeddings: Run vector_embeddings.py to process the personal data PDF/txt and store the results in the Chroma Vector Database in the "vector_db" directory.
```
python vector_embeddings.py
```
### Step 4: Launch InsightBot
Setup Groq API Key in config.json file:
```
Groq_API_TOKEN = "<Groq_API_TOKEN>"
```
After setting up the embeddings, launch the InsightBot interface by running:
```
streamlit run InsightBot.py
```
