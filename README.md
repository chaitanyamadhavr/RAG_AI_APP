# InsightBot: Your Personal AI Insights Hub

Video Link: 

Documentation Name: 

## What is InsightBot?
InsightBot 🤖 is a specialized chatbot designed to provide personalized insights about yourself. Leveraging a user's natural language queries, it delivers precise responses on your qualifications, experiences, and achievements, helping you navigate through your personal data with ease.

### Instructions on How to Setup and Run
### Step 1: Install Required Python Libraries
Install the necessary libraries from the requirements.txt file
```
pip install -r requirements.txt
```

NOTE: If there is issue with Pysqlite3 installation, follow below steps

For pysqlite3 installation keep the wheel file (available in Github repo) in your project directory and run below command
```
pip install pysqlite3_wheels-0.5.0-cp310-cp310-win_amd64
```
### Step 2: Generate and Store Embeddings
There are two Python files: embeddings_generator.py and WaLL-E.py.

Set Up API Keys: Ensure your HuggingFace API Token is in the .env file
```
HUGGINGFACEHUB_API_TOKEN = "<HUGGINGFACEHUB_API_TOKEN>"
```
Generate Embeddings: Run vector_embeddings.py to process the fashion data PDF and store the results in the Chroma Vector Database in the "data" directory.
```
python vector_embeddings.py
```
### Step 3: Launch InsightBot
After setting up the embeddings, launch the InsightBot interface by running:
```
streamlit run InsightBot.py
```
