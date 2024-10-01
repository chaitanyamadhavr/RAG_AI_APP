from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document  # Import Document class
import os
from dotenv import load_dotenv
from collections import OrderedDict

# Load environment variables from .env file
load_dotenv()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Define the directory containing text files
data_folder = "data"

# Ensure the directory exists
if not os.path.exists(data_folder):
    raise FileNotFoundError(f"The folder '{data_folder}' does not exist.")

# Read all .txt files from the data folder
text_content = ""
for file_name in os.listdir(data_folder):
    if file_name.endswith(".txt"):
        txt_file_path = os.path.join(data_folder, file_name)
        with open(txt_file_path, 'r', encoding='utf-8') as file:
            text_content += file.read() + "\n"  # Append content from each file

# Split the text using a text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
texts = text_splitter.split_text(text_content)

# Convert text chunks into Document objects
documents = [Document(page_content=text) for text in texts]

# Initialize the embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Convert texts to embeddings
try:
    embeddings = embedding_model.embed_documents([doc.page_content for doc in documents])
    print("Vector Embeddings created successfully")
except Exception as e:
    print(f"Error creating vector embeddings: {e}")

# Initialize Chroma vector store
vector_store = Chroma(embedding_function=embedding_model, persist_directory="vector_db")

# Add documents to the vector store
try:
    vector_store.add_documents(documents=documents)
except Exception as e:
    print(f"Error adding documents to vector store: {e}")

# Validate the setup
try:
    # Test query to validate data retrieval
    test_query = "What is the candidate name?"
    results = vector_store.search(query=test_query, search_type='similarity')

    # Deduplicate results
    unique_results = OrderedDict()
    for doc in results:
        if doc.page_content not in unique_results:
            unique_results[doc.page_content] = doc

    # Convert unique results to a list and limit to top 3
    final_results = list(unique_results.values())[:3]
    print(f"Unique query results: {final_results}")
except Exception as e:
    print(f"Error during test query: {e}")
