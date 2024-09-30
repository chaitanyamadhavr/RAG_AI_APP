import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import warnings, os
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
import json
from dotenv import load_dotenv
warnings.filterwarnings("ignore")
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

working_dir = os.path.dirname(os.path.abspath(__file__))
config_data = json.load(open(f"{working_dir}/config.json"))
GROQ_API_KEY = config_data["GROQ_API_KEY"]
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Load environment variables from .env file
load_dotenv()

data_directory = os.path.join(os.path.dirname(__file__), "vector_db")

os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Load the vector store from disk
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma(embedding_function=embedding_model, persist_directory=data_directory)

# Initialize the Hugging Face Hub LLM
hf_hub_llm = HuggingFaceHub(
     repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    # repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    model_kwargs={"temperature": 1, "max_new_tokens":1024},
)

prompt_template = """
As an AI assistant specializing in personal information retrieval, your role is to accurately interpret queries about the user's resume, academic achievements (including their 2nd-puc scorecard), and provide responses based on the stored personal data. Follow these directives to ensure accurate and relevant interactions:

1. Responses are limited to the specifics of the user’s academic achievements and professional experience.
2. If a question does not align with the stored personal data, inform the user politely and ask them to refine their question.
3. Each answer must be unique and avoid repetition within a single interaction.

Personal Information Query:
{context}

Question: {question}

Answer:
"""

custom_prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

rag_chain = RetrievalQA.from_chain_type(
    llm=hf_hub_llm, 
    chain_type="stuff", 
    retriever=vector_store.as_retriever(top_k=3),  # retriever is set to fetch top 3 results
    chain_type_kwargs={"prompt": custom_prompt})

def get_response(question):
    result = rag_chain({"query": question})
    response_text = result["result"]
    answer_start = response_text.find("Answer:") + len("Answer:")
    answer = response_text[answer_start:].strip()
    return answer

# Streamlit app
# Remove whitespace from the top of the page and sidebar
st.markdown(
        """
            <style>
                .appview-container .main .block-container {{
                    padding-top: {padding_top}rem;
                    padding-bottom: {padding_bottom}rem;
                    }}

            </style>""".format(
            padding_top=1, padding_bottom=1
        ),
        unsafe_allow_html=True,
    )

# st.header("### Discover the AI styling recommendations :dress:", divider='grey')
st.markdown("""
    <h3 style='text-align: left; color: black; padding-top: 35px; border-bottom: 3px solid red;'>
        Personal Insights Hub
    </h3>""", unsafe_allow_html=True)


side_bar_message = """
Hi! 👋 I'm here to assist you with your personal information and academic achievements. What would you like to know or explore?
\nHere are some areas you might be interested in:
1. **Academic Qualifications** 🎓 (e.g., 12th-grade score)
2. **Professional Experience** 💼 (e.g., roles, skills)
3. **Certifications and Achievements** 🏅
4. **Projects and Publications** 📄

Feel free to ask me anything about your personal data!
"""

with st.sidebar:
    st.title('🤖InsightBot: Your Personal Information Assistant')
    st.markdown(side_bar_message)

initial_message = """
Hi there! I'm your InsightBot 🤖 
Here are some questions you might ask me:\n
 🎓 What are my key academic qualifications?\n
 🎓 Can you summarize my professional experience?\n
 🎓 What certifications do I have?\n
 🎓 How can I highlight my projects effectively?\n
 🎓 What achievements should I focus on for my resume?
"""

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": initial_message}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": initial_message}]
st.button('Clear Chat', on_click=clear_chat_history)

# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Hold on, I'm fetching the latest information for you..."):
            response = get_response(prompt)
            placeholder = st.empty()
            full_response = response  # Directly use the response
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
