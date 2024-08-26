from transformers import AutoConfig
from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # Use Hugging Face for embeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
import re
import requests

repo_id = "google/flan-t5-large"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def create_db_from_youtube_video_url(video_url: str) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url,add_video_info =True)
    transcript = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=0)
    docs = text_splitter.split_documents(transcript)
    db = FAISS.from_documents(docs,embeddings)
    return db

def validate_api_key(api_key):
    if (api_key == None or api_key == "" ):
        return False
    url = "https://huggingface.co/api/whoami-v2" 
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return True  
    else:
        return False

def get_response_from_query(db, query,hf_access_token,k=4):
    """
    Handles a query by searching for the most similar documents in the vector store and
    generating a response using a language model.
    """
    docs = db.similarity_search(query, k=k)

    # docs_page_content = " ".join([f"Title : {d.metadata['title']} Content : {d.page_content}" for d in docs])
    docs_page_content = " ".join([d.page_content for d in docs])

    #  Cleaning the context to reduce the size
    docs_page_content = text_cleaning(docs_page_content)
    # Initialize the language model from Hugging Face Hub
    llm = HuggingFaceHub(repo_id=repo_id,
                         model_kwargs={"temperature":0.8, 'max_new_tokens' : 250,
                                          },huggingfacehub_api_token=hf_access_token)

    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        You are a knowledgeable assistant. Based on the provided context, answer the following question in as much detail as possible. If the context does not allow for a long answer, provide a concise yet informative response.
        Context: {docs}

        Question: {question}
        """,
    )
    chain = prompt | llm
    # Generate the response
    print(prompt.format(question=query, docs=docs_page_content))
    response = chain.invoke({"question":query , "docs":docs_page_content})
    response = response.replace("\n", "")
    return response, docs

def text_cleaning(text):
    text = re.sub(r'\d+:\d+', '', text)  # Remove timestamps
    text = re.sub(r'\b[A-Z][A-Z0-9]*:\s*', '', text)  # Remove speaker labels
    # Remove URLs and email addresses
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\S+@\S+', '', text)  # Remove email addresses
    # Lowering them for better readability
    text = text.lower()
    
    return text

