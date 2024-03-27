#import os
#import google.generativeai as genai

import os
#from dotenv import load_dotenv
import csv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.vectorstores import Chroma

# Load environment variables from .env file
#load_dotenv()

# Get Google API key from environment variables
google_api_key = os.getenv("GOOGLE_API_KEY")

# Load the models
llm = ChatGoogleGenerativeAI(model="gemini-pro", api_key=google_api_key)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Load CSV and create chunks
csv_file = "raw_data/movies.csv"
text_chunks = []
with open(csv_file, 'r', encoding='utf-8') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        text_chunks.extend(row)

# Turn the chunks into embeddings and store them in Chroma
vectordb = Chroma.from_documents(text_chunks, embeddings)

# Configure Chroma as a retriever with top_k=5
retriever = vectordb.as_retriever(search_kwargs={"k": 5})

# Create the retrieval chain
template = """
You are a helpful AI assistant.
Answer based on the context provided.
context: {context}
input: {input}
answer:
"""
prompt = PromptTemplate.from_template(template)
combine_docs_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

# Invoke the retrieval chain
response = retrieval_chain.invoke({"input":"How do I apply for personal leave?"})

# Print the answer to the question
print(response["answer"])
