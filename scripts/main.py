import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import google.generativeai as genai

# Set up Google API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize the Chat model
llm = ChatGoogleGenerativeAI(model="gemini-pro")

# Initialize the Chroma vector store
chroma = Chroma(persist_directory="../raw_data/chroma_db", embedding_function=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"))

# Convert Chroma to a retriever
retriever = chroma.as_retriever()

# Define the template for prompts
template = """
You are a helpful AI assistant. That knows everything about movies.
Answer based on the context provided.
context: {context}
input: {input}
answer:
"""

# Create a PromptTemplate from the template
prompt = PromptTemplate.from_template(template)

# Create a chain for combining documents
combine_docs_chain = create_stuff_documents_chain(llm, prompt)

# Create a retrieval chain
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

def interact_with_chatbot():
    print("Chatbot: Hi! What kind of movie do you want to watch today?")
    while True:
        user_input = input("You: ")

        # Provide some context related to movies


        # Invoke the retrieval chain with provided context and input
        response = retrieval_chain.invoke({"input": user_input})

        # Print the answer to the user's input
        print("Chatbot:", response["answer"])

        # Check for exit command
        if user_input.lower() == 'exit':
            print("Exiting chatbot...")
            break

if __name__ == "__main__":
    interact_with_chatbot()
