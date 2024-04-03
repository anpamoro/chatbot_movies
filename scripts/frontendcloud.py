import streamlit as st
from google.cloud import storage
import os

import json
import tempfile

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import google.generativeai as genai

gcp_type = st.secrets["type"]
gcp_p_id = st.secrets["project_id"]
gcp_pr_id = st.secrets["private_key_id"]
gcp_pr_key = st.secrets["private_key"]
gcp_cl_em = st.secrets["client_email"]
gcp_cl_id = st.secrets["client_id"]
gcp_auth_uri = st.secrets["auth_uri"]
gcp_token_uri = st.secrets["token_uri"]
gcp_auth_provider = st.secrets["auth_provider_x509_cert_url"]
gcp_client_x509_cert_url = st.secrets["client_x509_cert_url"]
gcp_universe_domain = st.secrets["universe_domain"]

#json credentials
json_credentials = {"type" : gcp_type,
                    "project_id" : gcp_p_id,
                    "private_key_id" : gcp_pr_id,
                    "private_key" : gcp_pr_key,
                    "client_email" : gcp_cl_em,
                    "client_id" : gcp_auth_uri,
                    "auth_uri" : gcp_token_uri,
                    "token_uri" : gcp_auth_uri,
                    "auth_provider_x509_cert_url" : gcp_auth_provider,
                    "client_x509_cert_url" : gcp_client_x509_cert_url,
                    "universe_domain" : gcp_universe_domain,
}

# Create a temporary file to store the JSON data
with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
    # Write the JSON data to the temporary file
    json.dump(json_credentials, temp_file)
    temp_file_path = temp_file.name

# Create the client using the temporary file path
storage_client = storage.Client.from_service_account_json(temp_file_path)

# Define function to download chroma_db directory from GCS
def download_chroma_db():
    bucket_name = "chatbot_movies"
    source_blob_name = "chroma_db"  # Update this to the directory path in GCS
    destination_folder = "./chroma_db"  # Local destination folder

    # Ensure the local destination folder exists
    os.makedirs(destination_folder, exist_ok=True)

    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=source_blob_name)  # List all blobs with the specified prefix

    # Download each blob to the local destination folder
    for blob in blobs:
        if blob.name.endswith('/'):  # Skip directories
            continue
        destination_file_path = os.path.join(destination_folder, os.path.relpath(blob.name, source_blob_name))
        os.makedirs(os.path.dirname(destination_file_path), exist_ok=True)  # Ensure directory exists
        blob.download_to_filename(destination_file_path)

# Call function to download chroma_db directory
download_chroma_db()

# Streamlit code ........
# Set up Google API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

class Chatbot:
    def __init__(self):
        self.conversation_history = []
        self.llm = ChatGoogleGenerativeAI(model="gemini-pro")
        self.chroma = Chroma(persist_directory= "./chroma_db", embedding_function=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"))
        self.retriever = self.chroma.as_retriever()

    def ask(self, question):
        def generate_response(question, conversation_history, llm, retriever):
            context = "\n".join(conversation_history)
            template = """
            You are a helpful AI assistant.
            Answer based on the context provided.
            context: {context}
            input: {question}
            answer:
            """
            prompt = PromptTemplate.from_template(template)
            combine_docs_chain = create_stuff_documents_chain(llm, prompt)
            retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

            #Invoke the retrieval chain
            response = retrieval_chain.invoke({'input': context, 'question': question})

            return response["answer"]

        # Add question to conversation history
        self.conversation_history.append(question)

        # Generate response based on current question and conversation history
        response = generate_response(question, self.conversation_history, self.llm, self.retriever)

        # Add response to conversation history
        self.conversation_history.append(response)

        return response

    def reset_history(self):
        self.conversation_history = []

def interact_with_chatbot():
    # Initialize chatbot
    chatbot = Chatbot()

    # Streamlit UI
    st.title("Chatbot Demo")

    st.write("Chatbot: Hi! What kind of movie do you want to watch today?")

    # Text input for user input
    user_input = st.text_input("You:")

    # Button to send user input
    if st.button("Send"):
        # Check for reset or exit commands
        if user_input.lower() == 'reset':
            chatbot.reset_history()
            st.write('Chatbot restarted')
            st.write("If you wish to exit type exit")
        elif user_input.lower() == 'exit':
            st.write("Exiting chatbot...")
        else:
            # Invoke the ask method with provided input
            response = chatbot.ask(user_input)
            st.write("Chatbot:", response)
            st.write("If you wish to exit type exit, if you wish to talk about other movies type reset")

if __name__ == "__main__":
    interact_with_chatbot()
