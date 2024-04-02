import streamlit as st
import os
from fastapi import FastAPI, Form
import uvicorn

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

class Chatbot:
    def __init__(self):
        self.conversation_history = []
        self.llm = ChatGoogleGenerativeAI(model="gemini-pro")
        self.chroma_db_path = os.path.join(os.path.dirname(__file__), "../raw_data/chroma_db")
        self.chroma = Chroma(persist_directory= self.chroma_db_path, embedding_function=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"))
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
