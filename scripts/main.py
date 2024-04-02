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

class Chatbot:

    def __init__(self):
        self.conversation_history = []
        self.llm = ChatGoogleGenerativeAI(model="gemini-pro")
        self.chroma = Chroma(persist_directory="../raw_data/chroma_db", embedding_function=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"))
        self.retriever = self.chroma.as_retriever()

    def ask(self, question):
        def generate_response(question,conversation_history,llm,retriever):

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
        response = generate_response(question, self.conversation_history,self.llm,self.retriever)

        # Add response to conversation history
        self.conversation_history.append(response)

        return response

    def reset_history(self):
        self.conversation_history = []


def interact_with_chatbot():
    # Initialize chatbot
    chatbot = Chatbot()
    print("Chatbot: Hi! What kind of movie do you want to watch today?")

    while True:
        user_input = str(input("You: "))

        #Check for a new conversation
        if user_input.lower() == 'reset':
            chatbot.reset_history()
            print('Chatbot restarted')
            print("If you wish to exit type exit")
            user_input = str(input("You: "))

        # Check for exit command
        if user_input.lower() == 'exit':
            print("Exiting chatbot...")
            break

        # Invoke the ask method with provided input and conversation history
        response = chatbot.ask(user_input)
        # Print the answer to the user's input
        print("Chatbot:", response)
        print("If you wish to exit type exit, if you wish to talk about other movies type reset")


interact_with_chatbot()
