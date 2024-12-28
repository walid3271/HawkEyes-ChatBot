# import streamlit as st
# import os
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import google.generativeai as genai
# from langchain.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv

# load_dotenv()
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# def get_conversational_chain():
#     prompt_template = """
#     Answer the question as detailed as possible from the provided context.\n\n
#     Context:\n{context}\n
#     Question:\n{question}\n
#     Answer:
#     """

#     model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.9)
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
#     return chain

# def user_input(user_question):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#     docs = new_db.similarity_search(user_question)

#     chain = get_conversational_chain()
#     response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
#     st.write("", response["output_text"])

# def main():
#     st.set_page_config(page_title="HawkEyes")
#     st.header("ChatBot For HawkEyes")

#     user_question = st.text_input("Ask Questions About HawkEyes")

#     if user_question:
#         user_input(user_question)

# if __name__ == "__main__":
#     main()


# # streamlit run he_test.py





import streamlit as st
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to get the conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context.\n\n
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.9)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to initialize FAISS index if it doesn't exist
def initialize_faiss_index(embeddings):
    if not os.path.exists("faiss_index/index.faiss"):
        st.warning("FAISS index not found. Initializing...")
        
        # Replace this with actual data to populate the index
        example_data = [
            Document(page_content="HawkEyes is an AI platform for monitoring and insights."),
            Document(page_content="It offers real-time analytics and enhanced surveillance."),
            Document(page_content="HawkEyes leverages advanced AI models for predictive analytics.")
        ]

        # Create FAISS index
        new_db = FAISS.from_documents(example_data, embeddings)
        if not os.path.exists("faiss_index"):
            os.makedirs("faiss_index")
        new_db.save_local("faiss_index")
        st.success("FAISS index created successfully.")
    else:
        st.info("FAISS index found. Loading...")

# Function to handle user input and provide response
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    initialize_faiss_index(embeddings)
    try:
        # Load FAISS index
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        # Get response from conversational chain
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        st.write(response["output_text"])
    except Exception as e:
        st.error(f"Error processing your request: {e}")

# Main function for Streamlit app
def main():
    st.set_page_config(page_title="HawkEyes")
    st.header("ChatBot For HawkEyes")

    user_question = st.text_input("Ask Questions About HawkEyes")
    if user_question:
        with st.spinner("Processing your question..."):
            user_input(user_question)

# Run the Streamlit app
if __name__ == "__main__":
    main()
