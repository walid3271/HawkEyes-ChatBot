import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
import bs4
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_text():
    urls = [
        "https://www.hedigital.tech/about/"
    ]

    all_text_documents = []
    for url in urls:
        loader = WebBaseLoader(web_paths=(url,), bs_kwargs=dict(parse_only=bs4.SoupStrainer(
                class_=("bg-[#181818] p-6 rounded-xl shadow-md",
                        "container mx-auto my-2 xl:px-14 py-10",
                        "container",
                        "mx-auto my-10",
                        "bg-primary py-20 mx-auto",
                        "footer md:flex md:justify-center md:items-center text-[#ffffff]")
            ))
        )
        text_documents = loader.load()
        all_text_documents.extend(text_documents)
    
    return all_text_documents

def get_text_chunks():
    text_documents = get_text()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    documents = text_splitter.split_documents(text_documents)
    texts = [doc.page_content for doc in documents]
    return texts

def get_vector_store():
    text_chunks = get_text_chunks()
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the context, say "answer is not available in the context" and do not provide an incorrect answer.\n\n
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    st.write("", response["output_text"])

def main():
    # get_vector_store()
    st.set_page_config(page_title="HawkEyes")
    st.header("ChatBot For HawkEyes")

    user_question = st.text_input("Ask Questions About HawkEyes")

    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    main()


# streamlit run he.py
