from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os
import streamlit as st

def main():
    # If running on Streamlit Sharing, use the secrets management system
    if st.secrets:
        openai_key = st.secrets["openai"]["key"]
    # If running locally, load from .env file
    else:
        from dotenv import load_dotenv
        load_dotenv()
        openai_key = os.getenv("OPENAI_KEY")

    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF")

    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    txt = ""  # Initialize the txt variable
    
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        num_pages = len(pdf_reader.pages)
        
        for page in pdf_reader.pages:
            txt += page.extract_text()

        print("Text extracted from PDF:", txt)

        # Split text into chunks using langchain
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(txt)

        print("Text split into chunks:", chunks)

        # Create Embeddings for similarity searches
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        print("Knowledge base created")

        # Show answers from user inputs
        user_question = st.text_input("Ask a question about the PDF")
        if user_question:
            print("User question:", user_question)

            docs = knowledge_base.similarity_search(user_question)

            print("Similar documents found:", docs)

            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=user_question)

            print("Response:", response)
            
            st.write(response)

if __name__ == '__main__':
    main()
