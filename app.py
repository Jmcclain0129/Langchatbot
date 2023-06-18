import os
import sys
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.embeddings import TensorflowHubEmbeddings

# Add these lines here
try:
    import aiohttp
    import dataclasses_json
    import numexpr
    import numpy
    import openapi_schema_pydantic
    import pydantic
    import yaml  # This is for PyYAML
    import requests
    import sqlalchemy
    import tenacity
    print("All packages imported successfully.")
except ImportError as e:
    print(f"Failed to import a package: {e}")


print("Python executable:", sys.executable)

def main():
    # If running on Streamlit Sharing, use the secrets management system
    if st.secrets:
        os.environ['OPENAI_API_KEY'] = st.secrets["openai"]["key"]
        print(os.environ['OPENAI_API_KEY'])  # Debugging line
    # If running locally, load from .env file
    else:
        from dotenv import load_dotenv
        load_dotenv()
        os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_KEY")
        print(os.environ['OPENAI_API_KEY'])  # Debugging line



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
