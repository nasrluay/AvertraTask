import streamlit as st
import boto3
import os
import uuid
from langchain_community.embeddings import BedrockEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

# AWS Configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
bedrock_client = boto3.client(service_name="bedrock-runtime", region_name=AWS_REGION)
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)
s3_client = boto3.client("s3", region_name=AWS_REGION)
BUCKET_NAME = os.getenv("BUCKET_NAME")
folder_path = "/tmp/"

def get_unique_id():
    """Generate a unique identifier."""
    return str(uuid.uuid4())

def split_text(pages, chunk_size=1000, chunk_overlap=200):
    """Split the text into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(pages)

def create_vector_store(request_id, documents):
    """Create and upload a vector store to S3 for admin use."""
    try:
        vectorstore_faiss = FAISS.from_documents(documents, bedrock_embeddings)
        file_name = f"{request_id}.bin"
        vectorstore_faiss.save_local(index_name=file_name, folder_path=folder_path)
        s3_client.upload_file(Filename=f"{folder_path}/{file_name}.faiss", Bucket=BUCKET_NAME, Key=f"{request_id}.faiss")
        s3_client.upload_file(Filename=f"{folder_path}/{file_name}.pkl", Bucket=BUCKET_NAME, Key=f"{request_id}.pkl")
        return True
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return False

def admin_interface():
    """Admin interface to manage document processing and vector store management."""
    st.title("Admin Dashboard for Document Management")

    uploaded_file = st.file_uploader("Upload a document for processing", type=["pdf", "txt"])
    if uploaded_file is not None:
        try:
            request_id = get_unique_id()
            st.write(f"Request ID: {request_id}")
            file_extension = "pdf" if uploaded_file.type == "application/pdf" else "txt"
            saved_file_name = f"{request_id}.{file_extension}"
            
            with open(saved_file_name, "wb") as file:
                file.write(uploaded_file.getvalue())

            if file_extension == "pdf":
                loader = PyPDFLoader(saved_file_name)
                pages = loader.load_and_split()
            else:
                with open(saved_file_name, "r") as file:
                    pages = file.read().split('\n\n')

            st.write(f"Total Pages: {len(pages)}")
            splitted_docs = split_text(pages)
            st.write(f"Splitted Docs length: {len(splitted_docs)}")

            if st.button("Create Vector Store"):
                result = create_vector_store(request_id, splitted_docs)
                if result:
                    st.success("Vector store created and uploaded successfully.")
                else:
                    st.error("Failed to create vector store. Check logs for details.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    admin_interface()


#####################################