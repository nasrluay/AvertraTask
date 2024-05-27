import streamlit as st  # Streamlit for creating the web interface
import boto3  # AWS SDK for Python
import os  # For environment variables
import uuid  # For generating unique identifiers
from langchain_community.embeddings import BedrockEmbeddings  # Embeddings from LangChain community
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Text splitter from LangChain
from langchain_community.document_loaders import PyPDFLoader  # PDF loader from LangChain community
from langchain_community.vectorstores import FAISS  # Vector store implementation

# AWS Configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")  # Get AWS region from environment variable or default to 'us-east-1'
bedrock_client = boto3.client(service_name="bedrock-runtime", region_name=AWS_REGION)  # Initialize Bedrock client
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)  # Initialize Bedrock embeddings
s3_client = boto3.client("s3", region_name=AWS_REGION)  # Initialize S3 client
BUCKET_NAME = os.getenv("BUCKET_NAME")  # Get S3 bucket name from environment variable
folder_path = "/tmp/"  # Local folder path for temporary file storage (AWS)

def get_unique_id():
    """Generate a unique identifier."""
    return str(uuid.uuid4())

def split_text(pages, chunk_size=1000, chunk_overlap=200):
    """Split the text into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)  # Initialize text splitter
    return text_splitter.split_documents(pages)  # Split the document into chunks

def create_vector_store(request_id, documents):
    """Create and upload a vector store to S3 for admin use."""
    try:
        # Create a FAISS vector store from the documents
        vectorstore_faiss = FAISS.from_documents(documents, bedrock_embeddings)
        file_name = f"{request_id}.bin"  # Generate the file name
        vectorstore_faiss.save_local(index_name=file_name, folder_path=folder_path)  # Save vector store locally
        
        # Upload the FAISS index and metadata files to S3
        s3_client.upload_file(Filename=f"{folder_path}{file_name}.faiss", Bucket=BUCKET_NAME, Key=f"{request_id}.faiss")
        s3_client.upload_file(Filename=f"{folder_path}{file_name}.pkl", Bucket=BUCKET_NAME, Key=f"{request_id}.pkl")
        return True
    except Exception as e:
        st.error(f"Error creating vector store: {e}")  # Display error message
        return False

def get_latest_object_key(bucket_name):
    """Retrieve the key of the latest object in the specified S3 bucket."""
    try:
        s3 = boto3.client('s3')  # Initialize S3 client
        response = s3.list_objects_v2(Bucket=bucket_name)  # List objects in the specified bucket

        latest_object = None  # Initialize a variable to keep track of the latest object
        for obj in response.get('Contents', []):  # Iterate through the objects
            if latest_object is None or obj['LastModified'] > latest_object['LastModified']:
                latest_object = obj  # Update the latest object

        if latest_object:
            return latest_object['Key']  # Return the key of the latest object
        else:
            st.error("Bucket is empty or no objects found.")  # Display error message if no objects found
            return None
    except Exception as e:
        st.error(f"Error retrieving latest object from S3: {e}")  # Display error message
        return None

def admin_interface():
    """Admin interface to manage document processing and vector store management."""
    st.title("Admin Dashboard for Knowledgebase Management")  # Set the title of the web interface

    # Display the latest object key in S3
    latest_key = get_latest_object_key(BUCKET_NAME)
    if latest_key:
        st.write(f"Latest object key in S3: {latest_key}")

    # File uploader for document processing
    uploaded_file = st.file_uploader("Upload a document for processing", type=["pdf", "txt"])
    if uploaded_file is not None:
        try:
            request_id = get_unique_id()  # Generate a unique request ID
            st.write(f"Request ID: {request_id}")
            file_extension = "pdf" if uploaded_file.type == "application/pdf" else "txt"  # Determine file extension
            saved_file_name = f"{request_id}.{file_extension}"  # Generate the saved file name
            
            # Save uploaded file locally
            with open(saved_file_name, "wb") as file:
                file.write(uploaded_file.getvalue())

            # Load and split document based on file type
            if file_extension == "pdf":
                loader = PyPDFLoader(saved_file_name)  # Initialize PDF loader
                pages = loader.load_and_split()  # Load and split PDF document
            else:
                with open(saved_file_name, "r") as file:
                    pages = file.read().split('\n\n')  # Split text document into pages

            st.write(f"Total Pages: {len(pages)}")
            splitted_docs = split_text(pages)  # Split the document into chunks
            st.write(f"Splitted Docs length: {len(splitted_docs)}")

            # Button to create vector store
            if st.button("Create Vector Store"):
                result = create_vector_store(request_id, splitted_docs)  # Create and upload vector store
                if result:
                    st.success("Vector store created and uploaded successfully.")
                else:
                    st.error("Failed to create vector store. Check logs for details.")

                # Update the latest object key in S3 after creating the vector store
                latest_key = get_latest_object_key(BUCKET_NAME)
                if latest_key:
                    st.write(f"Updated latest object key in S3: {latest_key}")

        except Exception as e:
            st.error(f"An error occurred: {e}")  # Display error message

if __name__ == "__main__":
    admin_interface()  # Run the admin interface
