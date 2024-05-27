import boto3  # AWS SDK for Python
import streamlit as st  # Streamlit for web app
import os  # For environment variables
import uuid  # For unique identifier generation
from langchain_community.embeddings import BedrockEmbeddings  # Embeddings from LangChain community
from langchain_community.llms import Bedrock  # Large Language Model from LangChain community
from langchain.prompts import PromptTemplate  # Template for creating prompts
from langchain.chains import RetrievalQA  # Chain for retrieval-based QA
from langchain_community.vectorstores import FAISS  # Vector store implementation
from datetime import datetime  # For handling date and time

# AWS and Bedrock configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")  # Get AWS region from environment variable or default to 'us-east-1'
bedrock_client = boto3.client(service_name="bedrock-runtime", region_name=AWS_REGION)  # Initialize Bedrock client
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)  # Initialize Bedrock embeddings
s3_client = boto3.client("s3", region_name=AWS_REGION)  # Initialize S3 client
BUCKET_NAME = os.getenv("BUCKET_NAME")  # Get S3 bucket name from environment variable
folder_path = "/tmp/"  # Local folder path for temporary file storage (AWS)

def load_latest_faiss():
    """Load the latest FAISS file from the specified S3 bucket."""
    print(f"{BUCKET_NAME}")
    response = s3_client.list_objects_v2(Bucket=BUCKET_NAME)  # List objects in the specified bucket
    latest_object = None  # Initialize a variable to keep track of the latest object

    # Iterate through the objects and find the latest one
    for obj in response.get('Contents', []):
        if latest_object is None or obj['LastModified'] > latest_object['LastModified']:
            latest_object = obj
    if latest_object:
        return latest_object['Key']  # Return the key of the latest object
    else:
        return None  # Return None if no objects found

def load_index(FAISS_file_name):
    """Download vector store files from S3 to local."""
    print(FAISS_file_name)
    # Download the FAISS and associated metadata files from S3 to local folder
    s3_client.download_file(Bucket=BUCKET_NAME, Key=f"{FAISS_file_name}.faiss", Filename=f"{folder_path}{FAISS_file_name}.faiss")
    s3_client.download_file(Bucket=BUCKET_NAME, Key=f"{FAISS_file_name}.pkl", Filename=f"{folder_path}{FAISS_file_name}.pkl")

def get_llm(model_id):
    """Initialize and return a Bedrock Large Language Model."""
    # Create an instance of the Bedrock LLM with specified model ID and parameters
    llm = Bedrock(model_id=model_id, client=bedrock_client, model_kwargs={'max_tokens_to_sample': 512})
    return llm

def get_response(llm, vectorstore, question, prompt_style="zero-shot"):
    """Generate a response for a given question using a specified LLM and prompt style."""
    # Define the prompt template based on the selected style
    prompt_template = {
        "zero-shot": """
        Human: Provide a concise answer to the question using the given context.
        Context: {context}
        Question: {question}
        Assistant:""",
        "few-shot": """
        Human: Given the context, provide a detailed answer to the question. Use examples if necessary.
        Context: {context}
        Question: {question}
        Assistant:"""
    }[prompt_style]

    # Create a prompt template
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    # Initialize the RetrievalQA chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Define the appropriate chain type
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    # Get the response from the QA chain
    answer = qa({"query": question})
    return answer['result']

def main():
    """Main function to run the Streamlit app for a utility chatbot."""
    st.header("Avertra Utility Chat Bot")
    # Load the latest FAISS file from S3
    FAISS_file_name = load_latest_faiss().split('.faiss')[0]
    # Download the index files
    load_index(FAISS_file_name)
    st.write(f"Knowledge Base in: {FAISS_file_name}")
    
    # Load the FAISS index from local files
    faiss_index = FAISS.load_local(
        index_name=f"{FAISS_file_name}",
        folder_path=folder_path,
        embeddings=bedrock_embeddings,
        allow_dangerous_deserialization=True
    )

    st.write("INDEX IS READY")
    # Input field for user question
    question = st.text_input("Please ask your question")

    # Options for selecting model and prompt style
    model_options = ["anthropic.claude-v2:1"]
    prompt_styles = ["zero-shot", "few-shot"]

    selected_model = st.selectbox("Choose the model", model_options)
    selected_prompt_style = st.selectbox("Choose the prompt style", prompt_styles)

    # Button to ask the question
    if st.button("Ask Question"):
        with st.spinner("Querying..."):
            llm = get_llm(selected_model)  # Initialize the LLM
            response = get_response(llm, faiss_index, question, selected_prompt_style)  # Get the response
            st.write(response)  # Display the response
            st.success("Done")

if __name__ == "__main__":
    main()  # Run the main function
