import boto3
import streamlit as st
import os
import uuid
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

# AWS and Bedrock configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
bedrock_client = boto3.client(service_name="bedrock-runtime", region_name=AWS_REGION)
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)
s3_client = boto3.client("s3", region_name=AWS_REGION)
BUCKET_NAME = os.getenv("BUCKET_NAME")
folder_path = "/tmp/"

def get_unique_id():
    """Generate a unique identifier."""
    return str(uuid.uuid4())


def load_index():
    """Download vector store files from S3 to local."""
    s3_client.download_file(Bucket=BUCKET_NAME, Key="my_faiss.faiss", Filename=f"{folder_path}my_faiss.faiss")
    s3_client.download_file(Bucket=BUCKET_NAME, Key="my_faiss.pkl", Filename=f"{folder_path}my_faiss.pkl")

def get_llm(model_id):
    """Initialize and return a Bedrock Large Language Model."""
    llm = Bedrock(model_id=model_id, client=bedrock_client, model_kwargs={'max_tokens_to_sample': 512})
    return llm

def get_response(llm, vectorstore, question, prompt_style="zero-shot"):
    """Generate a response for a given question using a specified LLM and prompt style."""
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

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Define the appropriate chain type
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    answer = qa({"query": question})
    return answer['result']

def main():
    """Main function to run the Streamlit app for a utility chatbot."""
    st.header("Avertra Utility Chat Bot")
        
    load_index()
    dir_list = os.listdir(folder_path)
    st.write(f"Knowledge Base in: {folder_path}")
    st.write(dir_list)

    faiss_index = FAISS.load_local(
        index_name="my_faiss",
        folder_path=folder_path,
        embeddings=bedrock_embeddings,
        allow_dangerous_deserialization=True
    )

    st.write("INDEX IS READY")
    question = st.text_input("Please ask your question")

    model_options = ["anthropic.claude-v2:1"]
    prompt_styles = ["zero-shot", "few-shot"]

    selected_model = st.selectbox("Choose the model", model_options)
    selected_prompt_style = st.selectbox("Choose the prompt style", prompt_styles)

    if st.button("Ask Question"):
        with st.spinner("Querying..."):
            llm = get_llm(selected_model)
            response = get_response(llm, faiss_index, question, selected_prompt_style)
            st.write(response)
            st.success("Done")

if __name__ == "__main__":
    main()
