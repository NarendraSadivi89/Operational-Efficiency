import os
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ConfluenceLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def handle_confluence(question):
    st.write(question)
    # These environment variables are set in .env file
    loader = ConfluenceLoader(
        url=os.getenv('confluence_url'),
        username=os.getenv('confluence_email'),
        api_key=os.getenv('confluence_api_key')
    )

    documents = loader.load(space_key='KB', include_attachments=False, limit=50)

    tf = Html2TextTransformer()
    fd = tf.transform_documents(documents)
    ts = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    splits = ts.split_documents(fd)

    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)
    embeddings = OpenAIEmbeddings()
    knowledge_base = Chroma.from_documents(
        documents=splits,
        embedding=embeddings
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=knowledge_base.as_retriever()
    )

    response = chain(question)
    st.write(response['result'])
