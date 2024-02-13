import os
import pysnc
import sqlite3
import streamlit as st
import pandas as pd
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.document_loaders import ConfluenceLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def handle_question(agent, chain, prompt):
    st.chat_message('user').write(prompt)
    response = chain(prompt)
    st.write(f"From Confluence:\n\n{response['result']}")
    st.write("From ServiceNOW/CMDB:\n\n")
    with st.chat_message('assistant'):
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent.invoke(
            {'input': prompt}, {'callbacks': [st_callback]}
        )
        st.write(response['output'])


def provision():
    client = pysnc.ServiceNowClient('https://cgigroupincdemo15.service-now.com', ('api_user', os.getenv('snow_pass')))
    table_list = [
        'incident',
        'change_request',
        'change_task'
    ]
    conn = sqlite3.connect("glide.db")
    for table_name in table_list:
        gr = client.GlideRecord(table_name)
        gr.query()
        df = pd.DataFrame(gr.to_pandas())
        df.to_sql(table_name, conn, if_exists='replace')

    llm = ChatOpenAI(model='gpt-3.5-turbo-1106', temperature=0)
    db = SQLDatabase.from_uri("sqlite:///glide.db")

    agent = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)

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

    return chain, agent
