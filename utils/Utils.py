import os
import pysnc
import sqlite3
import streamlit as st
import pandas as pd
from langchain.agents import initialize_agent, AgentType
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.agent_toolkits import create_sql_agent, JiraToolkit
from langchain_community.document_loaders import ConfluenceLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.utilities.jira import JiraAPIWrapper
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def handle_question(sql_agent, chain, jira_agent, prompt):
    st.chat_message('user').write(prompt)

    with st.spinner('Loading...'):
        response = chain(prompt)
    st.write("From Confluence:\n\n")
    st.chat_message('assistant').write(response)

    with st.spinner('Loading...'):
        try:
            response = jira_agent.run(prompt)
        except Exception:
            response = "I don't know."
    st.write("From JIRA:\n\n")
    st.chat_message('assistant').write(response)

    with st.spinner('Loading...'):
        response = sql_agent.run(prompt)
    st.write("From ServiceNOW/CMDB:\n\n")
    st.chat_message('assistant').write(response)


def provision():
    client = pysnc.ServiceNowClient('https://cgigroupincdemo15.service-now.com', ('api_user', os.getenv('snow_pass')))
    table_list = ['task','incident','sys_user','sys_user_group','core_company','cmn_location','cmn_cost_center','cmn_department',
                  'problem','kb_knowledge','change_request','change_task','std_change_producer_version',
                  'cmdb','cmdb_ci','cmdb_rel_ci','cmdb_ci_computer','cmdb_ci_database','cmdb_ci_service',
                  'cmdb_ci_storage_device','cmdb_class_info','alm_asset','cmdb_model']
    #'incident_task',
    #'change_request_template','change_collision',
    #cmdb_ci_network_host,
    #cmdb_ci_cloud_service_account
    #cmdb_ci_network_adapter
    #cmdb_ci_application_software
    
    conn = sqlite3.connect("glide.db")
    if os.path.exists('glide.db'):
        if os.path.getsize('glide.db') == 0:
            for table_name in table_list:
                gr = client.GlideRecord(table_name)
                gr.query()
                df = pd.DataFrame(gr.to_pandas())
                df.to_sql(table_name, conn, if_exists='replace')

    llm = ChatOpenAI(model='gpt-3.5-turbo-1106', temperature=0)
    db = SQLDatabase.from_uri("sqlite:///glide.db")

    sql_agent = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=False)

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

    jira = JiraAPIWrapper()
    toolkit = JiraToolkit.from_jira_api_wrapper(jira)
    jira_agent = initialize_agent(
        toolkit.get_tools(), llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )

    return chain, sql_agent, jira_agent
