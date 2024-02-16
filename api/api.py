import os
import sqlite3
import pysnc
import pandas as pd
from flask import Flask, request, jsonify
from langchain.agents import AgentType
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.document_loaders import ConfluenceLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.vectorstores.chroma import Chroma
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

app = Flask(__name__)


def provision_glide(record_type):
    client = pysnc.ServiceNowClient('https://cgigroupincdemo15.service-now.com', ('api_user', os.getenv('snow_pass')))
    gr = client.GlideRecord(record_type)
    gr.add_query('active', 'true')
    gr.query()
    df = pd.DataFrame(gr.to_pandas())

    llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo-0125')
    return create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
    )


@app.route('/handle_glide_question', methods=['POST'])
def api_glide_question():
    request_json = request.get_json()
    record_type = request_json['record_type']
    prompt = request_json['prompt']
    agent = provision_glide(record_type)
    response = agent.invoke(
        {'input': prompt}
    )
    return jsonify(response['output'])


@app.route('/handle_confluence_question', methods=['POST'])
def api_confluence_question():
    request_json = request.get_json()
    question = request_json['prompt']
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
    return response


@app.route('/handle_question', methods=['POST'])
def api_question():
    request_json = request.get_json()
    question = request_json['prompt']
    client = pysnc.ServiceNowClient('https://cgigroupincdemo15.service-now.com', ('api_user', os.getenv('snow_pass')))
    table_list = ['task', 'incident', 'sys_user', 'sys_user_group', 'core_company', 'cmn_location', 'cmn_cost_center',
                  'cmn_department',
                  'problem', 'kb_knowledge', 'change_request', 'change_task', 'std_change_producer_version',
                  'cmdb', 'cmdb_ci', 'cmdb_rel_ci', 'cmdb_ci_computer', 'cmdb_ci_database', 'cmdb_ci_service',
                  'cmdb_ci_storage_device', 'cmdb_class_info', 'alm_asset', 'cmdb_model']
    # 'incident_task',
    # 'change_request_template','change_collision',
    # cmdb_ci_network_host,
    # cmdb_ci_cloud_service_account
    # cmdb_ci_network_adapter
    # cmdb_ci_application_software

    conn = sqlite3.connect("glide.db")
    if not os.path.exists("glide.db"):
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

    response = {
        "confluence": chain(question),
        "glide": agent.invoke({'input': question})
    }

    return jsonify(response)
