import json
import os
import sqlite3
import pysnc
import pandas as pd
import spacy
from atlassian import Confluence
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from langchain.agents import AgentType, initialize_agent
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.agent_toolkits import create_sql_agent, JiraToolkit
from langchain_community.document_loaders import ConfluenceLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.utilities.jira import JiraAPIWrapper
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

app = Flask(__name__)

CONF_LLM = ChatOpenAI(model='gpt-3.5-turbo-0125', temperature=0)
JIRA_LLM = ChatOpenAI(model='gpt-3.5-turbo-0125', temperature=0)
SNOW_LLM = ChatOpenAI(model='gpt-3.5-turbo-0125', temperature=0)

nlp = spacy.load("en_core_web_sm")


def extract_keywords(prompt):
    doc = nlp(prompt)
    keywords = set()

    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"]:
            keywords.add(token.text.lower())

    for ent in doc.ents:
        keywords.add(ent.text.lower())

    return list(keywords)


def provision_snow():
    client = pysnc.ServiceNowClient(os.getenv('snow_url'), (os.getenv('snow_user'), os.getenv('snow_pass')))
    table_list = [
        'task', 'incident', 'sys_user', 'sys_user_group', 'problem', 'wf_workflow',
        'kb_knowledge_base', 'kb_category', 'kb_knowledge',
        'kb_feedback', 'change_request', 'change_task', 'std_change_producer_version',
        'cmdb', 'cmdb_ci', 'cmdb_rel_ci', 'cmdb_ci_computer'
    ]
    # 'incident_task',
    # 'change_request_template', 'change_collision',
    # cmdb_ci_network_host,
    # cmdb_ci_cloud_service_account
    # cmdb_ci_network_adapter
    # cmdb_ci_application_software

    glide_db_string = f"glide_{os.getenv('snow_user')}.db"
    conn = sqlite3.connect(glide_db_string)
    if os.path.getsize(glide_db_string) == 0:
        for table_name in table_list:
            gr = client.GlideRecord(table_name)
            try:
                gr.query()
                if gr.has_next():
                    df = pd.DataFrame(gr.to_pandas())
                    df.to_sql(table_name, conn, if_exists='replace')
            except Exception as e:
                print(e)
    db = SQLDatabase.from_uri(f"sqlite:///{glide_db_string}")

    return create_sql_agent(SNOW_LLM, db=db, agent_type="openai-tools", verbose=True)


def provision_conf():
    if not os.path.exists(f'chroma_db_{os.getenv("confluence_email")}'):
        loader = ConfluenceLoader(
            url=os.getenv('confluence_url'),
            username=os.getenv('confluence_email'),
            api_key=os.getenv('confluence_api_key'),
            cloud=True
        )

        confluence = Confluence(
            url=os.getenv('confluence_url'),
            username=os.getenv('confluence_email'),
            password=os.getenv('confluence_api_key'),
            cloud=True)

        space_keys = [obj['key'] for obj in confluence.get_all_spaces(start=0, limit=500, expand=None)['results']]

        all_documents = []
        for space_key in space_keys:
            documents = loader.load(space_key=space_key, include_attachments=False, limit=50)
            all_documents.extend(documents)

        tf = Html2TextTransformer()
        fd = tf.transform_documents(all_documents)
        ts = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
        splits = ts.split_documents(fd)

        embeddings = OpenAIEmbeddings()
        knowledge_base = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=f'chroma_db_{os.getenv("confluence_email")}'
        )
    else:
        embeddings = OpenAIEmbeddings()
        knowledge_base = Chroma(
            embedding_function=embeddings,
            persist_directory=f'chroma_db_{os.getenv("confluence_email")}'
        )

    return RetrievalQA.from_chain_type(
        llm=CONF_LLM,
        chain_type='stuff',
        retriever=knowledge_base.as_retriever(),
        return_source_documents=True
    )


def provision_jira():
    jira = JiraAPIWrapper()
    toolkit = JiraToolkit.from_jira_api_wrapper(jira)
    return initialize_agent(
        toolkit.get_tools(), JIRA_LLM, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )


@app.route('/handle_question', methods=['POST'])
def api_question():
    load_dotenv()
    request_json = request.get_json()
    prompt = request_json["prompt"]
    keywords = extract_keywords(prompt)
    seek_list = request_json["seek_list"]

    response = {}
    if "confluence" in seek_list:
        confluence_chain = provision_conf()
        confluence_response = confluence_chain(
            f"""You are a confluence chat bot.
                        Give me an accurate summary of information based on the prompt below.
                        If you can't find a solution or don't have enough information, just say 'I don't know':
                        PROMPT: '{prompt}'
                    """)
        response["confluence"] = {
            "result": confluence_response["result"]
        }
        if "I don't know" not in confluence_response["result"]:
            response["confluence"]["source"] = confluence_response["source_documents"][0].metadata["source"]

    if "jira" in seek_list:
        jira_agent = provision_jira()
        jira_response = jira_agent.run(
            f"""You are a JIRA chatbot.
                        Give me a summary of any relevant information from JIRA based on the prompt below.
                        If there is a ticket or user story related to the information, provide a link to it.
                        If you can't find a solution or don't have enough information, just say 'I don't know':
                        PROMPT: '{prompt}'
                    """)
        response["jira"] = jira_response
    if "snow" in seek_list:
        snow_agent = provision_snow()
        snow_response = snow_agent.run(
            f"""You are servicenow chatbot. 
                        Give me an accurate summary of information based on the prompt below.
                        If you can't provide accurate information then at least provide closely matching information by querying all the kb tables and incident tables matching any one of the '{keywords}' or matching '{" ".join(keywords)}.
                        Finally, if you can't find a solution or make a summary do not make one up, just say 'I don't know':
                        PROMPT: '{prompt}'
                    """)
        response["snow"] = snow_response

    return jsonify(response)
