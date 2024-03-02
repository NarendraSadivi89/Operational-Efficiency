import os
import pysnc
import sqlite3
import spacy
import streamlit as st
import pandas as pd
from atlassian import Confluence
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


def handle_question(sql_agent, confluence_chain, jira_agent, prompt, seek_list):
    st.chat_message('user').write(prompt)
    keywords = extract_keywords(prompt)

    if seek_list[0]:
        handle_conf(confluence_chain, prompt)
    if seek_list[1]:
        handle_jira(jira_agent, prompt)
    if seek_list[2]:
        handle_snow(sql_agent, prompt, keywords)


def handle_conf(confluence_chain, prompt):
    with st.spinner("Thinking..."):
        response = confluence_chain(
            f"""You are a confluence chat bot.
                Give me an accurate summary of information based on the prompt below.
                If you can't find a solution or don't have enough information, just say 'I don't know':
                PROMPT: '{prompt}'
            """)
        st.write("From Confluence:\n\n")
        if "I don't know" in response["result"]:
        #if response["result"] == "I don't know":
            st.chat_message('assistant').write(f'{response["result"]}')
        else:
            source_split = response['source_documents'][0].metadata['source'].split('.net/')
            source = f"{source_split[0]}.net/wiki/{source_split[1]}"
            st.chat_message('assistant').write(
                f'{response["result"]}\n\n Source URL: {source}')


def handle_jira(jira_agent, prompt):
    with st.spinner("Thinking..."):
        try:
            response = jira_agent.run(
                f"""You are a JIRA chatbot.
                    Give me a summary of any relevant information from JIRA based on the prompt below.
                    If you can't find anything based on the prompt, just say 'I don't know':
                    PROMPT: '{prompt}'
                """)
        except Exception:
            response = "I don't know."
        st.write("From JIRA:\n\n")
        st.chat_message('assistant').write(response)


def handle_snow(sql_agent, prompt, keywords):
    with st.spinner("Thinking..."):
        
        response = sql_agent.run(
            f"""You have access to glide_{os.getenv('snow_user')}.db and you are an sql expert who can query and could perform complex join operations on the tables based on the '{prompt}'. Run the select query needed in the background and give me the records or result set returned by query but don't give me the select query.
                If you can't provide results based on the '{prompt}' or query then get the kb articles by querying all the kb tables matching short description '{" ".join(keywords)}' or all of the '{keywords}'. Run the select query needed in the background and give me the records or result set returned by query but don't give me the select query or '{keywords}' in your response. Then provide detailed information about the kb article which you have retrieved from the query including kb_knowledge.number,kb_knowledge.short_description and the link from kb_knowledge.text.
                Finally, if you can't find anything then just say "I don't know".
                """)
        st.write("From ServiceNOW/CMDB:\n\n")
        st.chat_message('assistant').write(response)


def provision():
    conf_llm = ChatOpenAI(model='gpt-3.5-turbo-16k-0613', temperature=0)
    jira_llm = ChatOpenAI(model='gpt-4-turbo-preview', temperature=0)
    snow_llm = ChatOpenAI(model='gpt-4-turbo-preview', temperature=0)

    sql_agent = provision_snow(snow_llm)
    jira_agent = provision_jira(jira_llm)
    confluence_chain = provision_confluence(conf_llm)

    return sql_agent, jira_agent, confluence_chain


def provision_snow(llm):
    client = pysnc.ServiceNowClient(os.getenv('snow_url'), (os.getenv('snow_user'), os.getenv('snow_pass')))
    table_list = [
        'task', 'incident', 'sys_user', 'sys_user_group', 'problem', 'wf_workflow',
        'kb_knowledge_base', 'kb_category', 'kb_knowledge',
        'kb_feedback', 'change_request', 'change_task', 'std_change_producer_version',
        'cmdb', 'cmdb_ci', 'cmdb_rel_ci', 'cmdb_ci_computer'
    ]

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

    return create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)


def provision_confluence(llm):
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
        llm=llm,
        chain_type='stuff',
        retriever=knowledge_base.as_retriever(),
        return_source_documents=True
    )


def provision_jira(llm):
    jira = JiraAPIWrapper()
    toolkit = JiraToolkit.from_jira_api_wrapper(jira)
    return initialize_agent(
        toolkit.get_tools(), llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
