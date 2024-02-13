import os
import pysnc
import pandas as pd
from flask import Flask, request, jsonify
from langchain.agents import AgentType
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ConfluenceLoader
from langchain_community.document_transformers import Html2TextTransformer
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
    print(df)

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
