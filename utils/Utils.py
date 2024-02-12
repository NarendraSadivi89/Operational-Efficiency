import os
import pysnc
import streamlit as st
import pandas as pd
from langchain.agents import AgentType
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI


def handle_question(agent, prompt):
    st.chat_message('user').write(prompt)
    with st.chat_message('assistant'):
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent.invoke(
            {'input': prompt}, {'callbacks': [st_callback]}
        )
        st.write(response['output'])


def provision_glide(record_type):
    client = pysnc.ServiceNowClient('https://cgigroupincdemo15.service-now.com', ('api_user', os.getenv('snow_pass')))
    gr = client.GlideRecord(record_type)
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
