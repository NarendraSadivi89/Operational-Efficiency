import os

import pysnc
import streamlit as st
import pandas as pd
from langchain.agents.agent_types import AgentType
from langchain.callbacks.streamlit import StreamlitCallbackHandler
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI


# general_system_template = r"""
# Your are a professional ServiceNow developer. Give a detailed answer aimed at simple users. Start your explanations off in simple terms. Include code snippets if appropriate. If you don't know the answer, simply say you don't know.
#  ----
# {context}
# ----
# """
# general_user_template = "Question:```{question}```"
# messages = [
#     SystemMessagePromptTemplate.from_template(general_system_template),
#     HumanMessagePromptTemplate.from_template(general_user_template)
# ]
# qa_prompt = ChatPromptTemplate.from_messages(messages)


def handle_question(agent, prompt):
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent.invoke(
            {"input": prompt}, {"callbacks": [st_callback]}
        )
        st.write(response["output"])


def provision_snow():
    client = pysnc.ServiceNowClient('https://cgigroupincdemo15.service-now.com', ('api_user', os.getenv('snow_pass')))
    gr = client.GlideRecord('incident')
    gr.add_query("active", "true")
    gr.query()
    df = pd.DataFrame(gr.to_pandas())
    print(df)

    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0125")
    return create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
    )
