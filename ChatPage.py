import os
from utils.CheckPassword import check_password
from utils.Utils import provision, handle_question
import streamlit as st


class ChatPage:
    def __init__(
            self,
            page_title,
            page_icon,
            header
    ):
        if not check_password():
            st.stop()

        st.session_state.pred_prompt = None

        st.set_page_config(page_title=page_title, page_icon=page_icon)
        with st.sidebar:
            st.subheader('Helpful Info & Resources')
            st.info('ℹ  Use the Knowledge Base Chatbot to ask questions in regards to your Confluence, ServiceNOW, '
                    'CMDB, and JIRA instances. You can scope your query to specific knowledge bases using the \'Seek '
                    'answers from:\' expander. Input your question in the provided space towards the bottom of the '
                    'screen and send to get your answer.')
            with st.expander("Take a look under the hood"):
                st.image('assets/TechStackDiagram.png')
                st.write('Above is the tech stack diagram. You may expand the image or sidebar for better viewing.')
            with st.expander("View Atlassian source instance URLs"):
                st.write(f'Confluence: {os.getenv("confluence_url")}\n\n'
                         f'JIRA: {os.getenv("jira_instance_url")}\n\n'
                         f'ServiceNow/CMDB: {os.getenv("snow_url")}')

        st.image('assets/cgi-logo.png')
        st.header(header)

        with st.expander('Seek answers from:'):
            left_co_seek, cent_co_seek, right_co_seek = st.columns(3)
            with left_co_seek:
                seek_confluence = st.checkbox(label='Confluence', value=True)
            with cent_co_seek:
                seek_jira = st.checkbox(label='JIRA', value=True)
            with right_co_seek:
                seek_snow = st.checkbox(label='ServiceNow/CMDB', value=True)

        seek_list = [seek_confluence, seek_jira, seek_snow]

        left_co_button, cent_co_button, right_co_button = st.columns([1, 1, 1])
        with left_co_button:
            if st.button(
                'How do I reset my password?',
                type="primary"
            ):
                st.session_state.pred_prompt = 'How do I reset my password?'
        with cent_co_button:
            if st.button(
                'How do I set java system PATH on Windows?',
                type="primary"
            ):
                st.session_state.pred_prompt = 'How do I set java system PATH on Windows?'
        with right_co_button:
            if st.button(
                'What is Langchain?',
                type="primary"
            ):
                st.session_state.pred_prompt = 'What is Langchain?'

        with st.spinner('Loading...'):
            sql_agent, jira_agent, confluence_chain = provision()

        if st.session_state.pred_prompt:
            handle_question(sql_agent, confluence_chain, jira_agent, st.session_state.pred_prompt, seek_list)

        st.container()
        if prompt := st.chat_input(placeholder="Ask about your Knowledge Base..."):
            handle_question(sql_agent, confluence_chain, jira_agent, prompt, seek_list)


if __name__ == '__main__':
    ChatPage(
        page_title="Chat",
        page_icon="🤖",
        header="VishnuAI - Ask Your Knowledge Base"
    )
