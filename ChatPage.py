import os
from utils.CheckPassword import check_password
from utils.Utils import provision, handle_question
import streamlit as st
from dotenv import load_dotenv


class ChatPage:
    def __init__(
            self,
            page_title,
            page_icon,
            header
    ):
        load_dotenv()
        if not check_password():
            st.stop()
        st.session_state["username"] = st.session_state["username"]

        st.session_state.pred_prompt = None

        st.set_page_config(page_title=page_title, page_icon=page_icon)
        with st.sidebar:
            st.subheader(f"👋 Welcome {st.session_state['username']}")
            st.subheader('Helpful Info & Resources')
            st.info('ℹ  Use the Knowledge Base Chatbot to ask questions in regards to your Confluence, ServiceNOW, '
                    'CMDB, and JIRA instances. You can scope your query to specific knowledge bases using the \'Seek '
                    'answers from:\' expander. Input your question in the provided space towards the bottom of the '
                    'screen and send to get your answer.')
            # with st.expander("Take a look under the hood"):
            #     st.image('assets/TechStackDiagram.png')
            #     st.write('Above is the tech stack diagram. You may expand the image or sidebar for better viewing.')
            with st.expander("View source instance URLs"):
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

        with st.spinner('Loading...'):
            sql_agent, jira_agent, confluence_chain = provision()

            far_co_button, far_far_co_button = st.columns([1, 1])
            with far_co_button:
                if st.button(
                        'How many tickets are in GenAITeam project?',
                        type="primary"
                ):
                    st.session_state.pred_prompt = 'How many tickets are in GenAITeam project?'
            with far_far_co_button:
                if st.button(
                        'Give me information on ticket with key GT-4.',
                        type="primary"
                ):
                    st.session_state.pred_prompt = 'Give me information on ticket with key GT-4.'

            left_co_button, cent_co_button, right_co_button = st.columns([1, 1, 1])
            with left_co_button:
                if st.button(
                        'How do I reset my password?',
                        type="primary"
                ):
                    st.session_state.pred_prompt = 'How do I reset my password?'
            with cent_co_button:
                if st.button(
                        'How do I set PATH variable?',
                        type="primary"
                ):
                    st.session_state.pred_prompt = 'How do I set PATH variable?'
            with right_co_button:
                if st.button(
                        'How do I setup Selenium?',
                        type="primary"
                ):
                    st.session_state.pred_prompt = 'How do I setup Selenium?'


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
