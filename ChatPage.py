import streamlit as st
from dotenv import load_dotenv
from htmlTemplates import css
from utils.Utils import provision, handle_question


class ChatPage:
    def __init__(
            self,
            page_title,
            page_icon,
            header
    ):
        load_dotenv()
        st.set_page_config(page_title=page_title, page_icon=page_icon)
        st.write(css, unsafe_allow_html=True)
        st.image('assets/cgi-logo.png')
        st.header(header)

        # if "conversation" not in st.session_state:
        #     st.session_state.conversation = None
        #
        # if "chat_history" not in st.session_state:
        #     st.session_state.chat_history = None
        with st.spinner('Loading...'):
            chain, sql_agent, jira_agent = provision()

        st.container()
        if prompt := st.chat_input():
            handle_question(sql_agent, chain, jira_agent, prompt)


if __name__ == '__main__':
    ChatPage(
        page_title="Chat",
        page_icon="🤖",
        header="Ask your question"
    )
