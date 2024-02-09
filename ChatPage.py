import streamlit as st
from dotenv import load_dotenv
from htmlTemplates import css
from utils.Confluence_Utils import provision_confluence
from utils.Utils import provision_glide, handle_question


class ChatPage:
    def __init__(
            self,
            page_title,
            page_icon,
            header,
            options=None,
            glider_based=False
    ):
        load_dotenv()

        st.set_page_config(page_title=page_title, page_icon=page_icon)
        st.write(css, unsafe_allow_html=True)
        st.header(header)

        # if "conversation" not in st.session_state:
        #     st.session_state.conversation = None
        #
        # if "chat_history" not in st.session_state:
        #     st.session_state.chat_history = None
        if glider_based:
            label = 'Please choose the record type you\'re attempting to chat about.'
            record_type = st.selectbox(label=label, options=options, placeholder="Choose an option")
            agent = provision_glide(record_type)

            st.container()
            if prompt := st.chat_input():
                handle_question(agent, prompt)
        else:
            # TODO: Confluence
            try:
                provision_confluence()
            except Exception:
                pass