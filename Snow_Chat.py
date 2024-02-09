import streamlit as st
from dotenv import load_dotenv
from Snow_Utils import provision_snow, handle_question
from htmlTemplates import css


def main():
    load_dotenv()

    st.set_page_config(page_title="Snow Chat", page_icon="❄")
    st.write(css, unsafe_allow_html=True)

    # if "conversation" not in st.session_state:
    #     st.session_state.conversation = None
    #
    # if "chat_history" not in st.session_state:
    #     st.session_state.chat_history = None

    agent = provision_snow()

    st.header("Snow Chat ❄")
    st.container()
    if prompt := st.chat_input():
        handle_question(agent, prompt)


if __name__ == '__main__':
    main()
