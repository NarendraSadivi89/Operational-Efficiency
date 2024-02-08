import streamlit as st
from dotenv import load_dotenv
from Snow_Utils import handle_question, get_text_chunks, get_vectorstore, get_conversation_chain
from htmlTemplates import css


def main():
    load_dotenv()
    st.set_page_config(page_title="Snow Chat", page_icon="❄")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.session_state.provisioning = True
    while st.session_state.provisioning:
        # get the text chunks
        text_chunks = get_text_chunks('data/snow/response.json')
        st.write('✅ Finished getting chunks.')

        # create vectorstore
        st.write('Preparing embeddings...')
        vectorstore = get_vectorstore(text_chunks)
        st.write('✅ Embeddings completed.')

        # create conversation chain
        st.session_state.conversation = get_conversation_chain(vectorstore)

        st.session_state.provisioning = False

    st.header("Snow Chat ❄")
    question = st.text_input("Ask question from your domain:")
    if question:
        handle_question(question)

if __name__ == '__main__':
    main()
