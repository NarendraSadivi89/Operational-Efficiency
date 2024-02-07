import streamlit as st
from dotenv import load_dotenv
from htmlTemplates import css


def main():
    load_dotenv()
    st.set_page_config(page_title="Snow Chat", page_icon="❄")
    st.write(css, unsafe_allow_html=True)
    st.header("Snow Chat ❄")
#write logic

if __name__ == '__main__':
    main()
