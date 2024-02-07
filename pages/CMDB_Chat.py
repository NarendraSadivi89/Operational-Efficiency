import streamlit as st
from dotenv import load_dotenv
from htmlTemplates import css


def main():
    load_dotenv()
    st.set_page_config(page_title="CMDB Chat", page_icon="ℹ")
    st.write(css, unsafe_allow_html=True)
    st.header("CMDB Chat ℹ")


if __name__ == '__main__':
    main()
