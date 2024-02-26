import hmac
import os
import streamlit as st
from dotenv import dotenv_values


def check_password():
    """Returns `True` if the user had a correct password."""

    def set_env_vars(vars_dict):
        for key, var in vars_dict.items():
            os.environ[key.lower()] = var

    def login_form():
        """Form with widgets to collect user information"""
        with st.form("Credentials"):
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            st.form_submit_button("Log in", on_click=password_entered)

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["username"] in st.secrets[
            "passwords"
        ] and hmac.compare_digest(
            st.session_state["password"],
            st.secrets.passwords[st.session_state["username"]],
        ):
            st.session_state["password_correct"] = True
            if st.session_state["username"] == "admin":
                config = dotenv_values(".env")
                set_env_vars(config)
            elif st.session_state["username"] == "sub_user":
                config = dotenv_values(".env.sub-user")
                set_env_vars(config)

            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the username + password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show inputs for username + password.
    login_form()
    if "password_correct" in st.session_state:
        st.error("😕 User not known or password incorrect")
    return False

