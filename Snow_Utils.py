import streamlit as st
from operator import itemgetter
from langchain.chains import create_sql_query_chain
from langchain_community.chat_models import ChatOpenAI
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from htmlTemplates import user_template, bot_template


# Custom template to guide llm model
custom_template = """Given the following user question, corresponding SQL query, and SQL result, answer the user 
question.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """

CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)


def answer_sql_q(question, db_name):
    db = SQLDatabase.from_uri(f"sqlite:///{db_name}")
    st.write(user_template.replace("{{MSG}}", question, ), unsafe_allow_html=True)
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    write_query = create_sql_query_chain(llm, db)
    execute_query = QuerySQLDataBaseTool(db=db)

    msg = write_query.invoke({"question": question})
    st.write(bot_template.replace("{{MSG}}", "SQL Query Used Below"), unsafe_allow_html=True)
    st.code(msg)

    answer = CUSTOM_QUESTION_PROMPT | llm | StrOutputParser()
    chain = (
        RunnablePassthrough.assign(query=write_query).assign(
            result=itemgetter("query") | execute_query
        )
        | answer
    )
    msg = chain.invoke({"question": question})
    st.write(bot_template.replace("{{MSG}}", msg), unsafe_allow_html=True)


# def handle_db_upload(db_file, db_name):
#     stringio = StringIO(db_file.getvalue().decode("utf-8"))
#     sql_commands = stringio.read()
#
#     conn = sqlite3.connect(db_name)
#     cur = conn.cursor()
#     cur.executescript(sql_commands)
#     conn.commit()
#     conn.close()

