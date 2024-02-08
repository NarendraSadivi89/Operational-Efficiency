import streamlit as st
import pandas as pd
import tiktoken
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import faiss
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from htmlTemplates import user_template, bot_template


# Custom template to guide llm model
custom_template = """Given the following user question, corresponding SQL query, and SQL result, answer the user 
question.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """

CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)


def handle_question(question):
    response = st.session_state.conversation({'question': question})
    st.session_state.chat_history = response["chat_history"]
    for i, msg in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", msg.content, ), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)


def get_text_chunks(json_response_file_loc):

    # Create a dataframe from the list of texts
    df = pd.read_json(json_response_file_loc)
    df.head()

    # Load the cl100k_base tokenizer which is designed to work with the ada-002 model
    tokenizer = tiktoken.get_encoding("cl100k_base")

    shortened = []
    max_tokens = 500
    # Loop through the dataframe
    for row in df.iterrows():
        row_str = str(row)
        row_token_len = len(tokenizer.encode(row_str))

        # If the text is None, go to the next row
        if row_str is None:
            continue

        # If the number of tokens is greater than the max number of tokens, split the text into chunks
        if row_token_len > max_tokens:
            shortened += split_into_many(row_str, max_tokens, tokenizer)

        # Otherwise, add the text to the list of shortened texts
        else:
            shortened.append(row_str)

    df = pd.DataFrame(shortened, columns=['text'])
    return df['text'].astype(str).tolist()


def get_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    vectorstore = faiss.FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(temperature=0.2)
    memory = ConversationBufferMemory(memory_key='chat_history',
                                      return_messages=True,
                                      output_key='answer')  # using conversation buffer memory to hold past information
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        condense_question_prompt=CUSTOM_QUESTION_PROMPT,
        memory=memory)
    return conversation_chain


def remove_newlines(serie):
    serie = serie.str.replace('\n', ' ')
    serie = serie.str.replace('\\n', ' ')
    serie = serie.str.replace('  ', ' ')
    serie = serie.str.replace('  ', ' ')
    return serie


def split_into_many(text, max_tokens, tokenizer):
    # Split the text into sentences
    sentences = text.split('}, {')

    # Get the number of tokens for each sentence
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]

    chunks = []
    tokens_so_far = 0
    chunk = []

    # Loop through the sentences and tokens joined together in a tuple
    for sentence, token in zip(sentences, n_tokens):

        # If the number of tokens so far plus the number of tokens in the current sentence is greater
        # than the max number of tokens, then add the chunk to the list of chunks and reset
        # the chunk and tokens so far
        if tokens_so_far + token > max_tokens:
            chunks.append("}, {".join(chunk))
            chunk = []
            tokens_so_far = 0

        # If the number of tokens in the current sentence is greater than the max number of
        # tokens, go to the next sentence
        if token > max_tokens:
            continue

        # Otherwise, add the sentence to the chunk and add the number of tokens to the total
        chunk.append(sentence)
        tokens_so_far += token + 1

    return chunks
