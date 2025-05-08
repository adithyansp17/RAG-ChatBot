from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

def build_chat_history(chat_history_list):
    chat_history = []
    for message in chat_history_list:
        chat_history.append(HumanMessage(content=message[0]))
        chat_history.append(AIMessage(content=message[1]))

    return chat_history

def query(question, chat_history):
    chat_history = build_chat_history(chat_history)
    embeddings = OpenAIEmbeddings()
    new_db = FAISS.load_local("faiss_index",embeddings,allow_dangerous_deserialization=True)
    llm=ChatOpenAI(model_name = "gpt-4o",temperature = 0)

    condense_question_system_template = ()
    condense_question_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",condense_question_system_template),
            ("placeholder","{chat_history}"),
            ("human","{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm,new_db.as_retriever(),condense_question_prompt
    )

    system_promt = ()
    qa_promt = ChatPromptTemplate.from_messages(
        [
            ("system",system_promt),
            ("placeholder","{chat_history}"),
            ("human","{input}"),
        ]
    )

    qa_chain = create_stuff_documents_chain(llm,qa_promt)
    convo_qa_chain= create_retrieval_chain(history_aware_retriever,qa_chain)

    return convo_qa_chain.invoke(
        {
            "input":question,
            "chat_history":chat_history,
        }
    )

def showUI():
    st.title("hello")
    st.subheader("world")

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chat_history = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("enter"):
        with st.spinner("Loading.."):
            response = query(question=prompt,chat_history=st.session_state.chat_history)
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                st.markdown(response["answer"])

            st.session_state.messages.append({"role":"user","content":prompt})
            st.session_state.messages.append({"role":"assistant","content":response["answer"]})
            st.session_state.chat_history.extend([(prompt,response["answer"])])



if __name__ == "__main__":
    showUI()
            