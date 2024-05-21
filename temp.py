import streamlit as st
import tiktoken
from loguru import logger
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.memory import ConversationBufferMemory
from langchain.memory import StreamlitChatMessageHistory
#new added
from dotenv import load_dotenv
import os
from datetime import date


load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')


from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
import tiktoken
#from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


openai = ChatOpenAI(model_name="gpt-3.5-turbo-0125",
                    streaming=True, callbacks=[StreamingStdOutCallbackHandler()],
                    temperature = 0,
                    openai_api_key=OPENAI_API_KEY)

def main():
    st.set_page_config(
    page_title="사주풀이",
    page_icon=":four_leaf_clover:")

    st.title(":four_leaf_clover:_여러분의 타고난 성격이 궁금하신가요?_")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    dob = st.sidebar.date_input("생년월일을 입력해주세요.", value=None, min_value=date(1900, 1, 1), max_value=date.today())

    loader_test = TextLoader('test.txt', 'utf-8')

    test_document = loader_test.load()

    tokenizer = tiktoken.get_encoding("cl100k_base")
    def tiktoken_len(text):
        tokens = tokenizer.encode(text)
        return len(tokens)

    #split
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100, length_function = tiktoken_len)
    documents = text_splitter.split_documents(test_document)

    #add to vectorstore
    embedding_function = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small")
    db = FAISS.from_documents(documents, embedding_function)

    st.session_state.processComplete = True

    if 'dob_entered' not in st.session_state:
        st.session_state['dob_entered'] = False

    # 사용자가 생년월일을 입력했는지 확인
    if dob is not None and not st.session_state['dob_entered']:
        # 생년월일 입력 처리
        print("\n\ndebug")
        st.session_state['dob_entered'] = True
        st.session_state['dob'] = dob

    # session_state를 사용하여 메세지 상태 관리
    if 'messages' not in st.session_state or st.session_state['dob_entered']:
        # 생년월일이 입력되지 않았다면, 입력 요청 메세지 추가
        if not st.session_state['dob_entered']:  
            st.session_state['messages'] = [{"role": "user", "content": "사이드바를 열고, 생년월일을 먼저 입력해주세요!"}]
        else:
            st.session_state['messages'] = [{"role": "assistant", "content": f"안녕하세요! 궁금하신 것이 있으면 언제든 물어봐주세요! 당신의 생년월일은 {dob}입니다."}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    history = StreamlitChatMessageHistory(key="chat_messages")

    # Chat logic
    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            #chain = st.session_state.conversation
            docs = get_docs(db, query)
            print("\n\ndocs : ", docs)
            chain = create_consult_chain(query, docs)

            with st.spinner("Thinking..."):
                response = chain.invoke({"question": query, "personality": docs})

                st.markdown(response)

# Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks

def get_docs(vectorstore, user_input):
    print("\n\n", user_input)
    docs = vectorstore.similarity_search(user_input)
    return docs


def create_consult_chain(user_input, docs):
    template = """

    Analysis and advice: Please provide insights and advice on the personality of {personality}.
    Describe their tendencies, character, and traits. Additionally, analyze and offer advice on what they should be mindful of, what they need, and what they might be lacking. 
    Offer practical advice that could help this individual live a better life.
    Always answer in  Korean.

    Answer the question based on the personality:
    Question: {question}

    """

    system_message_prompt = ChatPromptTemplate.from_template(template)

    consult_chain = (
        {"question": RunnablePassthrough(), "personality": RunnablePassthrough()}
        | system_message_prompt
        | openai
        | StrOutputParser()
    )

    return consult_chain



if __name__ == '__main__':
    main()
