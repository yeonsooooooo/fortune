import streamlit as st
import tiktoken
from loguru import logger

from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI

#from langchain.document_loaders import PyPDFLoader
#from langchain.document_loaders import Docx2txtLoader
#from langchain.document_loaders import UnstructuredPowerPointLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.memory import ConversationBufferMemory
#from langchain.vectorstores import FAISS

# from streamlit_chat import message
#from langchain.callbacks import get_openai_callback
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
from langchain_chroma import Chroma
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

    st.title(":four_leaf_clover:_여러분의 타고난 성격이 궁금하신가요?_ :four_leaf_clover:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    dob = st.sidebar.date_input("생년월일을 입력해주세요.", value=None, min_value=date(1900, 1, 1), max_value=date.today())


    # with st.sidebar:
    #     uploaded_files =  st.file_uploader("Upload your file",type=['pdf','docx'],accept_multiple_files=True)
    #     openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    #     process = st.button("Process")
    # if process:
    #     if not openai_api_key:
    #         st.info("Please add your OpenAI API key to continue.")
    #         st.stop()
    #     files_text = get_text(uploaded_files)
    #     text_chunks = get_text_chunks(files_text)
    #     vetorestore = get_vectorstore(text_chunks)
     
    #     st.session_state.conversation = get_conversation_chain(vetorestore,openai_api_key) 

    #     st.session_state.processComplete = True

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
    db = Chroma.from_documents(documents, embedding_function)
    #selected_documents = db.similarity_search(query=user_input, k=5)
    #retriever = db.as_retriever(search_kwargs={"k": 3})
    
    # files_text = get_text(uploaded_files)
    # text_chunks = get_text_chunks(files_text)
    # vetorestore = get_vectorstore(text_chunks)
    
    st.session_state.conversation = get_conversation_chain(db, OPENAI_API_KEY) 
    #st.session_state.conversation = get_consult_chain(db, OPENAI_API_KEY)

    st.session_state.processComplete = True

    #생일 입력을 위함.
    # 세션 상태에 'dob_entered' 키가 없으면 초기화
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
                # result = chain({"question": query})
                # with get_openai_callback() as cb:
                #     st.session_state.chat_history = result['chat_history']
                # response = result['answer']
                # source_documents = result['source_documents']
                response = chain.invoke({"question": query, "personality": docs})

                st.markdown(response)
                # with st.expander("참고 문서 확인"):
                #     st.markdown(source_documents[0].metadata['source'], help = source_documents[0].page_content)
                #     st.markdown(source_documents[1].metadata['source'], help = source_documents[1].page_content)
                #     st.markdown(source_documents[2].metadata['source'], help = source_documents[2].page_content)



# Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

# def get_text(docs):

#     doc_list = []
    
#     for doc in docs:
#         file_name = doc.name  # doc 객체의 이름을 파일 이름으로 사용
#         with open(file_name, "wb") as file:  # 파일을 doc.name으로 저장
#             file.write(doc.getvalue())
#             logger.info(f"Uploaded {file_name}")
#         if '.pdf' in doc.name:
#             loader = PyPDFLoader(file_name)
#             documents = loader.load_and_split()
#         elif '.docx' in doc.name:
#             loader = Docx2txtLoader(file_name)
#             documents = loader.load_and_split()
#         elif '.pptx' in doc.name:
#             loader = UnstructuredPowerPointLoader(file_name)
#             documents = loader.load_and_split()

#         doc_list.extend(documents)
#     return doc_list


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks


#def get_vectorstore(text_chunks):
    # embeddings = HuggingFaceEmbeddings(
    #                                     model_name="jhgan/ko-sroberta-multitask",
    #                                     model_kwargs={'device': 'cpu'},
    #                                     encode_kwargs={'normalize_embeddings': True}
    #                                     )  
    #vectordb = FAISS.from_documents(text_chunks, embeddings)
    #return vectordb

def get_conversation_chain(vetorestore,openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name = 'gpt-3.5-turbo',temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, 
            chain_type="stuff", 
            retriever=vetorestore.as_retriever(search_type = 'mmr', vervose = True, search_kwargs={"k": 3}), 
            memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
            get_chat_history=lambda h: h,
            return_source_documents=True,
            verbose = True
        )

    return conversation_chain

def get_docs(vectorstore, user_input):
    # openai = ChatOpenAI(model_name="gpt-3.5-turbo-0125",
    #                 streaming=True, callbacks=[StreamingStdOutCallbackHandler()],
    #                 temperature = 0,
    #                 openai_api_key=OPENAI_API_KEY)
    
    print("\n\n", user_input)
    #retriever=vectorstore.as_retriever(search_type = 'mmr', vervose = True, search_kwargs={"k": 3}), 
    #docs = retriever.invoke(user_input)

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