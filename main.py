import streamlit as st
import tiktoken
from loguru import logger
import requests
import xml.etree.ElementTree as ET
import urllib.parse
import time

from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.memory import ConversationBufferMemory
from langchain.memory import StreamlitChatMessageHistory

#new added
from dotenv import load_dotenv
import os
from datetime import date
import datetime


load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
OPENAPI_KEY = os.getenv('OPENAPI_KEY')


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
                    temperature = 1,
                    openai_api_key=OPENAI_API_KEY)


def main():
    st.set_page_config(
        page_title="사주풀이",
        page_icon=":four_leaf_clover:"
    )

    st.title(":four_leaf_clover:_여러분의 타고난 성격이 궁금하신가요?_ :four_leaf_clover:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    dob = st.sidebar.date_input("생년월일을 입력해주세요.", value=None, min_value=date(1900, 1, 1), max_value=date.today())

    st.session_state.processComplete = True

    # 세션 상태에 'input_count'와 'dob_entered' 키가 없으면 초기화
    if 'input_count' not in st.session_state:
        st.session_state['input_count'] = 0  # 입력 횟수를 추적하는 변수
    if 'dob_entered' not in st.session_state:
        st.session_state['dob_entered'] = False

    # 사용자가 생년월일을 입력했는지 확인
    if dob and not st.session_state['dob_entered']:
        # 생년월일 입력 처리
        st.session_state['dob_entered'] = True
        st.session_state['dob'] = dob
        # 입력 횟수 증가
        st.session_state['input_count'] += 1

    # st.session_state['file_path'] = ""
    # session_state를 사용하여 메세지 상태 관리
    # 메세지가 없거나 생년월일 입력이 되었다면 실행
    if 'messages' not in st.session_state or st.session_state['dob_entered']:
        # 생년월일이 입력되지 않았다면, 입력 요청 메세지 추가
        if not st.session_state['dob_entered']:
            print("\n\nSTart from here")
            st.session_state['messages'] = [{"role": "user", "content": "사이드바를 열고, 생년월일을 먼저 입력해주세요!"}]
            st.write("\n왼쪽 사이드바를 열고, 생년월일을 먼저 입력해주세요 !")
        else:
            if st.session_state['dob_entered'] and st.session_state['input_count']:
                print("st.session_state['input_count'] : ", st.session_state['input_count'])
                st.session_state['messages'] = [{"role": "assistant", "content": f"당신만을 위한 사주 마스터 애기동자입니다. 궁금하신 것이 있으면 언제든 물어봐주세요! 당신의 생년월일은 {dob}입니다."}]
                
            if st.session_state['input_count'] == 1:
                # 최대 재시도 횟수 설정
                MAX_RETRIES = 5
                attempts = 0

                while attempts < MAX_RETRIES:
                    try:
                        # 인증키를 URL 인코딩
                        encoded_key = urllib.parse.quote(OPENAPI_KEY)
                        print("Encoded Key: ", encoded_key)
                        
                        # datetime 객체에서 년, 월, 일 추출
                        solYear = dob.year
                        solMonth = dob.month
                        solDay = dob.day

                        # API 요청 URL
                        api_url = 'http://apis.data.go.kr/B090041/openapi/service/LrsrCldInfoService/getLunCalInfo'

                        # API 요청에 필요한 파라미터 준비
                        params = {
                            'serviceKey': encoded_key,
                            'solYear': str(solYear),
                            'solMonth': str(solMonth).zfill(2),
                            'solDay': str(solDay).zfill(2)
                        }

                        print("INPUT PARAMS: ", params)
                        # API 요청 및 응답 받기
                        response = requests.get(api_url, params=params)
                        print("RESPONSE: ", response)

                        if response.status_code == 200:
                            # 응답 본문을 UTF-8로 디코딩
                            xml_data = response.content.decode('utf-8')
                            print("xml_Data: ", xml_data)

                            # XML 선언 제거
                            start_index = xml_data.find('<?xml')
                            if start_index != -1:
                                end_index = xml_data.find('?>', start_index)
                                if end_index != -1:
                                    xml_data = xml_data[end_index + 2:]

                            # XML 응답 파싱
                            root = ET.fromstring(xml_data)
                            print("ROOT: ", root)

                            # 필요한 데이터 추출
                            lun_date = root.find('.//lunIljin').text
                            if lun_date is None:
                                raise AttributeError("lunIljin element not found or has no text.")
                            lun_month = root.find('.//lunWolgeon').text
                            lun_year = root.find('.//lunSecha').text

                            # 기본값 설정
                            if lun_month is None:
                                print("\n\n월의 정보가 비어 있음.\n\n")
                                lun_month = "00"  

                            if lun_year is None:
                                print("\n\n연의 정보가 비어 있음.\n\n")
                                lun_year = "00"


                            _lun_date = lun_date
                            _lun_month = lun_month 
                            _lun_year = lun_year

                            eight = {
                                "1": "0",
                                "2": "0",
                                "3": _lun_date[0],
                                "4": _lun_date[1],
                                "5": _lun_month[0],
                                "6": _lun_month[1],
                                "7": _lun_year[0],
                                "8": _lun_year[1],
                            }
                            
                            #년월일 로 텍스트 조회하기.
                            base_url = "https://port-0-baby-monk-logic-rm6l2llwb02l9k.sel5.cloudtype.app"
                            endpoint = "/saju/saju-info"

                            saju_url = base_url + endpoint

                            data = {
                                "eight": eight
                            }

                            print("\n\ndata:", data)
                            saju_response = requests.post(saju_url, json=data)

                            print("saju_response Status Code:", saju_response.status_code)
                            print("\n\nsaju_response Response Body:", saju_response.json())

                            temp_text = saju_response.json()
                            
                            # 현재 시간을 기반으로 고유한 파일명 생성
                            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
                            filename = f'test_{timestamp}.txt'
                            file_path = os.path.join('usertext', filename)

                            # 디렉토리가 존재하지 않으면 생성
                            os.makedirs('usertext', exist_ok=True)

                            # 파일을 생성하고 내용을 기록
                            with open(file_path, 'w', encoding='utf-8') as file:
                                file.write(temp_text)

                            # 파일 경로를 세션 상태에 저장
                            st.session_state['file_path'] = file_path

                            st.session_state['input_count'] = 0

                            if saju_response.status_code == 200 or 201:
                                print(f"File saved successfully at {file_path}")
                                break
                            
                    except requests.exceptions.RequestException as e:
                        # 예외가 발생하면 에러 메시지를 출력하고 재시도합니다.
                        print(f"Request failed: {e}, retrying... ({attempts + 1}/{MAX_RETRIES})")
                        attempts += 1
                        time.sleep(1)  # 잠시 대기 후 재시도
                    except AttributeError as e:
                        print(f"Request failed: {e}, retrying... ({attempts + 1}/{MAX_RETRIES})")
                        attempts += 1
                        time.sleep(3)  # 잠시 대기 후 재시도
                else:
                    # 최대 재시도 횟수를 초과한 경우
                    raise Exception("Maximum retry attempts reached, failing...")

            #수정된 부분
            # 세션 상태에 저장된 파일 경로를 사용하여 불러온 텍스트 데이터 벡터 저장소에 임베딩
            loader_test = TextLoader(st.session_state['file_path'], 'utf-8')
            test_document = loader_test.load()

            tokenizer = tiktoken.get_encoding("cl100k_base")
            def tiktoken_len(text):
                tokens = tokenizer.encode(text)
                return len(tokens)

            # split
            text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100, length_function=tiktoken_len)
            documents = text_splitter.split_documents(test_document)

            # add to vectorstore
            embedding_function = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-ada-002")
            db = FAISS.from_documents(documents, embedding_function)
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            history = StreamlitChatMessageHistory(key="chat_messages")


            print("\n\nCOUNT: ", st.session_state['input_count'])
            if st.session_state['input_count'] == 0:
                print("\n\nCHAT LOGIC")
                # Chat logic
                if query := st.chat_input("질문을 입력해주세요."):
                    st.session_state.messages.append({"role": "user", "content": query})

                    with st.chat_message("user"):
                        st.markdown(query)

                    with st.chat_message("assistant"):
                        docs = get_docs(db, query)
                        print("\n\ndocs : ", docs)
                        chain = create_consult_chain(query, docs)

                        with st.spinner("애기동자가 사주를 보고 있어요..."):
                            response = chain.invoke({"question": query, "personality": docs})

                            st.markdown(response)

                    # Add assistant message to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})


            if st.session_state['input_count'] >= 2:
                st.session_state['messages'] = [{"role": "assistant", "content": f"당신만을 위한 사주 마스터 애기동자입니다. 궁금하신 것이 있으면 언제든 물어봐주세요! 당신의 생년월일은 {dob}입니다."}]
                print("\n\n이곳은 엘스 입니다 ")
                print("\n\ndob : ", dob)

            
                # Chat logic
                if query := st.chat_input("질문을 입력해주세요."):
                    st.session_state.messages.append({"role": "user", "content": query})

                    with st.chat_message("user"):
                        st.markdown(query)

                    with st.chat_message("assistant"):
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
    docs = vectorstore.similarity_search(user_input)
    return docs



def create_consult_chain(user_input, docs):
    template = """
    You are a person who provides counseling based on human personality.
    Analysis and advice: Please provide insights and advice on the personality of {personality}.
    Describe their tendencies, character, and traits. 
    Additionally, analyze and offer advice on what they should be mindful of, what they need, and what they might be lacking. 
    Offer practical advice that could help this individual live a better life.
    You should do your best to speak in the most friendly tone possible.
    Always answer in  Korean. Response within 120 words.

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