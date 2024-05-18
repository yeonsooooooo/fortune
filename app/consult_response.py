from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv

#new
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
import tiktoken
from langchain_core.runnables import RunnablePassthrough
import os

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

openai = ChatOpenAI(model_name="gpt-3.5-turbo-0125",
                    streaming=True,
                    temperature = 0,
                    openai_api_key=OPENAI_API_KEY)

openai_stream = ChatOpenAI(model_name="gpt-3.5-turbo-0125",
                    streaming=True, callbacks=[StreamingStdOutCallbackHandler()],
                    temperature = 0,
                    openai_api_key=OPENAI_API_KEY)
#load
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
# db = Chroma.from_documents(documents, embedding_function, persist_directory="./test_db")
db = Chroma.from_documents(documents, embedding_function)
#selected_documents = db.similarity_search(query=user_input, k=5)
retriever = db.as_retriever(search_kwargs={"k": 3})

template = """

Analysis and advice: Please provide insights and advice on the personality of {personality}.
Describe their tendencies, character, and traits. Additionally, analyze and offer advice on what they should be mindful of, what they need, and what they might be lacking. 
Offer practical advice that could help this individual live a better life.
Respond in Korean within 100 characters.

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