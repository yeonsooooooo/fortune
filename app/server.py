from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Union
from langserve.pydantic_v1 import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langserve import add_routes
#from app.response_chain import response_chain
from consult_response import consult_chain, retriever
#from app.chat import chain as chat_chain

from starlette.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

#streaming
from fastapi.responses import StreamingResponse
import httpx
import asyncio

#from consult_chain import consult_chain, retriever 

from dotenv import load_dotenv


load_dotenv()

app = FastAPI()

#app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="../templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

#get 메서드로 "/" 경로에 접속하면, 아래의 함수를 실행하라.
@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/mystery")

@app.get("/mystery", response_class=HTMLResponse)
async def chat_page(request: Request, query: str = None):  # 쿼리 파라미터를 받는 부분 추가
    # 쿼리 파라미터가 있는 경우, 해당 쿼리로 처리하여 답변 반환
    if query:
        try:
            docs = retriever.invoke(query)
            answer = consult_chain.invoke({"question": query, "personality": docs})
            return JSONResponse(content={"answer": answer})
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    # 쿼리 파라미터가 없는 경우, 일반 chat 페이지 반환
    else:
        return templates.TemplateResponse("chat.html", {"request": request})

# @app.post("/mystery")
# async def get_answer(user_input):
#     try:
#         docs = retriever.invoke(user_input)
#         answer = consult_chain.invoke({"question": user_input, "personality" : docs})
#         return JSONResponse(content={"answer": answer})
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

#for streaming
async def get_user_answer(user_input: str):
    chunks = []

    docs = retriever.invoke(user_input)
    
    async for chunk in consult_chain.astream({"question": user_input, "personality": docs}):
        chunks.append(chunk)
        #print("\n\nCHUNK:", chunk, "\n")
        yield f"data: {chunk}\n\n"
    
    print("\n\nCHUNKS : ", chunks)
    

@app.get('/stream')
async def stream(query):
    #답변 온걸 한번 끊고, 일단 실시간 답변을 하고 나머지는 이후에 일어나도록 해야한다.
    return StreamingResponse(get_user_answer(query), media_type='text/event-stream')

@app.post("/mystery")
async def get_answer(request: Request):
    try:
        # 요청 본문을 비동기적으로 처리
        user_input = await request.json()
        
        print("user_input : ", user_input)
        # httpx 라이브러리를 사용하여 비동기적으로 /stream 엔드포인트에 요청을 보냄
        async with httpx.AsyncClient() as client:
            #print(response.status_code)
            print("\n\n디버깅")
            async with client.stream("GET", f"http://localhost:8000/stream?query={user_input['query']}") as response:
                first_real_time_answer = None
                async for chunk in response.aiter_text():
                    if chunk:
                        print(chunk, end='', flush=True)
                        # 실시간으로 첫 번째 답변을 받은 후 반복문을 종료
                        first_real_time_answer = chunk.strip()
                        break

        return JSONResponse(content={"answer": first_real_time_answer})

    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)





if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


