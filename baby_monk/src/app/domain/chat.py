from fastapi import FastAPI, HTTPException
from typing import List
from pydantic import BaseModel


app = FastAPI()
class Chat(BaseModel):
    id: int
    messages: List[str]

chats = {}

@app.post("/chat/{member_id}", response_model=Chat)
async def create_chat(member_id: int, chat: Chat):
    chats[member_id] = chat
    return chat

@app.get("/chat/{member_id}", response_model=Chat)
async def get_chat(member_id: int):
    if member_id in chats:
        return chats[member_id]
    raise HTTPException(status_code=404, detail="Chat not found")
