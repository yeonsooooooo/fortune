from datetime import date, time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI()


class Member(BaseModel):
    id: str
    name: str
    birth_date: date
    birth_time: time


members = []


@app.post("/members/", response_model=Member)
async def create_member(member: Member):
    members.append(member)
    return member


@app.get("/members/", response_model=List[Member])
async def get_members():
    return members


@app.get("/members/{member_id}", response_model=Member)
async def get_member(member_id: int):
    for member in members:
        if member.id == member_id:
            return member
    raise HTTPException(status_code=404, detail="Member not found")
