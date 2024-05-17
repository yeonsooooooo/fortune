# 클라이언트에게 보낼 응답의 데이터 구조 정의

from datetime import date, time
from typing import List

from pydantic import BaseModel, ConfigDict

from app.commons.genderenum import GenderEnum


class MemberSchema(BaseModel):
    id: int
    user_id: str
    email: str
    password: str
    name: str
    gender: GenderEnum
    birth_date: date
    birth_time: time

    model_config = ConfigDict(from_attributes=True)


class MemberListSchema(BaseModel):
    members: List[MemberSchema]

    model_config = ConfigDict(from_attributes=True)


class MemberSajuInfoSchema(BaseModel):
    user_id: str
    saju_text: None
    eight: dict

    model_config = ConfigDict(from_attributes=True)


class JWTResponse(BaseModel):
    access_token: str
