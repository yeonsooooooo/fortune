# 클라이언트에게 받은 요청의 데이터 구조 정리

from datetime import date, time
from typing import Optional

from pydantic import BaseModel

from app.commons.genderenum import GenderEnum


# 회원가입 요청
class CreateMemberRequest(BaseModel):
    user_id: str
    email: str
    password: str
    name: str
    gender: GenderEnum
    birth_date: date
    birth_time: time


# 회원정보 수정
class UpdateMemberRequest(BaseModel):
    email: Optional[str] = None
    password: Optional[str] = None
    name: Optional[str] = None
    gender: Optional[GenderEnum] = None
    birth_date: Optional[date] = None
    birth_time: Optional[time] = None


# 로그인
class LogInRequest(BaseModel):
    member_user_id: str
    password: str


# 사주팔자 정보 받기
class CreateMemberSajuInfoRequest(BaseModel):
    eight: dict




