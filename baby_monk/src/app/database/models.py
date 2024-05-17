# SQLAlchemy를 사용해 데이터베이스 모델 정의

from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql.sqltypes import Enum, Date, Time, JSON

from app.commons.genderenum import GenderEnum
from app.schema.request import CreateMemberRequest, CreateMemberSajuInfoRequest

Base = declarative_base()


class Member(Base):
    __tablename__ = "members"

    id = Column(Integer, primary_key=True, autoincrement=True)  # 회원 아이디
    user_id = Column(String(20))  # 성별
    email = Column(String(30))
    password = Column(String(255))
    name = Column(String(20))
    gender = Column(Enum(GenderEnum))
    birth_date = Column(Date)  # 태어난 날짜 연월일
    birth_time = Column(Time)  # 태어난 시간

    saju_info = relationship("MemberSajuInfo", back_populates="member")

    def __repr__(self):
        return f"Member(id={self.id})"

    @classmethod
    def create(cls, request: CreateMemberRequest, password: str) -> "Member":
        return cls(
            user_id=request.user_id,
            email=request.email,
            password=password,
            name=request.name,
            gender=request.gender,
            birth_date=request.birth_date,
            birth_time=request.birth_time
        )

    @classmethod
    def update(cls, request: CreateMemberRequest) -> "Member":
        return cls(
            email=request.email,
            password=request.password,
            name=request.name,
            gender=request.gender,
            birth_date=request.birth_date,
            birth_time=request.birth_time
        )


class MemberSajuInfo(Base):
    __tablename__ = "member_saju_info"

    user_id = Column(String, ForeignKey('members.user_id'), primary_key=True)
    saju_text = Column(String)
    eight = Column(JSON)

    member = relationship("Member", back_populates="saju_info")

    @classmethod
    def create_saju(
            cls, request: CreateMemberSajuInfoRequest, user_id: str
    ) -> "MemberSajuInfo":
        return cls(
            user_id=user_id,
            eight=request.eight,
            saju_text=None
        )


class SajuInfo(Base):
    __tablename__ = "saju_info"
    id = Column(Integer, primary_key=True, index=True)
    letter = Column(String)
    mean = Column(String, nullable=True)
    locate = Column(Integer, nullable=True)
    color = Column(String, nullable=True)
    count = Column(Integer, nullable=True)
