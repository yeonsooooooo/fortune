# 데이터베이스 작업을 위한 함수와 클래스 포함 데이터 접근 계층 (실제로 데이터베이스에 추가되는 로직)

from typing import List

from fastapi import Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.connection import get_db
from app.database.models import Member, MemberSajuInfo


class MemberRepository:
    def __init__(self, session: Session = Depends(get_db)):
        self.session = session

    # 전체 회원 정보 조회
    def get_members(self) -> List[Member]:
        return list(self.session.scalars(select(Member)))

    # 회원 정보 검색 (사용자 아이디로 검색)
    def get_member_by_member_user_id(self, member_user_id=str) -> Member | None:
        return self.session.scalar(select(Member).where(Member.user_id == member_user_id))

    # 회원 가입
    def create_member(self, member: Member) -> Member:
        self.session.add(instance=member)
        self.session.commit()  # db save
        self.session.refresh(instance=member)  # db read -> member_id 확정
        return member

    # 회원 사용자 아이디 중복 확인
    def is_member_user_id_duplicate(self, member_user_id: str) -> bool:
        existing_member_user_id = self.session.query(Member).filter(Member.user_id == member_user_id).first()
        return existing_member_user_id is not None

    def update_member(self, member_user_id: str, member: Member, update_data: dict):
        for key, value in update_data.items():
            if hasattr(member, key):
                setattr(member, key, value)
        self.session.commit()
        self.session.refresh(member)
        return member


class MemberSajuInfoRepository:
    def __init__(self, session: Session = Depends(get_db)):
        self.session = session

    def save_saju(self, member_saju: MemberSajuInfo):
        self.session.add(instance=member_saju)
        self.session.commit()
        self.session.refresh(instance=member_saju)
        return member_saju

    def is_member_user_id_duplicate(self, member_user_id: str) -> bool:
        existing_member_user_id = self.session.query(MemberSajuInfo).filter(
            MemberSajuInfo.user_id == member_user_id).first()
        return existing_member_user_id is not None

    def saju_to_text(self, member_saju: MemberSajuInfo, member_user_id: str) -> MemberSajuInfo:
        # 1. 회원 아이디에 맞는 eight 받아오기

        # 2. eight 글자 키에 맞춰서 cnt

        # 3. key==3 그리고 글자가 맞는 mean 추출해서 txt 추가

        # 4. 각각 음양오행 cnt 비교해서 설명 추가하기
        return
