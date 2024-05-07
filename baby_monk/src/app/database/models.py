from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, index=True)  # 회원 아이디
    gender = Column(String)  # 성별
    birth_date = Column(DateTime)  # 태어난 날짜 연월일
    birth_time = Column(DateTime, Nullable=True)  # 태어난 시간
    user_saju_id = Column(Integer, ForeignKey('user_saju.id'))  # 사주 정보

    user_saju = relationship("User_Saju", back_populates="user")
#edsfsdfsf

class User_Saju(Base):
    __tablename__ = "user_saju"

    id = Column(Integer, primary_key=True, index=True)
    time_info1 = Column(String)  # 시주 1행
    time_relationInfo1 = Column(String)  # 시주 육친 1행
    time_info2 = Column(String)  # 시주 2행
    time_relationInfo2 = Column(String)  # 시주 육친 2행
    day_info1 = Column(String)  # 일주 1행
    day_relationInfo1 = Column(String)  # 일주 육친 1행
    day_info2 = Column(String)  # 일주 2행
    day_relationInfo2 = Column(String)  # 일주 육친 2행
    month_info1 = Column(String)  # 월주 1행
    month_relationInfo1 = Column(String)  # 월주 육친 1행
    month_info2 = Column(String)  # 월주 2행
    month_relationInfo2 = Column(String)  # 월주 육친 2행
    year_info1 = Column(String)  # 연주 1행
    year_relationInfo1 = Column(String)  # 연주 육친 1행
    year_info2 = Column(String)  # 연주 2행
    year_relationInfo2 = Column(String)  # 연주 육친 2행
    # 여기에 사주 정보와 관련된 필드를 추가
    # 웹 크롤링을 통해서 여기에 시주,일주,월주,년주 받기
    user = relationship("User", back_populates="user_saju", uselist=False)


class Saju_Info(Base):
    __tablename__ = "saju_info"

    letter = Column(String, primary_key=True, index=True)
    mean = Column(String, nullable=True)
    color = Column(String, nullable=True)
    count = Column(Integer, nullable=True)

