from datetime import date, time

from pydantic import BaseModel


class CreateUserRequest(BaseModel):
    id: str
    name: str
    birth_date: date
    birth_time: time


