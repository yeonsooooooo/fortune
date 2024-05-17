from fastapi import FastAPI

from app.routers import members, saju

app = (FastAPI())
app.include_router(members.router)
app.include_router(saju.router)
