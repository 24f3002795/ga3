from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from main import router

app = FastAPI(
    title="GA3 API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

