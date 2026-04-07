"""FastAPI entry point for the backend application."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import routes

app = FastAPI(title="Flood Risk System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes.router)


@app.get("/")
def read_root() -> dict[str, str]:
    """Basic health endpoint."""
    return {"message": "Flood Risk System backend is running."}
