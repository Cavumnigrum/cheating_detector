from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path

app = FastAPI(title="Phone Detection AI")

# Подключение статических файлов
static_path = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_path), name="static")

# Шаблоны
templates = Jinja2Templates(directory=static_path)

from app.api import endpoints

app.include_router(endpoints.router, prefix="/api")

@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
