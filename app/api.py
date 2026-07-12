import asyncio
import os
import uuid

from chainlit.server import app
from fastapi import Depends, File, Form, HTTPException, UploadFile
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from loguru import logger
from openai import AsyncOpenAI

from app.services.asr_funasr import funasr
from app.utils import utils

# REST API 通过 AUDIONOTES_API_TOKEN 环境变量启用。
# 未配置时接口默认关闭，避免未授权访问。
security = HTTPBearer(auto_error=False)


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = os.getenv("AUDIONOTES_API_TOKEN")
    if not token:
        raise HTTPException(
            status_code=503,
            detail="API is disabled. Set AUDIONOTES_API_TOKEN to enable it.",
        )
    if credentials is None or credentials.credentials != token:
        raise HTTPException(status_code=401, detail="Invalid or missing API token.")


async def summarize(text: str) -> str:
    client = AsyncOpenAI(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
    )
    response = await client.chat.completions.create(
        model=os.getenv("OLLAMA_MODEL", "qwen2:7b"),
        temperature=0.1,
        messages=[
            {"role": "system", "content": "你是一名笔记整理专家，根据用户提供的内容，整理出一份内容详尽的结构化的笔记"},
            {"role": "user", "content": text},
        ],
    )
    return response.choices[0].message.content or ""


@app.post("/api/v1/transcriptions", dependencies=[Depends(verify_token)])
async def create_transcription(
    file: UploadFile = File(...),
    with_notes: bool = Form(True),
):
    """上传一个音频/视频文件，返回识别结果，可选同时返回大模型整理的结构化笔记"""
    extname = os.path.splitext(file.filename or "")[1] or ".bin"
    file_path = os.path.join(utils.upload_dir(), f"{uuid.uuid4()}{extname}")
    with open(file_path, "wb") as f:
        f.write(await file.read())
    logger.info(f"api :: received file {file.filename} -> {file_path}")

    loop = asyncio.get_event_loop()
    try:
        transcript = await loop.run_in_executor(None, funasr.transcribe, file_path)
    except Exception as e:
        logger.exception("api :: transcription failed")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

    notes = None
    if with_notes:
        try:
            notes = await summarize(transcript)
        except Exception as e:
            logger.exception("api :: summarization failed")
            raise HTTPException(status_code=502, detail=f"Summarization failed: {e}")

    return {
        "filename": file.filename,
        "transcript": transcript,
        "notes": notes,
    }
