import os
import uuid
import asyncio
import chainlit as cl
from io import BytesIO
from chainlit import ThreadDict
from chainlit.element import ElementBased
from loguru import logger
from app.services import data_layer
from app.services.asr_funasr import funasr
from app.services.ollama import chat_with_ollama

from app.utils import utils

# load environment variables
from dotenv import load_dotenv

load_dotenv()

logger.remove()
logger.add(f"{utils.storage_dir('logs')}/log.log", rotation="500 MB")

data_layer.init()


@cl.password_auth_callback
def password_auth_callback(username: str, password: str):
    u = os.getenv("USERNAME", "admin")
    p = os.getenv("PASSWORD", "admin")
    if (username, password) == (u, p):
        return cl.User(
            identifier="admin", metadata={"role": "admin", "provider": "credentials"}
        )
    else:
        return None


@cl.on_chat_start
async def on_chat_start():
    files = None
    while files == None:
        msg = cl.AskFileMessage(
            content="请上传一个**音频/视频**文件",
            # accept=["audio/*", "video/*"],
            accept=["*/*"],
            max_size_mb=10240,
        )
        files = await msg.send()
    file = files[0]

    msg = cl.Message(content="")

    async def transcribe_file(uploaded_file):
        await msg.stream_token(f"文件 《{uploaded_file.name}》 上传成功, 识别中...\n")
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, funasr.transcribe, uploaded_file.path)
        await msg.stream_token(f"## 识别结果 \n{result}\n")
        return result

    async def summarize_notes(text):
        messages = [
            {"role": "system", "content": "你是一名笔记整理专家，根据用户提供的内容，整理出一份内容详尽的结构化的笔记"},
            {"role": "user", "content": text},
        ]

        async def on_message(content):
            await msg.stream_token(content)

        await msg.stream_token("## 整理笔记\n\n")
        await chat_with_ollama(messages, callback=on_message)

    asr_result = await transcribe_file(file)
    await summarize_notes(asr_result)
    await msg.send()


@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.AudioChunk):
    if chunk.isStart:
        buffer = BytesIO()
        buffer.name = f"input_audio.{chunk.mimeType.split('/')[1]}"
        cl.user_session.set("audio_buffer", buffer)
        cl.user_session.set("audio_mime_type", chunk.mimeType)
    cl.user_session.get("audio_buffer").write(chunk.data)


@cl.on_audio_end
async def on_audio_end(elements: list[ElementBased]):
    audio_buffer: BytesIO = cl.user_session.get("audio_buffer")
    audio_buffer.seek(0)
    file_path = f"{utils.upload_dir()}/{str(uuid.uuid4())}.wav"
    with open(file_path, "wb") as f:
        f.write(audio_buffer.read())

    result = funasr.transcribe(file_path)
    await cl.Message(
        content=result,
        type="user_message",
    ).send()

    await chat()


async def chat():
    history = cl.chat_context.to_openai()
    logger.info(history)

    msg = cl.Message(content="")
    messages = [
        {"role": "system", "content": "你是一名笔记整理专家，严格根据音频识别的结果和整理的笔记内容，回答用户的问题。"},
        {"role": "user", "content": "请识别这段音频文件并且整理成结构化的笔记"},  # 无实际意义，用于补充缺失的user消息
    ]
    messages.extend(history)  # history 中包含了用户的提问
    logger.info(messages)

    async def on_message(content):
        await msg.stream_token(content)

    await chat_with_ollama(messages, callback=on_message)
    await msg.send()


@cl.on_message
async def on_message(message: cl.Message):
    await chat()


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    pass
