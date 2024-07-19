import os

from loguru import logger
from openai import OpenAI


async def chat_with_ollama(messages: list[dict], callback=None):
    base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434/v1')
    model = os.getenv('OLLAMA_MODEL', 'qwen2:7b')
    logger.debug(f"chat with ollama: {base_url}, model: {model}")

    client = OpenAI(
        base_url=base_url,
        api_key=os.getenv('OLLAMA_API_KEY', 'ollama')
    )

    response = client.chat.completions.create(
        model=model,
        stream=True,
        temperature=0.1,
        messages=messages
    )
    full_content = ""
    for chunk in response:
        content = chunk.choices[0].delta.content
        await callback(content)
        full_content += content
    return full_content
