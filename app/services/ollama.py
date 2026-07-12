import os

from loguru import logger
from openai import AsyncOpenAI


async def chat_with_ollama(messages: list[dict], callback=None):
    base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434/v1')
    model = os.getenv('OLLAMA_MODEL', 'qwen2:7b')
    logger.debug(f"chat with ollama: {base_url}, model: {model}")

    client = AsyncOpenAI(
        base_url=base_url,
        api_key=os.getenv('OLLAMA_API_KEY', 'ollama')
    )

    response = await client.chat.completions.create(
        model=model,
        stream=True,
        temperature=0.1,
        messages=messages
    )
    full_content = ""
    async for chunk in response:
        # 最后一个chunk的content为None，部分服务的chunk可能没有choices
        if not chunk.choices:
            continue
        content = chunk.choices[0].delta.content
        if not content:
            continue
        if callback:
            await callback(content)
        full_content += content
    return full_content
