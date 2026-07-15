import os
from collections.abc import Awaitable, Callable
from time import perf_counter
from typing import Any, TypeVar

from loguru import logger
from openai import AsyncOpenAI


DEFAULT_OLLAMA_MODEL = "qwen3.5:2b"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"
DEFAULT_OLLAMA_TIMEOUT_SECONDS = 600.0
DEFAULT_OLLAMA_MAX_RETRIES = 1
DEFAULT_SUMMARY_CHUNK_SIZE = 6000
SUMMARY_SYSTEM_PROMPT = """你是一名严谨的会议与课程笔记整理专家。
请把转写原文整理为层次清晰的 Markdown 笔记，并遵守以下规则：
1. 只使用原文信息，不推测、不补充外部知识；无法确认的内容明确标注“原文未明确”。
2. 保留人名、数字、日期、专有名词、结论和限制条件，不擅自修正疑似识别错误。
3. 优先输出内容摘要、主题要点、决策或结论、待办事项、未决问题；没有内容的章节可以省略。
4. 待办事项仅在原文明确时记录负责人和期限，不得自行分配。
5. 合并重复口语和赘词，但不要为了简短而遗漏关键事实。"""
PARTIAL_SUMMARY_SYSTEM_PROMPT = """提炼当前转写片段，保留所有事实、观点、数字、日期、
专有名词、结论、限制条件、待办和未决问题。去除口语重复，但不要补充原文没有的信息，
也不要擅自修正疑似识别错误。"""
CONDENSE_SUMMARY_SYSTEM_PROMPT = """压缩合并当前一组片段笔记，去除重复表达，完整保留事实、
数字、日期、专有名词、结论、限制条件、待办和未决问题。不要补充原文没有的信息。"""

Number = TypeVar("Number", int, float)


def _prompt_messages(system_prompt: str, user_content: str) -> list[dict[str, str]]:
    """统一构造摘要消息，避免不同层级手写消息结构时出现角色或字段漂移。"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def _get_numeric_setting(
    name: str,
    default: Number,
    parser: Callable[[str], Number],
    is_valid: Callable[[Number], bool],
) -> Number:
    """读取并校验数值环境变量，配置无效时记录原值并使用安全默认值。"""

    raw_value = os.getenv(name, str(default))
    try:
        value = parser(raw_value)
        if not is_valid(value):
            raise ValueError
        return value
    except (TypeError, ValueError):
        # 统一日志格式能直接定位配置名，同时避免三个业务 getter 重复维护相同异常分支。
        logger.warning(
            "ollama :: invalid numeric setting, name: {}, value: {}, fallback: {}",
            name,
            raw_value,
            default,
        )
        return default


def _get_timeout_seconds() -> float:
    return _get_numeric_setting(
        "OLLAMA_TIMEOUT_SECONDS",
        DEFAULT_OLLAMA_TIMEOUT_SECONDS,
        float,
        lambda value: value > 0,
    )


def _get_max_retries() -> int:
    return _get_numeric_setting(
        "OLLAMA_MAX_RETRIES",
        DEFAULT_OLLAMA_MAX_RETRIES,
        int,
        lambda value: value >= 0,
    )


def _get_summary_chunk_size() -> int:
    # 片段过小会产生大量串行请求；1000 字符下限保留原有保护策略。
    return _get_numeric_setting(
        "OLLAMA_SUMMARY_CHUNK_SIZE",
        DEFAULT_SUMMARY_CHUNK_SIZE,
        int,
        lambda value: value >= 1000,
    )


def _split_summary_text(text: str, max_chars: int) -> list[str]:
    """按段落和句末切分转写文本，并确保每个片段都有明确上限。"""

    chunks: list[str] = []
    current = ""
    for paragraph in text.splitlines():
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        paragraph_parts = _split_oversized_paragraph(paragraph, max_chars)
        for part in paragraph_parts:
            separator = "\n" if current else ""
            if len(current) + len(separator) + len(part) <= max_chars:
                current = f"{current}{separator}{part}"
                continue
            chunks.append(current)
            current = part

    if current:
        chunks.append(current)
    return chunks


def _split_oversized_paragraph(text: str, max_chars: int) -> list[str]:
    """优先在后半段的句末切分超长段落，没有标点时再按字符上限切分。"""

    parts: list[str] = []
    remaining = text
    sentence_endings = "。！？；.!?;"
    while len(remaining) > max_chars:
        # 只在片段后半段寻找句末，可以避免为了命中很早的标点而产生大量过短请求。
        # ASR 可能完全没有标点，因此始终保留 max_chars 作为确定性的硬切回退点。
        search_start = max_chars // 2
        boundary = max(
            remaining.rfind(mark, search_start, max_chars + 1)
            for mark in sentence_endings
        )
        split_at = boundary + 1 if boundary >= search_start else max_chars
        parts.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()

    if remaining:
        parts.append(remaining)
    return parts


def _create_ollama_client(
    base_url: str,
    timeout_seconds: float,
    max_retries: int,
) -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url=base_url,
        api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
        timeout=timeout_seconds,
        max_retries=max_retries,
    )


async def chat_with_ollama(
    messages: list[dict[str, Any]],
    callback: Callable[[str], Awaitable[None]] | None = None,
    max_tokens: int | None = None,
    client: AsyncOpenAI | None = None,
) -> str:
    base_url = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
    model = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL).strip()
    if not model:
        # 空模型名通常来自 Compose 变量或手工编辑错误。回退到明确版本，既避免请求必然失败，
        # 也保证日志中的实际模型与文档默认值一致，便于排查不同部署之间的质量差异。
        model = DEFAULT_OLLAMA_MODEL
    timeout_seconds = _get_timeout_seconds()
    max_retries = _get_max_retries()
    started_at = perf_counter()
    logger.info(
        "ollama :: start chat, base_url: {}, model: {}, message_count: {}, "
        "timeout_seconds: {}, max_retries: {}",
        base_url,
        model,
        len(messages),
        timeout_seconds,
        max_retries,
    )

    try:
        request_options = {
            "model": model,
            "stream": True,
            "temperature": 0.1,
            "messages": messages,
        }
        # 中间摘要只承担信息压缩，不需要无限输出。限制输出可以控制后续合并输入的体积；
        # 最终笔记和普通问答不传该参数，继续由模型自身配置决定输出长度。
        if max_tokens is not None:
            request_options["max_tokens"] = max_tokens

        async def consume_stream(active_client: AsyncOpenAI) -> str:
            response = await active_client.chat.completions.create(**request_options)
            content_parts: list[str] = []
            async for chunk in response:
                # 流的首帧、结束帧或模型推理帧可能没有 choices，或者 content 为 None。
                # 过滤这些协议帧可以避免向 Chainlit 传入 None，也能防止字符串拼接抛出异常。
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content
                if not content:
                    continue

                if callback is not None:
                    await callback(content)
                content_parts.append(content)
            return "".join(content_parts)

        if client is None:
            # 普通问答独占一个客户端，请求结束后立即关闭；长文本摘要会从外层传入共享客户端。
            async with _create_ollama_client(
                base_url,
                timeout_seconds,
                max_retries,
            ) as owned_client:
                full_content = await consume_stream(owned_client)
        else:
            full_content = await consume_stream(client)
        logger.info(
            "ollama :: complete chat, model: {}, response_length: {}, elapsed_seconds: {:.2f}",
            model,
            len(full_content),
            perf_counter() - started_at,
        )
        return full_content
    except Exception:
        # 不记录 messages 正文，避免音频转写、用户问题等敏感内容进入日志。
        # 模型、服务地址和耗时足以帮助判断是连接失败、模型不可用还是响应超时。
        logger.exception(
            "ollama :: chat failed, base_url: {}, model: {}, elapsed_seconds: {:.2f}",
            base_url,
            model,
            perf_counter() - started_at,
        )
        raise


async def summarize_with_ollama(
    text: str,
    callback: Callable[[str], Awaitable[None]] | None = None,
) -> str:
    """整理转写文本；长文本先分别提炼，再合并为一份最终笔记。"""

    base_url = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
    # 一次摘要可能包含数十个串行请求。共用客户端可以复用 HTTP 连接池和 keep-alive 连接，
    # 同时仍由异步上下文管理器在整项任务结束时可靠关闭底层网络资源。
    async with _create_ollama_client(
        base_url,
        _get_timeout_seconds(),
        _get_max_retries(),
    ) as client:
        return await _summarize_with_client(text, callback, client)


async def _summarize_with_client(
    text: str,
    callback: Callable[[str], Awaitable[None]] | None,
    client: AsyncOpenAI,
) -> str:

    chunk_size = _get_summary_chunk_size()
    chunks = _split_summary_text(text, chunk_size)
    if not chunks:
        return ""

    logger.info(
        "ollama :: start summary, source_length: {}, chunk_count: {}",
        len(text),
        len(chunks),
    )
    if len(chunks) == 1:
        return await chat_with_ollama(
            _prompt_messages(SUMMARY_SYSTEM_PROMPT, chunks[0]),
            callback=callback,
            client=client,
        )

    partial_summaries: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        # 中间结果不向页面流式输出，否则用户会先看到多份重复片段。只在最终合并阶段输出完整笔记。
        partial_summary = await chat_with_ollama(
            _prompt_messages(
                PARTIAL_SUMMARY_SYSTEM_PROMPT,
                f"这是转写的第 {index}/{len(chunks)} 个片段：\n\n{chunk}",
            ),
            max_tokens=1024,
            client=client,
        )
        partial_summaries.append(partial_summary)
        logger.info(
            "ollama :: summarized chunk, chunk_index: {}, chunk_count: {}, "
            "source_length: {}, summary_length: {}",
            index,
            len(chunks),
            len(chunk),
            len(partial_summary),
        )

    merged_content = "\n\n".join(
        f"## 片段 {index}\n{summary}"
        for index, summary in enumerate(partial_summaries, start=1)
    )

    # 很长的录音可能产生多份较长的片段摘要，直接拼接后仍会超过上下文窗口。
    # 每轮按同一字符上限分组压缩，通常一到两轮即可收敛；设置轮次上限可避免异常模型
    # 不压缩反而持续扩写时形成无限请求。
    for merge_round in range(1, 5):
        merge_chunks = _split_summary_text(merged_content, chunk_size)
        if len(merge_chunks) == 1:
            break

        logger.info(
            "ollama :: condense summary groups, merge_round: {}, group_count: {}",
            merge_round,
            len(merge_chunks),
        )
        condensed_summaries = []
        for merge_chunk in merge_chunks:
            condensed_summaries.append(
                await chat_with_ollama(
                    _prompt_messages(
                        CONDENSE_SUMMARY_SYSTEM_PROMPT,
                        merge_chunk,
                    ),
                    max_tokens=1024,
                    client=client,
                )
            )
        merged_content = "\n\n".join(condensed_summaries)
    if len(_split_summary_text(merged_content, chunk_size)) != 1:
        logger.error(
            "ollama :: summary merge did not converge, content_length: {}",
            len(merged_content),
        )
        raise RuntimeError("摘要分组在限定轮次内未能完成合并")

    return await chat_with_ollama(
        _prompt_messages(SUMMARY_SYSTEM_PROMPT, merged_content),
        callback=callback,
        client=client,
    )
