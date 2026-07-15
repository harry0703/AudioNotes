import hmac
import os
import sys
import uuid
from io import BytesIO

import chainlit as cl
from chainlit.types import InputAudioChunk
from dotenv import load_dotenv
from loguru import logger

# 先加载项目根目录下的 .env，再导入业务模块，确保本地部署和 Docker 部署使用一致的配置。
# SQLite 不再需要连接参数，但 ASR、Ollama 等服务仍会在模块初始化时读取环境变量。
load_dotenv()

from app.services import data_layer  # noqa: E402
from app.services.asr_funasr import NoSpeechDetectedError, funasr  # noqa: E402
from app.services.ollama import chat_with_ollama, summarize_with_ollama  # noqa: E402
from app.utils import utils  # noqa: E402


def configure_logging() -> None:
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_dir = utils.storage_dir("logs", create=True)
    log_file = os.path.join(log_dir, "log.log")

    # 默认 Loguru handler 被移除后，原实现只写文件，导致 docker logs 无法查看应用状态。
    # stderr 会被 Docker 正常收集；文件 handler 用于保留有限期的历史记录，便于排查跨天问题。
    logger.remove()
    logger.add(
        sys.stderr,
        level=log_level,
        enqueue=True,
        backtrace=False,
        diagnose=False,
    )
    logger.add(
        log_file,
        level=log_level,
        rotation="100 MB",
        retention="14 days",
        compression="zip",
        encoding="utf-8",
        enqueue=True,
        # 异常诊断模式可能输出局部变量，其中可能包含完整转写或用户问题，因此显式关闭。
        backtrace=False,
        diagnose=False,
    )
    logger.info("logging :: initialized, level: {}, directory: {}", log_level, log_dir)


configure_logging()

if os.getenv("USERNAME", "admin") == "admin" and os.getenv("PASSWORD", "admin") == "admin":
    # 保留默认值便于本地首次体验，但必须在日志中明确提示，避免使用者误把弱凭据部署到公网。
    logger.warning(
        "auth :: default credentials are active; change USERNAME and PASSWORD before deployment"
    )

# 固定的开发密钥只用于降低本地首次体验门槛。公网复用该值会让不同部署共享签名密钥，
# 一旦密钥泄露，攻击者可能伪造登录令牌，因此启动时必须留下可检索的安全告警。
if os.getenv("CHAINLIT_AUTH_SECRET") == (
    "local-development-secret-change-before-deployment"
):
    logger.warning(
        "auth :: development JWT secret is active; change CHAINLIT_AUTH_SECRET before deployment"
    )

data_layer.init()


def get_max_upload_size_mb() -> int:
    raw_value = os.getenv("MAX_UPLOAD_SIZE_MB", "2048")
    try:
        max_size_mb = int(raw_value)
        if max_size_mb <= 0:
            raise ValueError
        return max_size_mb
    except ValueError:
        # 配置错误时使用安全默认值而不是让所有用户无法上传，同时记录原始配置便于定位。
        logger.warning(
            "upload :: invalid MAX_UPLOAD_SIZE_MB, value: {}, fallback: 2048",
            raw_value,
        )
        return 2048


@cl.password_auth_callback
def password_auth_callback(username: str, password: str):
    expected_username = os.getenv("USERNAME", "admin")
    expected_password = os.getenv("PASSWORD", "admin")
    # compare_digest 避免普通字符串比较在首个不同字符处提前返回。用户名和密码都使用常量时间比较，
    # 并且不在失败日志中记录输入值，减少远程登录接口泄露凭据特征的机会。
    credentials_match = hmac.compare_digest(
        username.encode("utf-8"),
        expected_username.encode("utf-8"),
    ) and hmac.compare_digest(
        password.encode("utf-8"),
        expected_password.encode("utf-8"),
    )
    if credentials_match:
        return cl.User(
            identifier=expected_username,
            metadata={"role": "admin", "provider": "credentials"},
        )
    return None


@cl.on_chat_start
async def on_chat_start():
    files = None
    while files is None:
        msg = cl.AskFileMessage(
            content="请上传一个**音频/视频**文件",
            accept=["audio/*", "video/*"],
            max_size_mb=get_max_upload_size_mb(),
            max_files=1,
        )
        files = await msg.send()
    file = files[0]

    msg = cl.Message(content="")

    try:
        # 上传流程只有一次识别和一次整理，直接按执行顺序展开可以减少无复用价值的嵌套函数，
        # 同时让异常处理边界更直观；stream_token 本身已满足摘要服务的异步回调接口。
        await msg.stream_token(f"文件 《{file.name}》 上传成功, 识别中...\n")
        asr_result = await funasr.transcribe_async(file.path)
        await msg.stream_token(f"## 识别结果 \n{asr_result}\n")
        await msg.stream_token("## 整理笔记\n\n")
        await summarize_with_ollama(asr_result, callback=msg.stream_token)
    except NoSpeechDetectedError:
        # 静音、背景噪声或无有效人声不是服务故障，单独提示用户可避免误导其检查 Ollama。
        logger.warning("workflow :: uploaded media contains no detectable speech")
        await msg.stream_token("\n\n> 未识别到有效语音，请检查音量和录音内容后重试。")
    except Exception:
        # 服务层已经记录模型和耗时信息，这里只记录处理阶段，不写入文件名或转写正文。
        logger.exception("workflow :: failed to process uploaded media")
        await msg.stream_token(
            "\n\n> 处理失败，请确认文件格式、ASR 模型和 Ollama 服务可用后重试。"
        )
    finally:
        # 即使识别或整理失败，也要发送已经生成的进度和错误提示，避免前端一直停在加载状态。
        await msg.send()


@cl.on_audio_chunk
# Chainlit 2.11 将浏览器上传的录音分片类型明确命名为 InputAudioChunk，且不再从
# chainlit 顶层导出旧 AudioChunk。使用类型的正式定义路径可避免模块加载阶段直接失败。
async def on_audio_chunk(chunk: InputAudioChunk):
    previous_buffer: BytesIO | None = cl.user_session.get("audio_buffer")
    if chunk.isStart:
        # 浏览器重连或用户快速重新录音时可能在上一次 on_audio_end 前再次收到开始事件。
        # 先关闭旧缓冲可避免内存泄漏，并以新录音为准恢复会话状态。
        if previous_buffer is not None:
            previous_buffer.close()
            logger.warning("audio :: replaced unfinished recording buffer")
        buffer = BytesIO()
        buffer.name = f"input_audio{utils.audio_extension_from_mime_type(chunk.mimeType)}"
        cl.user_session.set("audio_buffer", buffer)
        cl.user_session.set("audio_mime_type", chunk.mimeType)
    else:
        buffer = previous_buffer

    if buffer is None:
        # WebSocket 重连后首个分片不一定带 isStart。与其直接抛出 NoneType.write，
        # 这里创建恢复缓冲并记录一次告警，让本次录音仍有机会被识别。
        logger.warning("audio :: received chunk before start; recovering buffer")
        buffer = BytesIO()
        buffer.name = f"input_audio{utils.audio_extension_from_mime_type(chunk.mimeType)}"
        cl.user_session.set("audio_buffer", buffer)
        cl.user_session.set("audio_mime_type", chunk.mimeType)

    buffer.write(chunk.data)


@cl.on_audio_end
async def on_audio_end():
    audio_buffer: BytesIO | None = cl.user_session.get("audio_buffer")
    if audio_buffer is None:
        logger.warning("audio :: recording ended without a buffer")
        await cl.Message(content="未收到有效的录音数据，请重新录制。").send()
        return

    audio_mime_type = cl.user_session.get("audio_mime_type")
    audio_extension = utils.audio_extension_from_mime_type(audio_mime_type)
    # 麦克风录音只用于本次识别，不应暂存在可通过 /public 访问的附件目录中。
    file_path = os.path.join(
        utils.storage_dir("tmp", create=True),
        f"{uuid.uuid4()}{audio_extension}",
    )

    try:
        audio_buffer.seek(0)
        if audio_buffer.getbuffer().nbytes == 0:
            logger.warning("audio :: recording buffer is empty")
            await cl.Message(content="录音内容为空，请重新录制。").send()
            return
        with open(file_path, "wb") as f:
            f.write(audio_buffer.read())
        result = await funasr.transcribe_async(file_path)
    except NoSpeechDetectedError:
        logger.warning("audio :: microphone recording contains no detectable speech")
        await cl.Message(
            content="未识别到有效语音，请靠近麦克风并重新录制。"
        ).send()
        return
    except Exception:
        logger.exception("audio :: failed to transcribe microphone recording")
        await cl.Message(
            content="录音识别失败，请检查麦克风音频格式或稍后重试。"
        ).send()
        return
    finally:
        # 麦克风文件仅用于本次识别，不属于用户主动上传并持久化的附件。
        # 无论识别成功还是失败都释放内存并删除磁盘文件，避免长期运行后占满存储空间。
        audio_buffer.close()
        cl.user_session.set("audio_buffer", None)
        cl.user_session.set("audio_mime_type", None)
        try:
            os.remove(file_path)
        except FileNotFoundError:
            pass
        except OSError as error:
            logger.warning(
                "audio :: failed to remove temporary file, file_path: {}, error: {}",
                file_path,
                error,
            )

    await cl.Message(
        content=result,
        type="user_message",
    ).send()

    await chat()


async def chat():
    history = cl.chat_context.to_openai()

    msg = cl.Message(content="")
    messages = [
        {
            "role": "system",
            "content": (
                "你是一名严谨的笔记问答助手。只能根据音频识别结果和已整理的笔记回答，"
                "保留原文中的数字、日期和限制条件；如果上下文没有答案，请明确说明无法从录音确认，"
                "不要猜测或补充外部信息。"
            ),
        },
    ]
    # Chainlit 历史已经包含识别结果、整理笔记和当前提问。直接追加真实历史可以减少无效
    # token，也避免伪造的“请识别音频”指令在追问阶段与用户当前问题争夺模型注意力。
    messages.extend(history)

    # history 和 messages 包含完整音频转写、整理结果及用户问题，直接记录会泄露隐私，
    # 长音频还会快速撑大日志文件。消息数量足以判断上下文是否缺失或异常膨胀。
    logger.info(
        "chat :: start response, history_message_count: {}, request_message_count: {}",
        len(history),
        len(messages),
    )

    try:
        # Chainlit 的流式写入方法与服务层回调签名一致，无需再包装一层只做转发的函数。
        await chat_with_ollama(messages, callback=msg.stream_token)
    except Exception:
        logger.exception("chat :: failed to generate response")
        await msg.stream_token(
            "\n\n> 回答生成失败，请确认 Ollama 服务和模型可用后重试。"
        )
    finally:
        await msg.send()


@cl.on_message
async def on_message(_message: cl.Message):
    await chat()
