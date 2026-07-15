import os


# 浏览器录音常见 MIME 类型与 ffmpeg 可识别扩展名的映射。
# 不直接使用 MIME 子类型拼接路径，避免包含参数或异常字符时生成无效文件名。
_AUDIO_EXTENSION_BY_MIME_TYPE = {
    "audio/mp4": ".m4a",
    "audio/mpeg": ".mp3",
    "audio/ogg": ".ogg",
    "audio/wav": ".wav",
    "audio/webm": ".webm",
    "audio/x-m4a": ".m4a",
    "audio/x-wav": ".wav",
}


def audio_extension_from_mime_type(mime_type: str | None) -> str:
    normalized_mime_type = (mime_type or "").split(";", maxsplit=1)[0].strip().lower()
    return _AUDIO_EXTENSION_BY_MIME_TYPE.get(normalized_mime_type, ".audio")


def root_dir():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


def storage_dir(sub_dir: str = "", create: bool = False):
    d = os.path.join(root_dir(), "storage")
    if sub_dir:
        d = os.path.join(d, sub_dir)
    if create:
        # exist_ok 让并发请求首次创建同一目录时保持幂等，避免先判断再创建产生竞态。
        os.makedirs(d, exist_ok=True)

    return d


def upload_dir(sub_dir: str = ""):
    # Chainlit 2.x 只原生提供 /public 路由，因此持久化附件必须放在 public/uploads 下。
    # Docker Compose 会把宿主机 storage/uploads 挂载到该目录，源码目录本身不会保存用户数据。
    d = os.path.join(root_dir(), "public", "uploads")
    if sub_dir:
        d = os.path.join(d, sub_dir)
    os.makedirs(d, exist_ok=True)
    return d
