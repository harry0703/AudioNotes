import asyncio
import os
from time import perf_counter

from funasr import AutoModel
from loguru import logger


DEFAULT_ASR_MODEL = "FunAudioLLM/Fun-ASR-Nano-2512"


def _get_speaker_diarization_enabled() -> bool:
    raw_value = os.getenv("ASR_ENABLE_SPEAKER_DIARIZATION", "false").strip().lower()
    if raw_value in {"1", "true", "yes", "on"}:
        return True
    if raw_value not in {"0", "false", "no", "off", ""}:
        # 配置拼写错误时采用低资源默认值，并记录原值帮助定位说话人标签未生效的原因。
        logger.warning(
            "funasr :: invalid speaker diarization setting, value: {}, fallback: false",
            raw_value,
        )
    return False


class NoSpeechDetectedError(RuntimeError):
    """音频可以正常解码，但 ASR 没有识别到有效文本。"""


class FunASR:
    def __init__(self):
        self.__model = None
        self.__hotwords = ""
        self.__uses_fun_asr_nano = False
        self.__transcribe_lock = asyncio.Lock()

    def __init_model(self):
        if self.__model:
            return

        model_name = os.getenv("ASR_MODEL", DEFAULT_ASR_MODEL).strip()
        if not model_name:
            model_name = DEFAULT_ASR_MODEL
        device = os.getenv("ASR_DEVICE", "").strip()
        self.__hotwords = os.getenv("ASR_HOTWORDS", "").strip()
        self.__uses_fun_asr_nano = "fun-asr-nano" in model_name.lower()
        speaker_diarization_enabled = _get_speaker_diarization_enabled()

        # 设备留空时沿用 FunASR 的自动选择逻辑，便于同一份配置运行在 CPU、CUDA 等环境。
        # 只有用户显式设置 ASR_DEVICE 时才传给模型，避免默认值改变现有部署的硬件行为。
        model_options = {
            "model": model_name,
            "vad_model": "fsmn-vad",
            # Fun-ASR-Nano 官方示例将 VAD 单段限制为 30 秒，避免长语音片段让注意力计算
            # 占用过多内存；该限制对 Paraformer 同样安全，也更适合本项目的长录音场景。
            "vad_kwargs": {"max_single_segment_time": 30000},
            "log_level": "error",
            # hub 表示模型仓库，ms 为 ModelScope，hf 为 Hugging Face。
            "hub": "ms",
        }
        # Web 主流程只消费纯文本，默认加载 cam++ 不会改善文字准确率，却会增加模型下载、
        # 内存和推理耗时。需要 SRT 说话人标签的部署可以显式开启，功能仍保持可用。
        if speaker_diarization_enabled:
            model_options["spk_model"] = "cam++"
        if not self.__uses_fun_asr_nano:
            # Fun-ASR Nano 官方推理流程本身直接输出可读文本，无需再加载 1GB 级 ct-punc。
            # 额外标点模型会让默认组合在 8GB Docker Desktop 中加载时被 OOM 杀死；
            # Paraformer 等传统模型仍依赖独立标点恢复，因此只为经典配置保留。
            model_options["punc_model"] = "ct-punc"
        if device:
            model_options["device"] = device

        logger.info(
            "funasr :: init model, model: {}, inference_profile: {}, device: {}, "
            "hotword_enabled: {}, speaker_diarization_enabled: {}",
            model_name,
            "fun-asr-nano" if self.__uses_fun_asr_nano else "classic",
            device or "auto",
            bool(self.__hotwords),
            speaker_diarization_enabled,
        )
        self.__model = AutoModel(**model_options)
        logger.info("funasr :: init model complete, model: {}", model_name)

    def __convert_time_to_srt_format(self, time_in_milliseconds):
        hours = time_in_milliseconds // 3600000
        time_in_milliseconds %= 3600000
        minutes = time_in_milliseconds // 60000
        time_in_milliseconds %= 60000
        seconds = time_in_milliseconds // 1000
        time_in_milliseconds %= 1000

        return f"{hours:02}:{minutes:02}:{seconds:02},{time_in_milliseconds:03}"

    def __text_to_srt(
        self,
        idx,
        speaker_id,
        msg,
        start_milliseconds,
        end_milliseconds,
    ) -> str:
        start_time = self.__convert_time_to_srt_format(start_milliseconds)
        end_time = self.__convert_time_to_srt_format(end_milliseconds)
        speaker_prefix = f"[说话人 {speaker_id}] " if speaker_id is not None else ""

        # SRT 每个字幕块使用从 1 开始的编号、时间范围和正文，块末保留一个换行。
        # 说话人信息直接放入正文，可被常见播放器正常显示，也方便后续生成会议笔记。
        return f"{idx}\n{start_time} --> {end_time}\n{speaker_prefix}{msg}\n"

    def transcribe(self, audio_file: str, output_type: str = "txt"):
        self.__init_model()
        started_at = perf_counter()
        file_extension = os.path.splitext(audio_file)[1].lower() or "unknown"
        # 上传路径可能包含用户文件名或会话标识。日志只保留格式、结果长度和耗时，
        # 这些信息足以定位解码失败、异常慢文件和空识别，同时避免泄露文件身份。
        logger.info(
            "funasr :: start transcribe, file_extension: {}, output_type: {}",
            file_extension,
            output_type,
        )

        generate_options = {"input": audio_file}
        if self.__uses_fun_asr_nano:
            # LLM-ASR 模型按文件批处理并需要独立 cache；它的热词接口接收字符串列表。
            # 与传统模型分开组装参数，可保留用户通过 ASR_MODEL 回退 Paraformer 的兼容性。
            generate_options.update({"cache": {}, "batch_size": 1})
            if self.__hotwords:
                generate_options["hotwords"] = self.__hotwords.split()
        else:
            generate_options["batch_size_s"] = 300
            # 传统 FunASR 模型使用单个 hotword 字符串，不能直接复用 Nano 的列表参数。
            if self.__hotwords:
                generate_options["hotword"] = self.__hotwords
        res = self.__model.generate(**generate_options)
        if (
            not res
            or not isinstance(res[0], dict)
            or not isinstance(res[0].get("text"), str)
        ):
            # 不记录模型原始响应，其中可能包含转写片段；类型和数量足以判断模型适配问题。
            logger.error(
                "funasr :: invalid model response, response_type: {}, response_count: {}",
                type(res).__name__,
                len(res) if hasattr(res, "__len__") else "unknown",
            )
            raise RuntimeError("ASR 模型返回格式异常")

        text = res[0]["text"].strip()
        if not text:
            logger.warning(
                "funasr :: no speech detected, file_extension: {}, elapsed_seconds: {:.2f}",
                file_extension,
                perf_counter() - started_at,
            )
            raise NoSpeechDetectedError("音频中未检测到有效语音")
        logger.info(
            "funasr :: complete transcribe, file_extension: {}, text_length: {}, "
            "elapsed_seconds: {:.2f}",
            file_extension,
            len(text),
            perf_counter() - started_at,
        )
        if output_type == "srt":
            sentences = res[0]["sentence_info"]
            subtitles = []

            for idx, sentence in enumerate(sentences, start=1):
                sub = self.__text_to_srt(
                    idx,
                    sentence.get("spk"),
                    sentence["text"],
                    sentence["start"],
                    sentence["end"],
                )
                subtitles.append(sub)

            return "\n".join(subtitles)
        return text

    async def transcribe_async(self, audio_file: str, output_type: str = "txt"):
        # FunASR 的模型加载和 generate() 都是同步的 CPU/GPU 密集型操作。如果直接在 Chainlit
        # 回调中执行，会阻塞 asyncio 事件循环，使页面流式消息、心跳和其他用户请求全部停顿。
        # asyncio.to_thread() 将同步推理移到工作线程，同时保持现有 transcribe() 接口可复用。
        #
        # 当前服务复用同一个模型实例。模型首次加载和推理过程不保证线程安全，因此使用异步锁
        # 串行化推理任务；这适合单模型本地部署，可避免并发加载模型或同时占满显存导致进程崩溃。
        async with self.__transcribe_lock:
            return await asyncio.to_thread(self.transcribe, audio_file, output_type)


funasr = FunASR()
