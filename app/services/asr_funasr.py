from funasr import AutoModel
from loguru import logger


class FunASR:
    def __init__(self):
        self.__model = None

    def __init_model(self):
        if self.__model:
            return

        logger.debug("funasr :: init model start")
        self.__model = AutoModel(model="paraformer-zh",
                                 vad_model="fsmn-vad",
                                 punc_model="ct-punc",
                                 spk_model="cam++",
                                 log_level="error",
                                 hub="ms"  # hub：表示模型仓库，ms为选择modelscope下载，hf为选择huggingface下载。
                                 )
        logger.debug("funasr :: init model complete")

    def __convert_time_to_srt_format(self, time_in_milliseconds):
        hours = time_in_milliseconds // 3600000
        time_in_milliseconds %= 3600000
        minutes = time_in_milliseconds // 60000
        time_in_milliseconds %= 60000
        seconds = time_in_milliseconds // 1000
        time_in_milliseconds %= 1000

        return f"{hours:02}:{minutes:02}:{seconds:02},{time_in_milliseconds:03}"

    def __text_to_srt(self, idx, speaker_id, msg, start_microseconds, end_microseconds) -> str:
        start_time = self.__convert_time_to_srt_format(start_microseconds)
        end_time = self.__convert_time_to_srt_format(end_microseconds)

        msg = f"{msg}"
        srt = """%d
%s --> %s
%s
            """ % (
            idx,
            start_time,
            end_time,
            msg,
        )
        return srt

    def transcribe(self, audio_file: str, output_type: str = "txt"):
        self.__init_model()
        logger.info(f"funasr :: start transcribe audio file: {audio_file}")

        res = self.__model.generate(input=audio_file, batch_size_s=300)
        text = res[0]['text']
        logger.info(f"funasr :: complete transcribe audio file: {audio_file}")
        if output_type == "srt":
            sentences = res[0]['sentence_info']
            subtitles = []

            for idx, sentence in enumerate(sentences):
                sub = self.__text_to_srt(idx, sentence['spk'], sentence['text'], sentence['start'], sentence['end'])
                subtitles.append(sub)

            return "\n".join(subtitles)
        return text

    # def transcribe(audio_file):
    #     logger.info(f"funasr :: start transcribe audio file: {audio_file}")
    #
    #     res = model.generate(input=audio_file, batch_size_s=300)
    #     text = res[0]['text']
    #     subtitle_file = f"{audio_file}.funasr.txt"
    #     with open(subtitle_file, "w") as f:
    #         f.write(text)
    #
    #     sentences = res[0]['sentence_info']
    #     subtitles = []
    #
    #     for idx, sentence in enumerate(sentences):
    #         sub = text_to_srt(idx, sentence['spk'], sentence['text'], sentence['start'], sentence['end'])
    #         subtitles.append(sub)
    #
    #     subtitle_file = f"{audio_file}.funasr.srt"
    #     with open(subtitle_file, "w") as f:
    #         f.write("\n".join(subtitles))
    #     logger.info(f"funasr :: complete transcribe audio file: {audio_file}")
    #
    # def transcribe_streaming(audio_file):
    #     chunk_size = [0, 10, 5]  # [0, 10, 5] 600ms, [0, 8, 4] 480ms
    #     encoder_chunk_look_back = 4  # number of chunks to lookback for encoder self-attention
    #     decoder_chunk_look_back = 1  # number of encoder chunks to lookback for decoder cross-attention
    #
    #     model = AutoModel(model="paraformer-zh-streaming",
    #                       # vad_model="fsmn-vad",
    #                       # punc_model="ct-punc",
    #                       # spk_model="cam++",
    #                       log_level="error", hub="ms")
    #
    #     speech, sample_rate = soundfile.read(audio_file)
    #     chunk_stride = chunk_size[1] * 960  # 600ms
    #
    #     cache = {}
    #     total_chunk_num = int(len(speech - 1) / chunk_stride + 1)
    #     for i in range(total_chunk_num):
    #         speech_chunk = speech[i * chunk_stride:(i + 1) * chunk_stride]
    #         is_final = i == total_chunk_num - 1
    #         res = model.generate(input=speech_chunk, cache=cache, is_final=is_final, chunk_size=chunk_size,
    #                              encoder_chunk_look_back=encoder_chunk_look_back,
    #                              decoder_chunk_look_back=decoder_chunk_look_back)
    #         print(res)


funasr = FunASR()
