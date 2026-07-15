import ast
import asyncio
import importlib.util
import os
import sqlite3
import sys
import tempfile
import types
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from app.utils import utils


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeLogger:
    """测试只关心业务行为，日志方法保持空实现，避免引入 Loguru 依赖。"""

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def load_module(relative_path: str, module_name: str, fake_modules: dict):
    """使用假依赖加载服务模块，确保测试环境无需下载模型或安装完整依赖。"""

    module_path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, fake_modules):
        spec.loader.exec_module(module)
    return module


class UtilsTestCase(unittest.TestCase):
    def test_audio_extension_from_mime_type(self):
        cases = {
            "audio/webm;codecs=opus": ".webm",
            " AUDIO/X-WAV ": ".wav",
            "audio/mpeg": ".mp3",
            "audio/unknown": ".audio",
            None: ".audio",
        }
        for mime_type, expected in cases.items():
            with self.subTest(mime_type=mime_type):
                self.assertEqual(utils.audio_extension_from_mime_type(mime_type), expected)


class MainWorkflowStructureTestCase(unittest.TestCase):
    def test_user_workflows_have_visible_failure_handlers(self):
        source = (PROJECT_ROOT / "main.py").read_text(encoding="utf-8")
        tree = compile(source, "main.py", "exec", flags=ast.PyCF_ONLY_AST)
        functions = {
            node.name: node
            for node in tree.body
            if isinstance(node, ast.AsyncFunctionDef)
        }

        for function_name in ("on_chat_start", "on_audio_end", "chat"):
            function = functions[function_name]
            direct_tries = [node for node in function.body if isinstance(node, ast.Try)]
            self.assertTrue(
                any(node.handlers for node in direct_tries),
                f"{function_name} 必须捕获异常并向用户反馈",
            )

        self.assertIn("处理失败，请确认文件格式", source)
        self.assertIn("录音识别失败", source)
        self.assertIn("回答生成失败", source)
        self.assertIn("hmac.compare_digest", source)
        self.assertIn("identifier=expected_username", source)
        self.assertIn("default credentials are active", source)
        self.assertIn("development JWT secret is active", source)
        self.assertIn("received chunk before start; recovering buffer", source)
        self.assertIn("recording ended without a buffer", source)
        self.assertIn('cl.user_session.set("audio_mime_type", None)', source)
        self.assertNotIn("请识别这段音频文件并且整理成结构化的笔记", source)
        self.assertIn("from chainlit.types import InputAudioChunk", source)
        self.assertNotIn("cl.AudioChunk", source)

    def test_container_listens_on_external_interface(self):
        dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
        # 容器内 localhost 健康检查无法发现端口只绑定回环地址的问题，因此固定检查启动参数。
        self.assertIn('"--host", "0.0.0.0"', dockerfile)

    def test_compose_only_exposes_web_service_to_localhost(self):
        compose = (PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
        environment = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
        requirements = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")
        # Chainlit 在容器内部需要监听所有接口供端口映射使用，但宿主机只应绑定回环地址。
        # 这样既不影响本机访问，也能避免个人笔记在默认配置下直接暴露给同一局域网。
        self.assertIn("127.0.0.1:15433:15433", compose)

        # 个人本机应用只需要内嵌 SQLite。这里同时约束编排、公开配置和 Python 驱动，
        # 防止未来调整数据层时只删除部分 PostgreSQL 配置，导致用户仍需启动无用服务。
        combined_configuration = f"{compose}\n{environment}"
        self.assertNotIn("POSTGRES_", combined_configuration)
        self.assertNotIn("\n  pg:", compose)
        self.assertIn("aiosqlite==", requirements)
        self.assertNotIn("asyncpg==", requirements)
        self.assertNotIn("psycopg2", requirements)


class ASRServiceTestCase(unittest.TestCase):
    def setUp(self):
        self.model_instances = []
        model_instances = self.model_instances

        class FakeAutoModel:
            def __init__(inner_self, **options):
                inner_self.options = options
                inner_self.generate_options = None
                model_instances.append(inner_self)

            def generate(inner_self, **options):
                inner_self.generate_options = options
                return [{"text": "识别成功", "sentence_info": []}]

        fake_funasr = types.ModuleType("funasr")
        fake_funasr.AutoModel = FakeAutoModel
        fake_loguru = types.ModuleType("loguru")
        fake_loguru.logger = FakeLogger()
        self.module = load_module(
            "app/services/asr_funasr.py",
            "asr_service_under_test",
            {"funasr": fake_funasr, "loguru": fake_loguru},
        )

    def test_model_device_and_hotwords_are_configurable(self):
        environment = {
            "ASR_MODEL": "iic/SenseVoiceSmall",
            "ASR_DEVICE": "cuda:0",
            "ASR_HOTWORDS": "AudioNotes 魔搭",
            "ASR_ENABLE_SPEAKER_DIARIZATION": "true",
        }
        with patch.dict(os.environ, environment, clear=True):
            service = self.module.FunASR()
            self.assertEqual(service.transcribe("example.wav"), "识别成功")

        model = self.model_instances[-1]
        self.assertEqual(model.options["model"], "iic/SenseVoiceSmall")
        self.assertEqual(model.options["device"], "cuda:0")
        self.assertEqual(model.options["spk_model"], "cam++")
        self.assertEqual(model.options["punc_model"], "ct-punc")
        self.assertEqual(model.generate_options["hotword"], "AudioNotes 魔搭")

    def test_default_model_uses_fun_asr_nano_inference_options(self):
        environment = {"ASR_HOTWORDS": "AudioNotes 魔搭"}
        with patch.dict(os.environ, environment, clear=True):
            service = self.module.FunASR()
            self.assertEqual(service.transcribe("example.wav"), "识别成功")

        model = self.model_instances[-1]
        self.assertEqual(
            model.options["model"],
            "FunAudioLLM/Fun-ASR-Nano-2512",
        )
        self.assertEqual(
            model.options["vad_kwargs"],
            {"max_single_segment_time": 30000},
        )
        self.assertNotIn("spk_model", model.options)
        self.assertNotIn("punc_model", model.options)
        # Nano 的参数格式与 Paraformer 不同，分别断言可避免升级时静默丢失热词增强。
        self.assertEqual(model.generate_options["batch_size"], 1)
        self.assertEqual(model.generate_options["cache"], {})
        self.assertEqual(
            model.generate_options["hotwords"],
            ["AudioNotes", "魔搭"],
        )
        self.assertNotIn("batch_size_s", model.generate_options)

    def test_srt_uses_one_based_index_timestamp_and_speaker(self):
        class FakeModel:
            def generate(self, **options):
                return [{
                    "text": "你好世界",
                    "sentence_info": [
                        {"spk": 0, "text": "你好", "start": 0, "end": 1234},
                        {"text": "世界", "start": 3723004, "end": 3725006},
                    ],
                }]

        service = self.module.FunASR()
        service._FunASR__model = FakeModel()
        result = service.transcribe("example.wav", output_type="srt")
        expected = (
            "1\n00:00:00,000 --> 00:00:01,234\n[说话人 0] 你好\n\n"
            "2\n01:02:03,004 --> 01:02:05,006\n世界\n"
        )
        self.assertEqual(result, expected)

    def test_empty_and_invalid_model_responses_are_distinguished(self):
        class EmptySpeechModel:
            def generate(self, **options):
                return [{"text": "   ", "sentence_info": []}]

        service = self.module.FunASR()
        service._FunASR__model = EmptySpeechModel()
        with self.assertRaises(self.module.NoSpeechDetectedError):
            service.transcribe("silence.wav")

        class InvalidResponseModel:
            def generate(self, **options):
                return []

        service._FunASR__model = InvalidResponseModel()
        with self.assertRaisesRegex(RuntimeError, "ASR 模型返回格式异常"):
            service.transcribe("invalid.wav")


class OllamaServiceTestCase(unittest.IsolatedAsyncioTestCase):
    def test_summary_chunks_prefer_sentence_boundaries(self):
        fake_openai = types.ModuleType("openai")
        fake_openai.AsyncOpenAI = object
        fake_loguru = types.ModuleType("loguru")
        fake_loguru.logger = FakeLogger()
        module = load_module(
            "app/services/ollama.py",
            "ollama_split_service_under_test",
            {"openai": fake_openai, "loguru": fake_loguru},
        )

        source = f"{'甲' * 700}。{'乙' * 700}。"
        chunks = module._split_summary_text(source, 1000)
        self.assertEqual(len(chunks), 2)
        self.assertTrue(chunks[0].endswith("。"))
        self.assertEqual("".join(chunks), source)

        # 无标点的连续转写仍必须严格遵守上限，避免异常 ASR 输出绕过分段保护。
        chunks_without_punctuation = module._split_summary_text("字" * 2500, 1000)
        self.assertEqual(
            [len(chunk) for chunk in chunks_without_punctuation],
            [1000, 1000, 500],
        )

    async def test_async_stream_filters_empty_frames_and_passes_request_limits(self):
        client_options = []
        request_options = []

        class FakeStream:
            def __aiter__(self):
                async def chunks():
                    yield types.SimpleNamespace(choices=[])
                    yield types.SimpleNamespace(
                        choices=[types.SimpleNamespace(
                            delta=types.SimpleNamespace(content=None)
                        )]
                    )
                    yield types.SimpleNamespace(
                        choices=[types.SimpleNamespace(
                            delta=types.SimpleNamespace(content="完成")
                        )]
                    )
                return chunks()

        class FakeCompletions:
            async def create(self, **options):
                request_options.append(options)
                return FakeStream()

        class FakeAsyncOpenAI:
            def __init__(self, **options):
                client_options.append(options)
                self.chat = types.SimpleNamespace(completions=FakeCompletions())

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc_value, traceback):
                return None

        fake_openai = types.ModuleType("openai")
        fake_openai.AsyncOpenAI = FakeAsyncOpenAI
        fake_loguru = types.ModuleType("loguru")
        fake_loguru.logger = FakeLogger()
        module = load_module(
            "app/services/ollama.py",
            "ollama_service_under_test",
            {"openai": fake_openai, "loguru": fake_loguru},
        )

        received = []

        async def callback(content):
            received.append(content)

        environment = {
            "OLLAMA_TIMEOUT_SECONDS": "45.5",
            "OLLAMA_MAX_RETRIES": "2",
        }
        with patch.dict(os.environ, environment, clear=True):
            result = await module.chat_with_ollama([], callback=callback)

        self.assertEqual(result, "完成")
        self.assertEqual(received, ["完成"])
        self.assertEqual(
            client_options[-1]["base_url"],
            "http://localhost:11434/v1",
        )
        self.assertEqual(client_options[-1]["timeout"], 45.5)
        self.assertEqual(client_options[-1]["max_retries"], 2)
        # 默认模型是部署行为的一部分，测试可防止代码、Compose 和文档后续再次漂移。
        self.assertEqual(
            request_options[-1]["model"],
            "qwen3.5:2b",
        )

        invalid_environment = {
            "OLLAMA_TIMEOUT_SECONDS": "0",
            "OLLAMA_MAX_RETRIES": "-1",
            "OLLAMA_SUMMARY_CHUNK_SIZE": "invalid",
        }
        with patch.dict(os.environ, invalid_environment, clear=True):
            self.assertEqual(module._get_timeout_seconds(), 600.0)
            self.assertEqual(module._get_max_retries(), 1)
            self.assertEqual(module._get_summary_chunk_size(), 6000)

    async def test_long_summary_is_chunked_then_merged(self):
        requests = []
        client_count = 0

        class FakeStream:
            def __init__(self, content):
                self.content = content

            def __aiter__(self):
                async def chunks():
                    yield types.SimpleNamespace(
                        choices=[types.SimpleNamespace(
                            delta=types.SimpleNamespace(content=self.content)
                        )]
                    )
                return chunks()

        class FakeCompletions:
            async def create(self, **options):
                requests.append(options)
                request_index = len(requests)
                # 前三个请求模拟每段都产生较长摘要，使合并内容再次超过 1000 字符上限。
                # 后续分组合并返回短结果，用于验证层级合并能够收敛并输出最终笔记。
                if request_index <= 3:
                    return FakeStream(f"片段{request_index}" + "摘" * 600)
                return FakeStream(f"结果{request_index}")

        class FakeAsyncOpenAI:
            def __init__(self, **options):
                nonlocal client_count
                client_count += 1
                self.chat = types.SimpleNamespace(completions=FakeCompletions())

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc_value, traceback):
                return None

        fake_openai = types.ModuleType("openai")
        fake_openai.AsyncOpenAI = FakeAsyncOpenAI
        fake_loguru = types.ModuleType("loguru")
        fake_loguru.logger = FakeLogger()
        module = load_module(
            "app/services/ollama.py",
            "ollama_summary_service_under_test",
            {"openai": fake_openai, "loguru": fake_loguru},
        )

        streamed_content = []

        async def callback(content):
            streamed_content.append(content)

        # 2500 个连续字符模拟没有自然换行的 ASR 长转写。三个片段摘要拼接后仍超过上限，
        # 因此先产生三次分组压缩，再发起最终合并；中间内容不应流到页面。
        with patch.dict(os.environ, {"OLLAMA_SUMMARY_CHUNK_SIZE": "1000"}):
            result = await module.summarize_with_ollama("字" * 2500, callback)

        self.assertEqual(len(requests), 7)
        self.assertEqual(client_count, 1)
        self.assertEqual(result, "结果7")
        self.assertEqual(streamed_content, ["结果7"])
        self.assertEqual(requests[0]["max_tokens"], 1024)
        self.assertEqual(requests[5]["max_tokens"], 1024)
        self.assertNotIn("max_tokens", requests[-1])
        self.assertIn("结果4", requests[-1]["messages"][-1]["content"])
        self.assertIn("结果5", requests[-1]["messages"][-1]["content"])
        self.assertIn("结果6", requests[-1]["messages"][-1]["content"])
        final_system_prompt = requests[-1]["messages"][0]["content"]
        self.assertIn("待办事项", final_system_prompt)
        self.assertIn("原文未明确", final_system_prompt)
        self.assertIn("不得自行分配", final_system_prompt)


class DataLayerTestCase(unittest.TestCase):
    @staticmethod
    def _load_data_layer_module():
        class FakeStorageClientBase:
            pass

        class FakeSQLAlchemyDataLayer:
            def __init__(self, **options):
                self.options = options
                self.engine = types.SimpleNamespace(sync_engine=object())

            async def execute_sql(self, query, parameters):
                self.executed_query = query
                self.executed_parameters = parameters
                return getattr(self, "query_result", None)

            async def get_all_user_threads(self, user_id=None, thread_id=None):
                self.thread_query = (user_id, thread_id)
                return getattr(self, "threads", None)

        fake_chainlit = types.ModuleType("chainlit")
        fake_chainlit.__path__ = []
        fake_chainlit_data = types.ModuleType("chainlit.data")
        fake_chainlit_data.__path__ = []
        fake_chainlit_data._data_layer = None
        fake_chainlit.data = fake_chainlit_data
        fake_sqlalchemy = types.ModuleType("chainlit.data.sql_alchemy")
        fake_sqlalchemy.SQLAlchemyDataLayer = FakeSQLAlchemyDataLayer
        # Chainlit 2.11 不再从 chainlit.data 顶层导出 BaseStorageClient，必须从
        # storage_clients.base 使用稳定定义路径。这里模拟真实包结构，避免测试依赖完整运行镜像。
        fake_storage_clients = types.ModuleType("chainlit.data.storage_clients")
        fake_storage_clients.__path__ = []
        fake_storage_base = types.ModuleType("chainlit.data.storage_clients.base")
        fake_storage_base.BaseStorageClient = FakeStorageClientBase
        fake_loguru = types.ModuleType("loguru")
        fake_loguru.logger = FakeLogger()
        fake_sqlalchemy_root = types.ModuleType("sqlalchemy")
        fake_sqlalchemy_root.event = types.SimpleNamespace(listen=lambda *args: None)

        return load_module(
            "app/services/data_layer.py",
            "data_layer_under_test",
            {
                "chainlit": fake_chainlit,
                "chainlit.data": fake_chainlit_data,
                "chainlit.data.sql_alchemy": fake_sqlalchemy,
                "chainlit.data.storage_clients": fake_storage_clients,
                "chainlit.data.storage_clients.base": fake_storage_base,
                "loguru": fake_loguru,
                "sqlalchemy": fake_sqlalchemy_root,
            },
        )

    def test_sqlite_database_initialization_and_json_compatibility(self):
        module = self._load_data_layer_module()
        with tempfile.TemporaryDirectory() as directory:
            # 数据库目录包含空格，覆盖本地常见路径场景，并验证 URL 不依赖 PostgreSQL 式转义配置。
            storage_directory = Path(directory) / "local storage"
            storage_directory.mkdir()
            with patch.object(
                module.utils,
                "storage_dir",
                return_value=str(storage_directory),
            ):
                module.init()

            database_path = storage_directory / "audionotes.db"
            self.assertTrue(database_path.is_file())
            self.assertEqual(
                module.get_connection_url(str(database_path)),
                f"sqlite+aiosqlite:///{database_path}",
            )

            with closing(sqlite3.connect(database_path)) as connection:
                tables = {
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    )
                }
                step_columns = {
                    row[1] for row in connection.execute('PRAGMA table_info("steps")')
                }
                journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]

            self.assertEqual(
                tables,
                {"users", "threads", "steps", "elements", "feedbacks"},
            )
            self.assertTrue(
                {"command", "defaultOpen", "autoCollapse", "modes"}.issubset(
                    step_columns
                )
            )
            self.assertEqual(journal_mode, "wal")

        # init 已经把 SQLite 适配层注册给 Chainlit，直接复用该实例继续验证结构化字段转换。
        layer = module.cl_data._data_layer
        self.assertIsInstance(layer, module.SQLiteDataLayer)
        layer.query_result = [
            {
                "thread_tags": '["audio", "meeting"]',
                "step_metadata": '{"favorite": true}',
                "props": '{"duration": 10}',
            }
        ]
        result = asyncio.run(
            layer.execute_sql(
                "SELECT 1",
                {"tags": ["audio"], "metadata": {"language": "zh"}},
            )
        )
        self.assertEqual(layer.executed_parameters["tags"], '["audio"]')
        self.assertEqual(
            layer.executed_parameters["metadata"],
            '{"language": "zh"}',
        )
        self.assertEqual(result[0]["thread_tags"], ["audio", "meeting"])
        self.assertEqual(result[0]["step_metadata"], {"favorite": True})
        self.assertEqual(result[0]["props"], '{"duration": 10}')

        layer.threads = [
            {"elements": [{"props": '{"duration": 10}'}]},
        ]
        threads = asyncio.run(layer.get_all_user_threads(user_id="user-id"))
        self.assertEqual(threads[0]["elements"][0]["props"], {"duration": 10})

    def test_storage_client_handles_local_files(self):
        module = self._load_data_layer_module()

        # Chainlit 2.x 要求本地存储实现完整的上传、读取 URL、删除和关闭接口。
        # 同时覆盖字符串内容编码和旧附件迁移，避免升级后历史附件无法读取。
        with tempfile.TemporaryDirectory() as directory:
            public_directory = Path(directory) / "public"
            legacy_directory = Path(directory) / "legacy"
            public_directory.mkdir()
            legacy_directory.mkdir()
            with (
                patch.object(
                    module.utils,
                    "upload_dir",
                    return_value=str(public_directory),
                ),
                patch.object(
                    module.utils,
                    "storage_dir",
                    return_value=str(legacy_directory),
                ),
            ):
                storage = module.StorageClient()
                uploaded = asyncio.run(
                    storage.upload_file(
                        "会议 记录.txt",
                        "测试内容",
                        content_disposition="attachment",
                    )
                )
                stored_path = public_directory / uploaded["object_key"]
                self.assertEqual(stored_path.read_text(encoding="utf-8"), "测试内容")
                self.assertTrue(uploaded["url"].startswith("/public/uploads/"))
                self.assertTrue(asyncio.run(storage.delete_file(uploaded["object_key"])))
                self.assertFalse(stored_path.exists())

                legacy_file = legacy_directory / "legacy.txt"
                legacy_file.write_text("旧附件", encoding="utf-8")
                legacy_url = asyncio.run(storage.get_read_url("legacy.txt"))
                self.assertEqual(legacy_url, "/public/uploads/legacy.txt")
                self.assertFalse(legacy_file.exists())
                self.assertEqual(
                    (public_directory / "legacy.txt").read_text(encoding="utf-8"),
                    "旧附件",
                )
                asyncio.run(storage.close())


if __name__ == "__main__":
    unittest.main()
