import json
import os
import sqlite3
import uuid
from contextlib import closing
from typing import Any
from urllib.parse import quote

import chainlit.data as cl_data
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.data.storage_clients.base import BaseStorageClient
from loguru import logger
from sqlalchemy import event

from app.utils import utils


_DATABASE_FILE_NAME = "audionotes.db"


class StorageClient(BaseStorageClient):
    def __init__(self):
        logger.info("storage :: local client initialized")

    @staticmethod
    def _file_path(object_key: str) -> str:
        # object_key 会持久化到数据库，读取和删除时不能假设其一定由当前版本生成。
        # 仅允许单层文件名，避免旧数据或被篡改记录通过 ../ 访问 uploads 目录之外的文件。
        if (
            not object_key
            or object_key != os.path.basename(object_key)
            or "\\" in object_key
        ):
            raise ValueError("无效的存储对象标识")
        return os.path.join(utils.upload_dir(), object_key)

    @classmethod
    def _migrate_legacy_file(cls, object_key: str) -> str:
        file_path = cls._file_path(object_key)
        if os.path.exists(file_path):
            return file_path

        # 旧版本把附件写入 storage/uploads，但返回了一个实际不存在的 /uploads 路由。
        # 本地部署升级后首次读取附件时执行一次惰性迁移；Docker 的两个挂载指向同一宿主机目录，
        # 正常情况下新路径已经可见，不会触发移动。
        legacy_path = os.path.join(utils.storage_dir("uploads"), object_key)
        if os.path.isfile(legacy_path):
            os.replace(legacy_path, file_path)
            logger.info("storage :: migrated legacy file")
        return file_path

    async def upload_file(
        self,
        object_key: str,
        data: bytes | str,
        mime: str = "application/octet-stream",
        overwrite: bool = True,
        content_disposition: str | None = None,
    ) -> dict[str, Any]:
        extension = os.path.splitext(object_key)[1].lower()
        stored_key = f"{uuid.uuid4()}{extension}"
        payload = data.encode("utf-8") if isinstance(data, str) else data
        try:
            file_path = self._file_path(stored_key)
            with open(file_path, "wb") as file:
                file.write(payload)
            logger.info(
                "storage :: uploaded file, extension: {}, byte_count: {}, mime: {}",
                extension or "unknown",
                len(payload),
                mime,
            )
            return {
                "object_key": stored_key,
                "url": await self.get_read_url(stored_key),
            }
        except Exception:
            # 不记录原始 object_key，其中可能包含用户上传的文件名；扩展名和大小足以定位问题。
            # 上传失败必须继续抛出，让 Chainlit 正确标记元素失败，而不是持久化一个空字典。
            logger.exception(
                "storage :: failed to upload file, extension: {}, byte_count: {}",
                extension or "unknown",
                len(payload),
            )
            raise

    async def delete_file(self, object_key: str) -> bool:
        try:
            os.remove(self._migrate_legacy_file(object_key))
            logger.info("storage :: deleted file")
            return True
        except FileNotFoundError:
            logger.warning("storage :: file already missing during deletion")
            return False
        except (OSError, ValueError):
            # 删除由框架后台触发，返回 False 让调用方决定是否重试；日志保留堆栈用于定位权限问题。
            logger.exception("storage :: failed to delete file")
            return False

    async def get_read_url(self, object_key: str) -> str:
        # 先执行路径校验，再对 URL 路径段编码，避免特殊字符改变请求路径语义。
        self._migrate_legacy_file(object_key)
        return f"/public/uploads/{quote(object_key, safe='')}"

    async def close(self) -> None:
        # 本地文件存储没有连接池或后台任务，此方法用于满足 Chainlit 2.x 的统一生命周期接口。
        return None


class SQLiteDataLayer(SQLAlchemyDataLayer):
    """为 Chainlit 通用 SQLAlchemy 数据层补齐 SQLite 的 JSON 兼容处理。"""

    def __init__(self, database_path: str, storage_provider: StorageClient):
        super().__init__(
            conninfo=get_connection_url(database_path),
            connect_args={"timeout": 30},
            storage_provider=storage_provider,
            show_logger=False,
        )

        # SQLite 的外键开关和等待时间都是连接级设置。SQLAlchemy 连接池创建新连接时必须重复设置，
        # 否则删除会话时可能留下孤立记录，并发写入也会过早返回 database is locked。
        event.listen(self.engine.sync_engine, "connect", self._configure_connection)

    @staticmethod
    def _configure_connection(dbapi_connection, connection_record) -> None:
        del connection_record
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys = ON")
            cursor.execute("PRAGMA busy_timeout = 30000")
        finally:
            cursor.close()

    async def execute_sql(self, query: str, parameters: dict):
        # PostgreSQL 驱动原生支持列表参数，SQLite 只接受标量。Chainlit 会把 tags 和 modes
        # 作为列表传入，因此在唯一的数据访问边界统一编码，避免业务代码感知数据库差异。
        normalized_parameters = {
            key: json.dumps(value, ensure_ascii=False)
            if isinstance(value, (dict, list))
            else value
            for key, value in parameters.items()
        }
        result = await super().execute_sql(query, normalized_parameters)
        if not isinstance(result, list):
            return result

        # Chainlit 的 PostgreSQL 查询会直接得到 JSON/数组对象；SQLite 返回文本。
        # 只转换框架明确按结构使用的字段，props 仍保留文本供 get_element 原生解析。
        structured_fields = {
            "metadata",
            "generation",
            "modes",
            "tags",
            "step_generation",
            "step_metadata",
            "step_modes",
            "step_tags",
            "thread_metadata",
            "thread_tags",
        }
        for row in result:
            for field in structured_fields.intersection(row):
                value = row[field]
                if not isinstance(value, str):
                    continue
                try:
                    row[field] = json.loads(value)
                except json.JSONDecodeError:
                    # 历史或手工修改的数据不应让整个会话列表失效；保留原值便于界面继续展示，
                    # 同时记录字段名而不记录可能包含隐私内容的字段值。
                    logger.warning("database :: invalid JSON field, field: {}", field)
        return result

    async def get_all_user_threads(
        self,
        user_id: str | None = None,
        thread_id: str | None = None,
    ):
        threads = await super().get_all_user_threads(
            user_id=user_id,
            thread_id=thread_id,
        )
        if not threads:
            return threads

        # Chainlit 的单元素查询会自行解析 props，但会话列表不会。这里在返回边界补齐转换，
        # 使音频播放器等元素属性在历史会话和当前会话中的类型保持一致。
        for thread in threads:
            for element in thread.get("elements", []):
                props = element.get("props")
                if isinstance(props, str):
                    try:
                        element["props"] = json.loads(props)
                    except json.JSONDecodeError:
                        logger.warning("database :: invalid element props JSON")
                        element["props"] = {}
        return threads


def get_database_path() -> str:
    # storage_dir(create=True) 只创建持久化根目录；数据库文件本身由 SQLite 首次连接时创建。
    return os.path.join(utils.storage_dir(create=True), _DATABASE_FILE_NAME)


def get_connection_url(database_path: str | None = None) -> str:
    # SQLite 的绝对路径 URL 需要四个斜杠。路径来自项目目录而非用户输入，保留原字符可正确支持空格。
    path = os.path.abspath(database_path or get_database_path())
    return f"sqlite+aiosqlite:///{path}"


def _initialize_database(database_path: str) -> None:
    schema = '''
CREATE TABLE IF NOT EXISTS users (
    "id" TEXT PRIMARY KEY,
    "identifier" TEXT NOT NULL UNIQUE,
    "metadata" TEXT NOT NULL DEFAULT '{}',
    "createdAt" TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS threads (
    "id" TEXT PRIMARY KEY,
    "createdAt" TEXT,
    "name" TEXT,
    "userId" TEXT,
    "userIdentifier" TEXT,
    "tags" TEXT,
    "metadata" TEXT,
    FOREIGN KEY ("userId") REFERENCES users("id") ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS steps (
    "id" TEXT PRIMARY KEY,
    "name" TEXT NOT NULL,
    "type" TEXT NOT NULL,
    "threadId" TEXT NOT NULL,
    "parentId" TEXT,
    "disableFeedback" INTEGER,
    "streaming" INTEGER NOT NULL,
    "waitForAnswer" INTEGER,
    "isError" INTEGER,
    "metadata" TEXT,
    "tags" TEXT,
    "input" TEXT,
    "output" TEXT,
    "createdAt" TEXT,
    "command" TEXT,
    "start" TEXT,
    "end" TEXT,
    "generation" TEXT,
    "showInput" TEXT,
    "language" TEXT,
    "indent" INTEGER,
    "defaultOpen" INTEGER,
    "autoCollapse" INTEGER,
    "modes" TEXT,
    FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS elements (
    "id" TEXT PRIMARY KEY,
    "threadId" TEXT,
    "type" TEXT,
    "url" TEXT,
    "chainlitKey" TEXT,
    "name" TEXT NOT NULL,
    "display" TEXT,
    "objectKey" TEXT,
    "size" TEXT,
    "page" INTEGER,
    "language" TEXT,
    "forId" TEXT,
    "mime" TEXT,
    "props" TEXT,
    "autoPlay" INTEGER,
    "playerConfig" TEXT,
    FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS feedbacks (
    "id" TEXT PRIMARY KEY,
    "forId" TEXT NOT NULL,
    "threadId" TEXT,
    "value" INTEGER NOT NULL,
    "comment" TEXT
);

CREATE INDEX IF NOT EXISTS "idx_threads_userId" ON threads("userId");
CREATE INDEX IF NOT EXISTS "idx_steps_threadId_createdAt" ON steps("threadId", "createdAt");
CREATE INDEX IF NOT EXISTS "idx_elements_threadId" ON elements("threadId");
CREATE INDEX IF NOT EXISTS "idx_feedbacks_forId" ON feedbacks("forId");
'''
    try:
        # sqlite3.Connection 的上下文只管理事务，不会关闭连接，因此额外使用 closing，
        # 避免开发热重载或初始化失败时积累文件描述符。
        with closing(sqlite3.connect(database_path, timeout=30)) as connection:
            # WAL 允许读取历史会话与写入识别结果并行；NORMAL 在本机应用中兼顾持久性和写入性能。
            # busy_timeout 为短暂写锁提供等待窗口，避免连续写入步骤时出现偶发锁冲突。
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA busy_timeout = 30000")
            connection.executescript(schema)
    except sqlite3.Error:
        logger.exception("database :: SQLite initialization failed")
        raise


def init() -> None:
    database_path = get_database_path()
    _initialize_database(database_path)
    cl_data._data_layer = SQLiteDataLayer(
        database_path=database_path,
        storage_provider=StorageClient(),
    )
    logger.info(
        "database :: SQLite initialized, file: {}",
        os.path.basename(database_path),
    )
