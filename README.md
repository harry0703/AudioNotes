# AudioNotes

## 基于 FunASR 和 Qwen2 构建的音视频转结构化笔记系统

能够快速提取音视频的内容，并且调用大模型进行整理，成为一份结构化的markdown笔记，方便快速阅读

FunASR: https://github.com/modelscope/FunASR

Qwen2: https://ollama.com/library/qwen2

## 效果展示

### 音视频识别和整理

![image](docs/1.jpg)

### 与音视频内容对话

![image](docs/2.jpg)

## 使用方法

### ① 安装 Ollama

下载对应系统的 Ollama 安装包进行安装

https://ollama.com/download

### ② 拉取模型

我以 `阿里的千问2 7b` 为例 https://ollama.com/library/qwen2

```bash
ollama pull qwen2:7b
```

### ③ 部署服务

有两种部署方式，一种是使用 Docker 部署，另一种是本地部署

#### Docker部署（推荐）🐳

```bash
curl -fsSL https://github.com/harry0703/AudioNotes/raw/main/docker-compose.yml -o docker-compose.yml
docker-compose up
```
docker 启动后，访问 http://localhost:15433/

> 登录账号为 admin，密码为 admin （可以在 docker-compose.yml 文件里面修改）

#### 本地部署 📦

需要有可访问的 postgresql 数据库

```bash
conda create -n AudioNotes python=3.10 -y
conda activate AudioNotes
git clone https://github.com/harry0703/AudioNotes.git
cd AudioNotes
pip install -r requirements.txt
```

将 `.env.example` 重命名为 `.env`，修改相关配置信息

```bash
chainlit run main.py
```
服务启动后，访问 http://localhost:8000/

> 登录账号为 admin，密码为 admin （可以在 .env 文件里面修改）

## REST API（可选）

除了网页界面，还提供了一个 REST 接口，方便脚本或其他系统调用。

出于安全考虑，接口默认关闭。在 `.env`（或 docker-compose 的 environment）中配置 `AUDIONOTES_API_TOKEN` 后启用：

```bash
AUDIONOTES_API_TOKEN=your-api-token
```

上传音视频文件，返回识别结果和整理好的笔记：

```bash
curl -X POST http://localhost:8000/api/v1/transcriptions \
  -H "Authorization: Bearer your-api-token" \
  -F "file=@/path/to/audio.mp3" \
  -F "with_notes=true"
```

返回示例：

```json
{
  "filename": "audio.mp3",
  "transcript": "识别出来的文字内容...",
  "notes": "# 结构化笔记\n..."
}
```

> `with_notes=false` 时只做语音识别，不调用大模型整理笔记。