# Python 3.12 仍处于安全维护期，Bookworm 也比已经接近生命周期末期的 Bullseye 更适合作为生产基线。
FROM python:3.14-slim-bookworm

# Set the working directory in the container
WORKDIR /app

# 容器日志需要立即输出到 stdout/stderr；禁止生成 pyc 可减少只读源码目录中的运行时写入。
ENV PYTHONPATH="/app" \
    PYTHONUNBUFFERED="1" \
    PYTHONDONTWRITEBYTECODE="1"

# 通用镜像默认使用 CPU wheel，避免把体积很大的 CUDA 运行库打进不使用 GPU 的部署。
# 构建 GPU 镜像时可以通过 --build-arg 指向与宿主机 CUDA 版本匹配的 PyTorch 官方软件源。
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

# Copy only the requirements.txt first to leverage Docker cache
COPY requirements.txt .

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    --no-install-recommends \
    libgomp1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 先固定安装同版本的 torch/torchaudio，后续依赖解析会复用它们，不会从 PyPI 换回带 CUDA 组件的 wheel。
RUN pip install --no-cache-dir \
        --index-url "${PYTORCH_INDEX_URL}" \
        torch==2.11.0 torchaudio==2.11.0 \
    && pip install --no-cache-dir -r requirements.txt

# 运行镜像只需要入口、业务代码和许可证。显式复制白名单可以避免测试、Workflow、
# Compose 及未来新增的开发文件意外进入镜像，同时让生产文件边界保持清晰。
COPY main.py LICENSE ./
COPY app ./app

# 生产容器不启用 -w 热重载，避免额外的文件监控和子进程开销。Chainlit 2.11 默认只监听
# 127.0.0.1，必须显式绑定所有容器接口，否则内部健康检查正常但宿主机端口无法访问。
CMD ["chainlit", "run", "./main.py", "--host", "0.0.0.0", "--port", "15433"]
