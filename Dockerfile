FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

COPY pyproject.toml .
RUN uv sync --frozen

COPY . .

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
