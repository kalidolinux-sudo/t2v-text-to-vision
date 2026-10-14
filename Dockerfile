# T2V · Text to Vision — Gradio demo.
#   CPU:              docker build -t t2v .
#   AMD GPU (ROCm):   docker build -t t2v --build-arg TORCH_INDEX=https://download.pytorch.org/whl/rocm6.4 .
#                     docker run --device=/dev/kfd --device=/dev/dri --group-add video ...
# Run (the TokenFactory key is passed at run time, never baked into the image):
#   docker run -p 7860:7860 -e TOKENFACTORY_API_KEY=... t2v   # fetches t2v_1k.pt from the GitHub release on first start
FROM python:3.11-slim
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
WORKDIR /app
RUN pip install --no-cache-dir torch --index-url ${TORCH_INDEX}
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV TOKENFACTORY_BASE_URL=https://developer.amd.com.cn/radeon/api/v1 \
    TOKENFACTORY_MODEL=DeepSeek-V4-Flash \
    GRADIO_ANALYTICS_ENABLED=False
EXPOSE 7860
CMD ["python", "app.py", "--port", "7860"]
