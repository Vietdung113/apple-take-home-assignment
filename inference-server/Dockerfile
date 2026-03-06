FROM ghcr.io/ggml-org/llama.cpp:server

# Download model at build time for faster startup
RUN apt-get update && apt-get install -y python3 python3-pip && \
    pip3 install --break-system-packages huggingface-hub && \
    python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('unsloth/Qwen3-0.6B-GGUF', 'Qwen3-0.6B-Q4_K_M.gguf', local_dir='/models')" && \
    apt-get remove -y python3-pip && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

EXPOSE 8080

ENTRYPOINT ["/llama-server"]
CMD ["--model", "/models/Qwen3-0.6B-Q4_K_M.gguf", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--ctx-size", "4096", \
     "--parallel", "4", \
     "--n-gpu-layers", "-1", \
     "--cont-batching"]
