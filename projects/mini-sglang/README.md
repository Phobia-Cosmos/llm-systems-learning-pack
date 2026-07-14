# mini-sglang for MiniLLM

This is a tiny teaching server that exposes MiniLLM through OpenAI-like HTTP
endpoints. It is not the real SGLang runtime. Its purpose is to make the
serving layer visible before implementing RadixAttention, prefix cache, and
runtime scheduling inside SGLang.

Run:

```bash
cd /home/undefined/Desktop/ai
source /home/undefined/Desktop/ai/use_disk_ai_env.sh
python projects/mini-sglang/mini_sglang_server.py \
  --checkpoint projects/minillm/artifacts/checkpoints/minillm.pt \
  --host 127.0.0.1 \
  --port 8011 \
  --device cpu
```

Completion request:

```bash
curl http://127.0.0.1:8011/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"用户: 什么是 decoder-only Transformer？\n助手:","max_tokens":80,"temperature":0.7}'
```

Chat request:

```bash
curl http://127.0.0.1:8011/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"什么是 attention？"}],"max_tokens":80}'
```
