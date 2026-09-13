# RECREATE.md — subir um droplet novo do zero

Tempo estimado: **~20 min** (a maior parte é download). Custo: ~$0,70 de GPU.

## 0. Criar o droplet (console AMD, ~4 min)

| Campo | Valor |
|---|---|
| Imagem | **Unsloth Studio 2026.9.4** |
| GPU | **1× MI300X** (limite da conta: nunca 2) |
| Região | **ATL1** |
| SSH key | the one registered in your cloud console |
| Backup/volume | off |

Anote o **IP** e atualize a conexão:

```bash
bash ops/set_ip.sh <NOVO_IP>     # atualiza ~/.ssh/config (alias `amd`) e ops/droplet.env
ssh amd 'hostname'               # valida
```

## 1. Subir base + adapters + scripts (~5 min)

```bash
cd ~/Dev/agent-native
tar czf /tmp/an2.tgz artifacts/*adapter* evals/serve_lora.sh evals/measure_ctx.py \
    evals/measure_stream.py evals/try_mtp2.sh evals/apply_bfcl_model_config.py \
    evals/run_bfcl.sh evals/run_tau3.sh evals/eval_suite_v2.py ops/status.sh
scp /tmp/an2.tgz amd:/root/ && ssh amd 'cd /root && tar xzf an2.tgz && mkdir -p adapters scripts2 && mv artifacts/* adapters/ && mv evals/* scripts2/ && mv ops/status.sh /root/status.sh 2>/dev/null; ls adapters'
```

Baixar o modelo base e criar a chave da API (em paralelo, ~1 min):

```bash
ssh amd 'PY=/root/.unsloth/studio/unsloth_studio/bin/python; mkdir -p /root/models /root/logs
nohup bash -c "$PY -c \"from huggingface_hub import snapshot_download as s; s(repo_id=\\\"Qwen/Qwen3.8-27B\\\", local_dir=\\\"/root/models/qwen3.8-27b\\\")\"" > /root/logs/dl_base.log 2>&1 &
openssl rand -hex 24 > /root/vllm_key; chmod 600 /root/vllm_key; echo KEY_OK'
```

## 2. Imagem do vLLM (~4 min, 47GB)

```bash
ssh amd 'nohup docker pull vllm/vllm-openai-rocm:latest > /root/logs/vllm_pull.log 2>&1 & echo PULL_STARTED'
```

## 3. Subir o servidor — config mais rápida (256k + graphs + MTP)

```bash
ssh amd 'MAXLEN=262144 SPEC_JSON="{\"method\": \"qwen3_5_mtp\", \"num_speculative_tokens\": 1}" \
  nohup bash /root/scripts2/serve_lora.sh > /root/logs/serve.log 2>&1 & echo SERVING'
sleep 300; ssh amd 'cat /root/logs/serve.log'
```

Esperado: `READY after ~300s` e `served: qwen38-base, qwen38-dxlam`.

## 4. Point a client at the endpoint

The server speaks the OpenAI protocol, so any compatible client works. The one detail that
matters: a LoRA module name is not a Hugging Face repo id, so the client must be configured with
the served name (`qwen38-dxlam`) rather than the adapter path.

```bash
cd ~/llm-finetune-eval
KEY=$(ssh amd 'cat /root/vllm_key')

# generic OpenAI-compatible client
export OPENAI_BASE_URL=http://<NEW_IP>:8000/v1
export OPENAI_API_KEY=$KEY
curl -s "$OPENAI_BASE_URL/models" -H "Authorization: Bearer $KEY" | head -c 300
```

If your client stores per-endpoint settings, each provider entry needs the same base URL and key.
Verify with a one-line completion before trusting the setup.

## 5. Validar (2 min)

```bash
bash ops/watch.sh                          # painel do grid (se for rodar BFCL)
ssh amd 'K=$(cat /root/vllm_key); for m in qwen38-base qwen38-dxlam; do
  curl -s -m 60 http://localhost:8000/v1/chat/completions -H "Authorization: Bearer $K" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$m\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":20}" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(\"$m OK\")"; done'
```

## Ordem importa (economiza tempo)

1. Criar droplet e **já disparar** o download do base + pull do vLLM (rodam em paralelo)
2. Enquanto baixam, subir adapters/scripts via scp
3. Servir só quando pull e base terminarem
4. **Nunca** rodar 2 coisas na GPU ao mesmo tempo: o vLLM pré-aloca 88% da VRAM

## Pitfalls já mapeados (não repetir)

| Sintoma | Causa | Fix |
|---|---|---|
| `/v1/models` = 000 pra sempre após "Capturing CUDA graphs" | `PIECEWISE` trava nesta stack | `CUDAGRAPH=FULL_DECODE_ONLY` |
| `unrecognized arguments: {...}` ao servir | JSON do `--speculative-config` word-split pelo shell | usar `SPEC_JSON` (o script monta o array) |
| `tool_calls: null` no agente | parser `hermes` não lê o XML do Qwen | `--tool-call-parser qwen3_xml` (já no script) |
| `Cannot copy out of meta tensor` | vLLM segurando a VRAM | parar o container antes de treinar |
| treino PEFT ~26s/step | sem Unsloth | `train_distill_unsloth.py` (import `unsloth` antes de `unsloth_zoo`) |
| `pkill -f <padrão>` mata o próprio ssh | o padrão casa com a própria linha | `pkill -x`, ou matar por PID |
