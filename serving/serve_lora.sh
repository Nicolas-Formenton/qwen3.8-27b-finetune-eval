#!/bin/bash
# Serve Qwen3.8-27B (+ LoRA adapters) on the AMD MI300X via vLLM/ROCm.
#
# Design notes (all learned the hard way — see docs/FAILURES.md):
#   * LoRA modules instead of merged checkpoints: no 51GB merge per variant, and no
#     merge step that can silently no-op.
#   * Serve ONLY the adapter you use (`LORAS`), not all of them: each loaded adapter
#     occupies a LoRA slot that the kernels walk on every token, and the buffers are
#     sized by --max-loras.
#   * VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 lets you switch variants over the API without
#     a restart:
#         curl -H "Authorization: Bearer $(cat /root/vllm_key)" \
#              -H 'Content-Type: application/json' \
#              -d '{"lora_name":"qwen38-xlam","lora_path":"/root/adapters/xlam-adapter"}' \
#              http://localhost:8000/v1/load_lora_adapter
#   * Tool calling: Qwen3.x emits the native XML dialect; the matching parser is
#     `qwen3_xml` (with `hermes`, which expects JSON, vLLM returns tool_calls=null
#     silently and the model looks incapable of tool use).
#   * --api-key: the port is published on a public IP; without a key it is an open GPU.
#   * --enforce-eager is required on this stack (PIECEWISE cudagraph capture hangs).
#
# Env knobs:  MAXLEN (default 131072)  LORAS (default the D-xLAM winner)
#             MAX_LORAS (default 1)    GPU_UTIL (default 0.88)
#             CUDAGRAPH (default FULL_DECODE_ONLY)
#               PIECEWISE capture HANGS this stack (captures 51/51 then never serves —
#               docs/FAILURES.md #3), which is why --enforce-eager was used as a workaround.
#               But PIECEWISE is not the only mode: FULL_DECODE_ONLY captures full graphs
#               for DECODE only (what tok/s measures) and NONE for prefill, so it never
#               touches the hanging code path. Measured on this stack:
#                 --enforce-eager  : base 14.8 tok/s | D-xLAM  9.5 tok/s
#                 FULL_DECODE_ONLY : base 67.0 tok/s | D-xLAM 52.6 tok/s   (4.5-5.5x)
#               Set CUDAGRAPH="" to fall back to --enforce-eager.
#
# Usage:  bash serve_lora.sh
#         MAXLEN=65536 bash serve_lora.sh
#         CUDAGRAPH="" bash serve_lora.sh          # safe fallback

set -u
BASE=/models/qwen3.8-27b
PORT=8000
MAXLEN=${MAXLEN:-131072}
MAX_LORAS=${MAX_LORAS:-1}
GPU_UTIL=${GPU_UTIL:-0.88}
CUDAGRAPH=${CUDAGRAPH-FULL_DECODE_ONLY}
# Speculative decoding (MTP). Pass the JSON only; the flag is added here as an ARRAY so
# the JSON's spaces are never word-split by the shell (that silently dropped the flag and
# vLLM errored with "unrecognized arguments: {...}").
#   SPEC_JSON='{"method":"qwen3_5_mtp","num_speculative_tokens":1}' bash serve_lora.sh
SPEC_JSON=${SPEC_JSON:-}
SPEC_ARGS=()
[ -n "$SPEC_JSON" ] && SPEC_ARGS=( --speculative-config "$SPEC_JSON" )

# Attention backend. Leaving it unset lets vLLM auto-select; with AITER=0 the priority list
# is [ROCM_ATTN, TRITON_ATTN, TURBOQUANT] (vllm/platforms/rocm.py:481), i.e. the "legacy"
# backend AMD measures at 2.7-4.4x slower than ROCM_AITER_FA. Set ATTN_BACKEND=ROCM_AITER_FA
# together with AITER=1 for the recommended path (see docs/optimization-plan-2026-09-11.md).
ATTN_BACKEND=${ATTN_BACKEND:-}
ATTN_ARGS=()
[ -n "$ATTN_BACKEND" ] && ATTN_ARGS=( --attention-backend "$ATTN_BACKEND" )

# Graph mode: --enforce-eager (safe) OR an explicit cudagraph mode (faster experiment)
if [ -n "$CUDAGRAPH" ]; then
  GRAPH_ARGS=( --compilation-config "{\"cudagraph_mode\": \"$CUDAGRAPH\"}" )
  echo "cudagraph_mode=$CUDAGRAPH (experimental: avoids the PIECEWISE path that hangs)"
else
  GRAPH_ARGS=( --enforce-eager )
fi

# "<adapter dir name>:<served name>" — must match evals/apply_bfcl_model_config.py
DEFAULT_LORAS="/root/adapters/dxlam-adapter:qwen38-dxlam"
# NOTE: use ${VAR-default} (no colon), NOT ${VAR:-default}: with the colon, LORAS="" (empty
# but explicitly set) falls back to the default instead of meaning "no adapters", which
# silently loads an adapter you asked to skip.
LORAS=${LORAS-$DEFAULT_LORAS}

LORA_ARGS=()
for pair in $LORAS; do
  path="${pair%%:*}"; name="${pair##*:}"
  if [ -d "$path" ]; then
    LORA_ARGS+=( "${name}=${path}" )
  else
    echo "!! missing adapter dir: $path (skipping $name)" >&2
  fi
done

echo "serving base + ${#LORA_ARGS[@]} LoRA module(s) | maxlen=$MAXLEN | max_loras=$MAX_LORAS | util=$GPU_UTIL"
printf '   %s\n' "${LORA_ARGS[@]}"

API_KEY_FILE=/root/vllm_key
API_KEY="dummy"
[ -f "$API_KEY_FILE" ] && API_KEY="$(cat "$API_KEY_FILE")"

docker rm -f vllm-lora >/dev/null 2>&1

docker run -d --name vllm-lora \
  --device=/dev/kfd --device=/dev/dri --group-add=video --ipc=host --shm-size 16G \
  -p "${PORT}:8000" \
  -e VLLM_ROCM_USE_AITER="${AITER:-1}" \
  -e VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 \
  -v /root/models:/models \
  -v /root/adapters:/root/adapters \
  vllm/vllm-openai-rocm:latest \
  --model "$BASE" --served-model-name qwen38-base \
  --enable-lora --max-lora-rank 16 --max-loras "$MAX_LORAS" \
  ${LORA_ARGS:+--lora-modules "${LORA_ARGS[@]}"} \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --api-key "$API_KEY" \
  --max-model-len "$MAXLEN" --gpu-memory-utilization "$GPU_UTIL" \
  "${GRAPH_ARGS[@]}" "${SPEC_ARGS[@]}" "${ATTN_ARGS[@]}" >/dev/null 2>&1

echo "container started; waiting for the API (graph=${CUDAGRAPH:-enforce-eager}, spec=${SPEC_JSON:-none}, attn=${ATTN_BACKEND:-auto}, aiter=${AITER:-0})"
for i in $(seq 1 40); do
  sleep 15
  if [ "$(curl -s -m 4 -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $API_KEY" "http://localhost:${PORT}/v1/models" 2>/dev/null)" = "200" ]; then
    echo "READY after $((i*15))s"
    curl -s -H "Authorization: Bearer $API_KEY" "http://localhost:${PORT}/v1/models" \
      | tr ',' '\n' | grep -o '"id": *"[^"]*"' | grep -v modelperm | sed 's/.*: *"/   served: /;s/"$//'
    exit 0
  fi
done

echo "!! NOT READY after 600s -- last log lines:"
docker logs vllm-lora 2>&1 | tail -15 | cut -c1-200
exit 1
