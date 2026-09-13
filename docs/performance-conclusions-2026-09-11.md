# Performance: o que destravou os 50 tok/s (11/09/2026)

Resumo executivo do dia. Todas as medições: 1x MI300X (gfx942), vLLM 0.29.0 (imagem Unsloth
Studio), 256k de contexto, `FULL_DECODE_ONLY`, 400 tokens de saída, temperature 0,
`ignore_eos`, 2 repetições, GPU ociosa antes de cada linha.

## O resultado

| Config | contexto curto | ~90k tokens |
|---|---|---|
| **Ontem (o que o usuário usava)** — eager, AITER off, 5 LoRAs | 5,7 | ~5 |
| AITER off (controle de hoje) | 67,0 | **6,8** |
| **AITER + ROCM_AITER_FA** | 67,3 | **60,3** |
| AITER + MTP | 58,1 | 52,1 |
| **AITER, sem MTP, com o LoRA D-xLAM** ← config final | 54,2 | **50,3** ✅ |

**Meta do usuário (≥50 tok/s no caso de uso real) atingida: 50,3 tok/s a 90k tokens com o
modelo fine-tuned.** Ontem eram ~5 tok/s na mesma situação.

## A causa raiz: um workaround meu, não retestado

O servidor rodava com `VLLM_ROCM_USE_AITER=0`. Adicionei isso na **imagem antiga**
(`vllm/vllm-openai-rocm:qwen38`) porque o AITER causava segfault no load. **Nunca retestei
na imagem atual (Unsloth Studio).** Custo: 9x.

Mecanismo, confirmado no log de cada config:

| | AITER off | AITER on |
|---|---|---|
| Backend escolhido | `Overriding with ROCM_ATTN` (legado) | `Using AITER Flash Attention backend` |
| GDN decode | `triton` | `triton` (inalterado) |
| 90k tok/s | 6,8 | 60,3 |

O doc oficial da AMD diz que `ROCM_AITER_FA` é **2,7–4,4x** sobre o `ROCM_ATTN` legado, e que
o "Extend Path" do AITER usa **LSE merging para lidar com contextos de 100K+**. É exatamente
o nosso caso (turnos de continuação com contexto longo) — e o ganho aparece **só** ali
(67,0 → 67,3 em contexto curto, 6,8 → 60,3 em 90k).

## Duas hipóteses minhas que estavam erradas

1. **"O gargalo é o kernel do GDN"** — errado. O GDN continua no fallback Triton
   (`torch.ops._C.fused_gdn_decode_post_conv_mtp is not built`) mesmo com tudo otimizado, e
   mesmo assim chegamos a 50 tok/s. O ganho veio inteiramente da **atenção**, não do GDN.
   (O fix do GDN flat-layout existe no PR #53623 do vLLM, não mergeado — não contar com ele.)
2. **"MTP é sempre 2,55x"** — errado. Com o backend legado o MTP dava 2,55x; com o AITER
   ele **custa 13%** (60,3 → 52,1 em 90k). Ganhos de otimização não são independentes:
   MTP compensava a lentidão do backend ruim.

## Config final

```bash
AITER=1 ATTN_BACKEND=ROCM_AITER_FA MAXLEN=262144 CUDAGRAPH=FULL_DECODE_ONLY \
  bash evals/serve_lora.sh      # sem MTP; serve base + D-xLAM
```

Ou o wrapper: `bash evals/serve_final.sh`.

Defaults do `serve_lora.sh` atualizados para o que foi medido: `AITER=1`,
`CUDAGRAPH=FULL_DECODE_ONLY`, `MAXLEN=131072`, 1 LoRA, **sem MTP**.

## Teste de aceitação: sessão real do Hermes (2026-09-11)

Mesma conversa de ontem (`20260910_210112_e06943`, "Create HTML for voting roles"), contexto
de ~104k tokens. Engine amostrado a cada 12–15s durante a chamada.

| | ontem (eager, AITER off) | hoje (FULL_DECODE_ONLY + AITER) |
|---|---|---|
| tokens in/out por turno | 92.602 / 1.711 | 104.329 / 9.339 |
| latência do turno | 394,4 s | **225,9 s** |
| **tok/s efetivo** | **4,3** | **41,3** |
| pico sustentado (engine) | ~4,6 | **47,3** por 180 s |

O arquivo cresceu em writes sucessivos com o engine a ~46,5 tok/s quase todo o tempo — as
quedas (18,1 e 8,3) são prefill de turno novo, não decode lento:

```
t=15s   arquivo=254b    18,1 tok/s   (prefill)
t=30s   arquivo=2190b   38,5
t=135s  arquivo=12809b  46,5
t=195s  arquivo=20998b  42,8
t=225s  arquivo=24589b   8,3         (fim do turno)
```

**10x no caso de uso real, e o artefato saiu.**

## Dois bugs de integração que a velocidade não resolvia

Velocidade não bastou para a sessão funcionar: dois defeitos silenciosos a bloqueavam, ambos
diagnosticados no `state.db` e no `agent.log` do perfil.

### 1. A sessão guardava o IP do droplet destruído

```json
{"provider": "custom:qwen-amd-fast", "base_url": "http://129.212.188.58:8000/v1"}
```

O `model_config` fica gravado **por sessão**; recriar o droplet invalida o snapshot. O probe
de contexto falhava em silêncio e o Hermes caía num valor fixo de catálogo:

```
Could not detect context length ... defaulting to 256,000
Using hardcoded context length 131,072 for model 'qwen38-dxlam' (catalog match on 'qwen')
```

131k em vez dos 262k configurados.

**Correções:** (a) `UPDATE sessions SET model_config=…, billing_base_url=…` no `state.db`;
(b) `context_length` declarado **por provider** em `providers.<nome>.models.<modelo>`, porque
o `model.context_length` global é descartado quando o provider ativo difere do default
(`agent/agent_init.py:_scope_context_length_to_default_runtime`).

### 2. Write de arquivo grande numa única tool call nunca completa

```
WARNING agent.message_sanitization: Unrepairable tool_call arguments for write_file
  — replaced with empty object
⚠️ Response truncated (finish_reason='length')
```

Um HTML de ~175 KB são ~45k tokens: a chamada de ferramenta estoura o cap de saída, o JSON
chega truncado, o Hermes **descarta a chamada inválida** e o modelo entra em loop de
retentativa. Parece "o modelo não sabe usar ferramentas" — não é.

**Fix:** escrever em partes (write_file pequenos + patch). Resultado: 24.589 bytes, 28 tool
calls, **zero truncamentos**.

**Lição:** ao integrar um modelo local como agente, validar a escrita de arquivos grandes
separadamente da velocidade — os dois defeitos acima se confundem com incapacidade do modelo.

## Ordem de ganho (do maior para o menor)

| Alavanca | Ganho | Onde |
|---|---|---|
| AITER (backend de atenção) | **8,9x em 90k** / ~1x em curto | contexto longo |
| `FULL_DECODE_ONLY` (vs `--enforce-eager`) | 4,5x | global |
| 1 LoRA em vez de 5 carregados | +68% | global |
| MTP (com backend legado) | 2,55x | global |
| Contexto 256k vs 64k | **0** | não custa |
| LoRA ativo | −25% | global |

## Lições

- **Workaround vira dívida técnica invisível.** Um `=0` que "resolveu" um crash continuou
  custando 9x por dias. Ao trocar de imagem/versão, re-testar os workarounds é obrigatório —
  é a primeira coisa a fazer, não a última.
- **Ganhos de otimização não são aditivos.** MTP ajudava num backend e atrapalhava em outro.
  Sempre medir a combinação final, não somar ganhos de medições separadas.
- **Medir a condição do uso real.** Todas as medições antigas de contexto curto davam
  ~67 tok/s e escondiam o problema; o defeito só aparecia a 90k, que era onde o usuário
  realmente operava.
