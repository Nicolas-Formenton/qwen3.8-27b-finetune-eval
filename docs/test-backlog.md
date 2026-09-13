# Test Backlog — agent-native (matriz de caracterização agêntica)

> Objetivo: caracterizar cada modelo para uso AGÊNTICO (acerto de tool-call, aderência a formato/instrução, velocidade, cache, custo) com a mesma bateria aplicada a todos. Atualizado: 2026-09-10.

## Modelos na matriz

| Código | Modelo | Stack | Status |
|---|---|---|---|
| BASE | Qwen/Qwen3.8-27B (BF16) | vLLM ROCm | ✅ medido |
| OH | +OpenHermes LoRA r16 (run 1, PEFT, 3000 linhas) | vLLM ROCm | ✅ medido |
| XLAM | +xLAM LoRA r16 (run 2, Unsloth, 6000 linhas) | vLLM ROCm | ✅ medido (após correção de merge) |
| DBASE | +distill LoRA (base pura) | vLLM ROCm | ✅ medido |
| DOH | +distill LoRA (sobre OH) | vLLM ROCm | ✅ medido |
| DXLAM | +distill LoRA (sobre XLAM corrigido) | vLLM ROCm | ✅ medido |
| GLM | GLM-5.3-Flash UD-Q3_K_XL (147.5GB) | llama.cpp ROCm + MTP | ⏳ pendente (GGUF baixado, build ok) |

## Resultado principal — eval-03-matrix-200 (n=200, temp 0, thinking off)

| Modelo | Suíte 20 (thinking ON, ver defeito D1) | Tool-call 200 | IC95% | Diverge do base | tok/s | TTFT |
|---|---|---|---|---|---|---|
| BASE | 6/20 | 83.5% (167) | [0.777,0.880] | — | 20.6 | 0.201 |
| OH | 10/20 | 83.5% (167) | [0.777,0.880] | 2 | 19.2 | 0.195 |
| XLAM (corrigido) | 7/20 | 84.0% (168) | [0.783,0.884] | 1 | 20.8 | 0.189 |
| DBASE | 7/20 | 84.0% (168) | [0.783,0.884] | 1 | 19.5 | 0.195 |
| DOH | 4/20 | 84.0% (168) | [0.783,0.884] | 1 | 20.0 | 0.199 |
| DXLAM (corrigido) | 6/20 | 84.0% (168) | [0.783,0.884] | 1 | 20.0 | 0.196 |

Perfil de erro idêntico em todos: ~23/200 sem JSON parseável, ~9-10/200 com função errada. Quatro modelos distintos (XLAM, DBASE, DOH, DXLAM) produzem **vetor de acerto bit-idêntico** → a métrica saturou: SFT de 6k linhas muda ~1 de 200 respostas, e isso é ruído, não ganho.

## Defeitos de harness encontrados (honestidade metodológica)

- **D1 — `eval_suite.py` roda com thinking ON:** as respostas começam com o raciocínio do modelo, então os verificadores de JSON/formato leem texto de reasoning. A coluna `json 0/3` para *todos* os modelos é artefato disso, não propriedade dos modelos. Correção: passar `chat_template_kwargs {"enable_thinking": false}` no `ask()` (igual ao script de 200) e/ou tirar o bloco de thinking antes do parse.
- **D2 — métrica de tool-call saturada:** só o *primeiro nome de função* é comparado (sem argumentos, sem chamadas paralelas). Score ~84% já no base.
- **D3 — merge silencioso (corrigido):** `PeftModel.merge_and_unload()` produziu checkpoint bit-idêntico ao base para o adapter salvo pelo Unsloth (regex `target_modules` + `auto_mapping` sob transformers 5.x). Detecção: `(W_merged − W_base)` deve ser igual a `scale·(B@A)`. Correção: merge manual via safetensors (496 módulos aplicados, verificado). Rodadas `xlam` e `dxlam` anteriores foram invalidadas e refeitas.

## Bateria de testes

| # | Métrica | Método | Para que serve no agente |
|---|---|---|---|
| A | Tool-call (nome) | 200 linhas held-out do xLAM (reservoir seed fixo, linhas 6000+), temp 0, concorrência 12 | O modelo chama a função CERTA? ✅ feito |
| A2 | Tool-call (argumentos + paralelas) | mesma base de 200, mas comparando argumentos e todas as chamadas do turno | Diferencia fine-tunes que a métrica A não separa ⏳ |
| B | Formato/instrução | suíte própria de 20 tarefas determinísticas — **precisa rodar com thinking off** (D1) | Segue o dialeto do harness ⏳ refazer |
| C | tok/s + TTFT | stream, 256 tokens | Velocidade real ✅ feito — final: 54,2 curto / 50,3 @90k |
| C2 | tok/s com CUDA graphs + AITER | `VLLM_ROCM_USE_AITER=1` + `ROCM_AITER_FA` + `FULL_DECODE_ONLY` | ✅ feito — 47 tok/s @104k numa sessão real (era 4,3) |
| E | Prefix cache | medido em carga concorrente (4 subagents) | ✅ feito — 87% de hit rate; custo do prefill quase some |
| H | BFCL v4 (instrumento principal) | `bfcl-eval` via `REMOTE_OPENAI_BASE_URL`; aceita `--enable-lora` sem merge | **Número-título do CV** ⏳ bloqueado por bug de registro (rodou a 1,5%) |
| H2 | τ³-bench (instrumento secundário) | instalado no droplet (`/root/tau-env`), `check-data` OK | Mede comportamento agêntico multi-turno ⏳ nunca rodou |
| F | Quantização | FP8/GGUF do XLAM vs BF16 | Custo/velocidade vs qualidade ⏳ |
| I | Qualitativo lado a lado | 4-6 prompts com outputs completos salvos no repo | Prova visual pro CV/repo ⏳ |
| G | Custo por tarefa | tokens × $/h self-host | $ por tarefa agêntica ⏳ |
| J | Prefill por nível de contexto | `measure_prefill.py`, 6 níveis (1k→256k), cold vs warm, 3 repeats | Curva de custo do contexto ⏳ see abaixo |
| J2 | Prefill × quantização | mesmo script, Qwen3.8-27B quantizado (8-bit dense + 4-bit experts, affine gs64 / ngram gs32) | Testar o claim "2.000 tok/s prefill" ⏳ |

## Próximos passos (ordem sugerida, revisada 11/09)

1. **BFCL v4 — destravar o número-título.** Re-registrar os 6 modelos corrigindo
   `underscore_to_dot` e o dialeto do prompt (o `func_name1` que derrubou o score a 1,5%),
   rodar o base como controle e só então os 5 adapters. É o único instrumento que ainda
   não produziu número válido, e é o que sustenta o repo no CV. (~1-2h GPU)
2. **A2 — argumentos + chamadas paralelas.** A métrica A saturou (vetores bit-idênticos em
   4 modelos); A2 é o único eixo que ainda pode separar os fine-tunes. (~30 min GPU)
3. **τ³-bench** — já instalado no droplet, nunca rodado. Responde se o D-xLAM, que colapsou
   a abstenção para 5,8%, faz estrago num agente multi-turno real. (~1h GPU)
4. **Suíte B refeita com thinking off** (fix D1, 1 linha) nos 6 modelos. (~15 min GPU)
5. **GLM-5.3-Flash na GPU** (prebuilts llama.cpp ROCm da imagem Unsloth):
   `-ngl 999 --flash-attn on --jinja --parallel 6 --spec-type draft-mtp`, medir tok/s
   com/sem MTP e rodar A/A2. (~1h GPU)
6. **Head-to-head GLM × melhor Qwen** em 5-10 prompts agênticos reais (prova visual, item I).
7. **Publicação** (sem GPU): adapters + model cards no HF Hub, README do repo com a tabela
   de performance e a metodologia; citar licenças (xLAM é CC-BY-NC; teacher do distill).

## J — Prefill: o claim "2.000 tok/s" e como testar (12/09)

**O claim de referência:** Qwen3.8-Flash-Next, build de dev, contexto 16k, `mlx-serve` (Apple MLX),
hardware M5 Max 128GB → "breaking 2.000 tok/s prefill". Quantização do autor: **8-bit dense +
4-bit experts**, affine groupsize 64, ngram 4-bit groupsize 32.

**Por que não é comparável direto com a gente:**

| | Claim | Nós |
|---|---|---|
| Hardware | M5 Max 128GB (unified, Apple Silicon) | 1× MI300X 192GB (HBM3) |
| Runtime | MLX | vLLM ROCm + AITER |
| Quantização | 8-bit dense + 4-bit experts | **BF16 completo** |
| Modelo | Flash-Next (build de dev) | Qwen3.8-27B |

O claim combina dois ganhos: quantização agressiva (menos bytes por peso → mais rápido) **e** a
largura de banda unificada do M5 Max. Baixar a precisão é o fator que mais muda a ordem de
grandeza, então é o eixo que precisamos isolar.

**Já medido (BF16, sem quantização)** — ver `prefill_profile.json`: cold pica em ~7.200 tok/s aos
16k e cai para 1.244 aos 256k; warm (prefix cache) chega a 170.827 tok/s aos 256k. Ou seja, aos
16k e em BF16 **nós já passamos dos 2.000 tok/s cold** (7.229). O ponto do claim não é o número
bruto — é atingi-lo **com quantização** e em contexto longo.

**J2 — o experimento que falta:**

1. Quantizar o Qwen3.8-27B no formato do claim (dense 8-bit, experts 4-bit, affine gs64) e servir
   no vLLM ROCm.
2. Rodar `measure_prefill.py` nos mesmos 6 níveis, mesmos 3 repeats.
3. Reportar cold e warm lado a lado com a curva BF16 que já temos.
4. Medir também a **perda de qualidade**: repetir o BFCL (ou um subconjunto — `non_live` já dá a
   coluna de abstenção) no modelo quantizado. Velocidade sem o custo de acurácia não vale.

Guardrail: sem o passo 4 o resultado é marketing, não engenharia.

## Regras

- Sempre mesma suíte, mesma temperatura, mesmas instruções; anotar diferenças de engine (vLLM vs llama.cpp) e de quantização como variável
- Nunca aceitar merge sem verificar: `(W_merged − W_base)` vs `scale·(B@A)` com tolerância de poucos ulps bf16
- Thinking: desligado para evals de formato/tool-call (senão o parse lê reasoning); ligado apenas no qualitativo
- Resultados: `results/raw/*.json` (brutos) + `results/eval-03-matrix-200.json` (consolidado)

---

# Estado em 2026-09-12 (parada por falta de GPU na AMD)

Droplet destruído. Crédito expira **13/09**. Tudo que importa está salvo localmente:
5 adapters LoRA (`artifacts/`), BFCL completo (`results/bfcl/`), τ³ bruto
(`results/tau3/raw/`, 3 `results.json`), `docs/handoff-2026-09-12.md`.

## Fechado

- **BFCL v4 (número-título do repo).** Config: handler `QwenFCHandler`, thinking LIGADO,
  temp 0.001, 16 threads, seed 300. Resultado: xLAM **89,69%** · D-xLAM **88,06%** ·
  base **87,87%** (Non-Live Overall). Abstenção: 5,42% / **85,83%** / 67,92%.
  Achado: o SFT com xLAM destrói a abstenção; o distill conserta e supera o base.
- **Velocidade:** 47 tok/s @104k numa sessão real do Hermes (era 4,3). AITER é obrigatório
  para contexto longo: 6,8 → 50,3 tok/s @90k.
- **6 defeitos de harness** documentados com script de correção (2 no BFCL, 1 no τ³/run_tau3,
  1 no cron do Hermes no Windows, 1 de contenção zumbi, 1 de thinking).

## Aberto — retomar quando houver GPU

1. **τ³ faltando:** dxlam (airline + retail) e o **retail do xLAM**. ~1,5h.
   `chain_tau3.sh` faz load/unload de adapter por modelo.
2. **Investigar a tarefa travada `13.0`** do airline do xLAM: rodou 1783s+, processo a 0,5%
   de CPU (ocioso, não trabalhando). Loop agente↔simulador que o `--max-steps 60` não cortou.
   É um achado de comportamento, não só um run perdido.
3. **Publicação (sem GPU):** adapters no HF Hub + README com a tabela do BFCL. Nada bloqueia.
4. GLM-5.3-Flash na GPU (prebuilts llama.cpp ROCm da imagem Unsloth) + claim de 3,3x do MTP.
5. A2 (argumentos + chamadas paralelas) — único eixo que ainda separa os fine-tunes.
6. Suíte B refeita com thinking off (fix D1).

## Novo eixo proposto — perfil de prefill (2026-09-12)

Pergunta do dono: dá pra chegar a 2.000 tok/s de **prefill** no nosso Qwen3.8-27B, como o
demo do Qwen3.8-Flash-Next em mlx-serve no M5 Max alcançou?

O que falta medir: **prefill single-stream em função do contexto** (TTFT a 1k / 16k / 64k /
128k / 256k). Só temos TTFT de prompt curto (53 ms, `results/baseline-01.json`) e throughput
**agregado** de prompt visto nos logs de engine (~1.4–2.8k tok/s com 4–16 requisições
concorrentes) — nenhum dos dois é comparável à afirmação deles.

Por que provavelmente alcançamos: o nosso é **dense 27,8B** e o deles é **MoE** (a descrição
da quantização separa "dense" de "experts"), mas o MI300X tem 5,3 TB/s de HBM3 e cerca de
1,3 PFLOPS BF16 denso — ordens de magnitude acima de um M5 Max. Prefill é limitado por
compute, então o hardware favorece muito a gente.

Por que não é maçã-com-maçã: modelo diferente (dense vs MoE), contexto diferente (16k vs
nossos até 256k), stack diferente (mlx-serve vs vLLM ROCm). Comparar os dois números direto
seria desonesto nos dois sentidos.

Entregável se rodar: curva de TTFT × contexto + tok/s de prefill por faixa, em
`results/prefill/`. É benchmark barato (poucos minutos de GPU), rende gráfico pro README e
mostra competência de infraestrutura.

### Plano de medição (aprovado 2026-09-12)

Script **já escrito e validado**: `evals/measure_prefill.py` (sintaxe e lógica conferidas
sem GPU). Basta subir pro droplet e rodar — zero tempo de montagem quando a GPU voltar.

**Níveis de contexto (em tokens, medidos de verdade via `usage.prompt_tokens`):**

| Alvo | Para que serve |
|---|---|
| 1.000 | baseline curto, comparação com o TTFT de 53 ms que já temos |
| 4.000 | faixa de chat normal |
| **16.000** | **o ponto exato do demo deles** — é o número que interessa comparar |
| 64.000 | faixa de RAG / documento longo |
| 128.000 | o mínimo que o dono exigiu pro uso real |
| 256.000 | teto do modelo, o caso extremo |

**Cada nível é medido de duas formas:**

- **COLD** — prompt único, com salt aleatório no INÍCIO (isso invalida o prefix cache, então
  o prefill é pago inteiro). **É este o número honesto.**
- **WARM** — o mesmo prompt reenviado; mede o ganho do prefix cache (o vLLM roda com
  `enable_prefix_caching=True`). Já vimos 87% de hit rate sob carga, mas nunca por faixa de
  contexto.

**Controle de qualidade da medida:** `max_tokens=1` + `stream=True`, então o TTFT isola o
prefill e nenhum decode contamina. `PREFILL_REPEATS=3` por nível, reportando mediana e
min/max. Se um nível estourar o limite do modelo, o script registra o erro e segue.

**Armadilhas que o script já trata:** sem o salt no início, o nível seguinte reaproveitaria o
prefixo do anterior e mediria cache em vez de prefill — o resultado sairia bonito e falso.

**Saída:** `results/prefill/prefill_profile.json` (bruto por repetição + mediana) + um gráfico
de TTFT × contexto pro README.

**Regra de honestidade na publicação:** reportar como **"nosso perfil de prefill"**, com as 3
diferenças declaradas (dense vs MoE, 16k vs 256k, vLLM ROCm vs mlx-serve). Nunca como
"ganhamos do demo do M5" — seria desonesto nos dois sentidos.
