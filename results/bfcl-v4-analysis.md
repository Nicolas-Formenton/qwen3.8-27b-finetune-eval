# BFCL v4 — resultados (48 runs, 2026-09-10)

Matrix completa: 6 variantes x 4 categorias x 2 modos. Dados brutos em
`results/eval-04-bfcl-v4.json`; scores por item em `evals/bfcl-scores/`.

## O que cada coluna mede

| Categoria | O que testa |
|---|---|
| `multiple` | várias chamadas **em sequência** numa única resposta (argumentos conferidos por AST) |
| `parallel` | várias chamadas **em paralelo** |
| `parallel_multiple` | ambas as formas combinadas |
| `irrelevance` | **abstenção**: a pergunta NÃO exige ferramenta; o correto é não emitir chamada |

## Modo FC (template nativo de tool-call)

| Variante | multiple | parallel | parallel_multiple | irrelevance |
|---|---|---|---|---|
| base | 95.5 | 89.5 | 87.5 | 68.3 |
| +OH | 95.0 | 89.5 | 88.5 | 67.5 |
| **+xLAM** | 95.5 | **93.0** | **90.0** | **5.8** |
| D-base | 96.0 | 89.5 | 87.0 | 68.3 |
| D-OH | 95.5 | 89.5 | 87.0 | 68.8 |
| **D-xLAM** | 95.0 | 91.0 | 89.0 | **87.1** |

## Modo prompt (dialeto do próprio benchmark)

| Variante | multiple | parallel | parallel_multiple | irrelevance |
|---|---|---|---|---|
| base | 86.5 | 87.5 | 85.5 | 66.2 |
| +OH | 86.5 | 86.5 | 85.5 | 67.1 |
| **+xLAM** | **10.0** | **1.5** | **6.0** | 76.7 |
| D-base | 88.0 | 87.0 | 84.5 | 65.4 |
| D-OH | 86.5 | 88.0 | 85.5 | 67.1 |
| **D-xLAM** | **89.0** | 88.0 | 84.0 | **90.0** |

Contagens absolutas do `irrelevance` (n=240): base 164, xLAM 14, D-xLAM 209.

## Achados

### 1. SFT no xLAM destrói a abstenção (FC: 68.3 -> 5.8)

Todos os 226 erros do xLAM em `irrelevance` têm o mesmo tipo:
`irrelevance_error:decoder_success` — o modelo emitiu uma chamada **sintaticamente
válida** para uma pergunta que não precisa de ferramenta nenhuma.

Mecanismo: o dataset xLAM-60k é composto de pares pergunta->chamada, praticamente
sem exemplos de recusa. Treinar nele ensina a **sempre** chamar. No modo prompt o
efeito aparece como falha de dialeto; no modo FC, onde o formato não é obstáculo,
o defeito comportamental fica visível.

### 2. SFT no xLAM melhora as chamadas múltiplas (FC) — NÃO CONFIRMADO

`parallel` 89.5 -> 93.0 e `parallel_multiple` 87.5 -> 90.0. **Correção:** com n=200
por célula isso **não é estatisticamente significativo** (z=-1.24, p=0.21 e z=-0.79,
p=0.43). A versão anterior deste documento afirmava que o treino "melhorou a
capacidade alvo"; está errado e fica retirado. O único eixo em que o xLAM difere do
base de forma detectável é a abstenção.

### 2b. Significância (teste z de duas proporções, modo FC)

| Comparação | Eixo | Diferença | p | Veredito |
|---|---|---|---|---|
| +xLAM vs base | abstenção | **-62,5pp** | < 1e-4 | significativo |
| D-xLAM vs base | abstenção | **+18,8pp** | < 1e-4 | significativo |
| D-xLAM vs +xLAM | abstenção | **+81,2pp** | < 1e-4 | significativo |
| D-OH vs base | abstenção | +0,4pp | 0,92 | ruído |
| +OH, D-base, D-OH vs base | múltiplas/paralelas/abstenção | ≤1,5pp | todos > 0,7 | ruído |

Resumo: das 6 variantes, **apenas o D-xLAM** difere do base de forma detectável, e
**apenas num eixo** (abstenção). Nos outros três eixos todas as variantes são
estatisticamente empatadas.


### 3. O mesmo adapter quebra o dialeto do benchmark (prompt: 86.5 -> 10.0)

`+xLAM` cai de 86.5 para 10.0 em `multiple` e de 87.5 para 1.5 em `parallel`.
Degradação massiva e específica do formato, não da capacidade — no modo nativo o
mesmo modelo é o **melhor** dos seis em `parallel`.

### 4. A destilação de raciocínio sobre o xLAM reverte tudo

`D-xLAM` (distill sobre o merged xLAM) recupera a abstenção para **87.1%**, acima
do próprio base (68.3%), e volta a ~88 no modo prompt. Ou seja: a especialização
excessiva do SFT foi em grande parte desfeita por 1500 exemplos de traço de
raciocínio na etapa seguinte.

**Hipótese (não verificada):** os traços de raciocínio contêm muitos exemplos em
que o modelo conclui que nenhuma ferramenta é necessária, restaurando o
comportamento de recusa. Testar exigiria contar a frequência de recusas no dataset
de distill e/ou treinar um D-xLAM com dados filtrados só por isso.

### 5. As outras variantes não se movem

`+OH`, `D-base` e `D-OH` ficam dentro de ~1.5pp do base em quase tudo. Com n=200
por célula, isso é indistinguível de ruído.

## Ressalvas metodológicas

- **Os dois modos não são comparáveis entre si.** O gap é o achado, não um erro.
- **`irrelevance` no modo prompt premia quem não consegue emitir chamada nenhuma.**
  O 76.7 do `+xLAM` ali é, em boa parte, silêncio por incapacidade de dialeto, e
  não abstenção informada. Ler sempre junto com o modo FC.
- **n=200 por célula** (~±6pp de IC 95%). Diferenças abaixo disso não são afirmadas.
- Parte dos erros do base no modo prompt são cópia literal do placeholder da
  instrução (`func_name1(params_name1=params_value1)`) — ver `docs/FAILURES.md` #7.
