#!/bin/bash
# Reinicia a matriz tau3 com a ordem corrigida (todos os airline antes dos retail).
# Existe como script proprio porque, ao rodar o kill direto por ssh, o proprio comando ssh
# contem o nome do arquivo e o `pkill -f` mata a sessao no meio (ja aconteceu 3x).
set -u
pkill -9 -f "matrix_tau3" 2>/dev/null
pkill -9 -f "tau2.cli" 2>/dev/null
sleep 5

# limpa resultados da rodada com ordem antiga (nao confundir com a nova)
rm -rf /root/tools-tau3-bench/data/simulations/*t3*

cd /root/scripts2 || exit 1
nohup bash matrix_tau3.sh > /root/matrix_nohup.log 2>&1 &
echo "MATRIZ_REINICIADA_PID=$!"
sleep 40
echo "--- primeiras linhas ---"
head -10 /root/tools-tau3-bench/matrix_tau3.log 2>/dev/null | tail -6
