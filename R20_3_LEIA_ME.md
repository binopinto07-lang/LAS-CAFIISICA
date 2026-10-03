# R20.3 — Breakline-Safe Ground Recovery

## Objetivo

Reduzir os pequenos vazios do FINAL GROUND nas faces e transições dos socalcos sem transformar uma quebra de terreno num corredor de propagação.

## Alteração

A R20.3 preserva o manto R20/R20.1 e a recuperação medida R20.2, mas conserva a máscara de breaklines do manto.

Uma célula medida numa breakline pode ser recuperada como **face terminal** quando:
- existe observação física;
- está próxima da referência PTD;
- está próxima do manto;
- não está bloqueada pelo veto de telhado/copa;
- a dispersão dentro da célula é limitada.

A célula de breakline **não pode ser usada como ponte** para propagar Ground para o outro lado da descontinuidade.

## Não fazer

- Não criar Ground em células sem observação.
- Não exportar o manto inferido como Ground medido.
- Não remover o veto de telhados/copas.
- Não usar class 2, last/only return ou ponto-order como autoridade.

## Validação

O teste de campo deve comparar a mesma câmara:
1. ORIGINAL
2. FINAL GROUND
3. MANTO R20.3

A meta é reduzir os pequenos vazios reais, mantendo fora linhas/pontos de infraestrutura e vegetação elevada.

Estado: experimental até BUILD + TESTES + campo.
