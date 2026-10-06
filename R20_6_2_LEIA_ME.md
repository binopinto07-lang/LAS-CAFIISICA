# LAS-CAFIISICA R20.6.2 — STRONG VETO + FAST MDT

## Estado

Experimental. Validar primeiro com BUILD + TESTES e depois na mesma P1/L3 usada
nas revisões anteriores.

Branch: `r20-6-2-ground-complete-veto-mdt`

Source revision:

`LAS_CAFIISICA_GROUND_V2_2026_10_R20_6_2`

## Problema corrigido 1 — objetos acima do solo ainda apareciam no FINAL GROUND

Na R20.6, o FINAL GROUND já combinava Ground medido e Ground reconstruído.
Contudo, havia dois caminhos pelos quais vegetação/objetos podiam continuar
demasiado altos:

1. uma célula com forte estrutura vertical podia não entrar no veto de ilha
   elevada;
2. uma célula rejeitada pelo veto final podia ser reconstruída depois a partir
   do próprio manto dessa célula, recuperando novamente uma altura próxima do
   objeto.

### R20.6.2

O veto final passa a combinar:

- ilhas elevadas multiescala;
- espessura vertical normalizada pelo plano local;
- discordância com planos tangentes dos vizinhos.

A espessura é avaliada na direção normal ao terreno, por isso um talude muito
inclinado não é automaticamente confundido com vegetação.

O resultado do veto final também é passado para
`mantle_reconstruction.py`.

Quando uma célula vetada precisa de Ground reconstruído:

- a célula é tratada como Ground escondido;
- a cota visível do objeto NÃO é usada como cota de terreno;
- o terreno é projetado a partir de Ground medido/reliable vizinho;
- breaklines continuam protegidas.

O MANTO original continua sem ser alterado.

## Problema corrigido 2 — MDT Preview

Na R20.6.1 o MDT podia voltar a percorrer e reclassificar toda a LAS/LAZ. Em
`soalheira.las`, isso significava voltar a trabalhar sobre 318 milhões de
pontos apenas para criar uma pré-visualização já depois do Ground ter sido
resolvido.

A R20.6.2 adiciona um caminho FAST MDT:

`Ground model -> measured_ground_cells + mantle + reconstruction -> MDT`

Não existe uma segunda classificação da nuvem quando o modelo R20.6.2 já está
disponível.

O FAST MDT:

- usa o manto preservado onde existe Ground medido;
- usa a superfície de reconstrução onde o Ground é sintético;
- mantém os estados 0/1/2/3;
- cria o preview em memória;
- só escreve GeoTIFF depois de `EXPORTAR MDT VALIDADO`.

Estados:

- 0 = NO_GROUND_OBSERVATION;
- 1 = MEASURED_GROUND;
- 2 = RECONSTRUCTED_GROUND;
- 3 = INTERPOLATED_MDT.

## Visualização

O viewer mantém amostragem espacial XYZ independente da ordem dos pontos.
A cache é versionada, evitando reutilizar visualizações antigas feitas por
stride de índice.

## Proveniência

R20.6.2 usa `GroundMethod = 14`.

Ground reconstruído continua identificado separadamente de Ground medido.

## Validação recomendada

Na P1 `soalheira.las`:

1. executar R20.6.2;
2. confirmar que o MANTO continua igual ao manto preservado;
3. abrir FINAL GROUND;
4. verificar especialmente videiras/arbustos/objetos acima dos patamares;
5. confirmar que zonas rejeitadas podem ser reconstruídas abaixo do objeto;
6. premir `CRIAR MDT (PREVIEW)`;
7. confirmar que o preview aparece sem segunda classificação longa;
8. alternar ELEVAÇÃO / HILLSHADE / OBSERVAÇÃO;
9. só depois exportar o MDT.

Não considerar validado apenas por testes unitários.
