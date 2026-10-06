# LAS-CAFIISICA R20.6 — GROUND COMPLETE

## Objetivo

Para P1, o FINAL GROUND não deve ficar cheio de buracos apenas porque o terreno
real não foi observado pela fotogrametria sob vegetação/objetos.

A R20.6 mantém duas verdades separadas:

- **GROUND MEDIDO**: pontos realmente observados e aceites pelo classificador;
- **GROUND RECONSTRUÍDO**: pontos sintéticos gerados apenas nas células que o
  MANTO marca como `NO_GROUND_OBSERVATION`.

O MANTO R20.4 continua preservado e não é alterado.

## Pipeline

```text
LAS/LAZ P1, L3 ou UNKNOWN
 -> classificação medida universal
 -> MANTO R20.4 preservado
 -> veto de objetos elevados
 -> continuidade medida
 -> GROUND MEDIDO
 -> reconstrução do MANTO apenas em NO_GROUND_OBSERVATION
 -> FINAL GROUND = medido + reconstruído
 -> MDT PREVIEW
 -> EXPORTAR MDT VALIDADO
```

## Reconstrução

Novo módulo:

`src/las_classifier/terrain/mantle_reconstruction.py`

São elegíveis:

1. `mantle.inferred`: células vazias pequenas suportadas pelo manto;
2. `mantle.possible_no_ground_observation`: existem retornos superiores, mas
   não existe observação física de Ground.

As breaklines nunca são sintetizadas.

Uma célula de telhado/copa não pode ser usada como âncora medida para construir
Ground escondido. Para `possible_no_ground_observation`, a altura é projetada
a partir do Ground medido/reliable mais próximo, em vez de copiar simplesmente
a altura do objeto visível.

## Proveniência LAS

O GROUND reconstruído é exportado como classe 2 para poder ser usado como
terreno, mas fica identificado com:

- `GroundSource = 2`;
- `GroundMethod = 13` para R20.6;
- confiança separada;
- não deve ser interpretado como observação física da P1/L3.

O botão **EXPORT GROUND** permite manter a reconstrução ativa por defeito na
R20.6. Se for desativada, exporta apenas o Ground medido.

## Visualização

O FINAL GROUND mostra automaticamente:

`GROUND MEDIDO + GROUND RECONSTRUÍDO`.

O MANTO continua a ser uma camada diagnóstica separada.

## MDT

O MDT R20.6 tem quatro estados:

- 0 = sem Ground;
- 1 = Ground medido;
- 2 = Ground reconstruído do MANTO;
- 3 = interpolação raster adicional do MDT.

Na vista OBSERVAÇÃO:

- verde = medido;
- azul = reconstruído;
- amarelo = interpolado apenas no raster.

## Validação de campo

Na `soalheira.las` P1:

1. executar R20.6;
2. confirmar que o MANTO continua como R20.4;
3. abrir FINAL GROUND;
4. verificar se os buracos correspondentes a NO_GROUND_OBSERVATION foram
   preenchidos;
5. confirmar que vegetação/objetos reais não voltaram como pontos medidos;
6. criar MDT PREVIEW;
7. comparar Elevação/Hillshade/Observação;
8. só depois exportar.

A R20.6 é experimental até passar BUILD + TESTES no Windows e validação visual.
