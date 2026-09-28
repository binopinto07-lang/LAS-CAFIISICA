# LAS-CAFIISICA

Aplicação desktop Windows para extração e reconstrução de terreno a partir de nuvens LAS/LAZ.

## Ground Engine V2

A branch `ground-engine-v2` mantém o SMRF existente como motor Legacy e acrescenta:

- **Hybrid** — modo predefinido; Adaptive PTD como autoridade geométrica e CSF como segunda opinião;
- **Adaptive PTD** — seeds robustos multiescala, Delaunay TIN, distância real ponto→plano e densificação progressiva;
- **CSF** — superfície independente cloth-like para validação;
- **SMRF Legacy** — motor anterior preservado;
- análise automática de spacing/densidade;
- filtragem conservadora de outliers por plano local;
- proteção de descontinuidades por mudança de normal entre triângulos;
- deteção de gaps suportados, de bordo, grandes/desconhecidos e de descontinuidade;
- reconstrução apenas de gaps suportados;
- pontos reconstruídos marcados como LAS `synthetic`;
- **EXPORT GROUND ONLY** para exportar apenas solo real e, opcionalmente, solo reconstruído;
- viewport Original / Final Ground / Comparar.

A classificação original do LAS é preservada mas não é usada como input para decidir o novo terreno.

## Objetivo

A saída Ground Only deve conter:

- pontos medidos aceites como terreno;
- pontos sintéticos apenas em gaps geometricamente suportados;
- sem árvores, copas, arbustos, vegetação suspensa, edifícios, veículos ou ruído.

É preferível deixar um gap sem preencher do que inventar uma superfície sem suporte.

## Desenvolvimento local

```text
python -m pip install -e ".[dev]"
pytest
las-cafiisica
```

## Local Build Manager

Esta branch usa:

```text
LAS_CAFIISICA_GROUND_V2_2026_09_R3
```

O projeto continua sem depender de GitHub Actions.
