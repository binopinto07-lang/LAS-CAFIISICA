# LAS-CAFIISICA R20.7 — LOW-SEED TIN GROWTH

## Objetivo

R20.7 muda a lógica de decisão do Ground medido para uma abordagem mais
conservadora, adequada a P1 e também aplicável a L3/UNKNOWN:

LOW-SEED GRID -> TIN -> crescimento iterativo -> veto de estruturas elevadas
-> reconstrução apenas onde o Ground não foi observado.

Não existe código Agisoft no projeto. A implementação é independente e usa
apenas o princípio geométrico documentado de começar por pontos baixos e
expandir Ground por distância/ângulo relativamente a uma superfície de terreno.

## Pipeline

```text
LAS/LAZ
 -> dense spatial evidence
 -> preserved R20 mantle
 -> coarse low-seed grid
 -> Delaunay terrain TIN
 -> normal-distance / normal-angle compatibility
 -> iterative connected Ground growth
 -> R20.6.2 elevated / vertical-structure veto
 -> FINAL measured Ground
 -> mantle reconstruction only where Ground is missing
 -> FINAL GROUND COMPLETE
 -> FAST MDT preview
```

## Regras importantes

- original LAS class 2 is never terrain authority;
- return number is not required;
- P1 and L3 use the same R20.7 pipeline;
- seed points are the physically measured LOW cell in each coarse XY block;
- cells far ABOVE the low-seed TIN are vetoed;
- cells below a coarse TIN are never rejected solely for being below it;
- breakline cells are protected from the new hard TIN veto;
- the R20 mantle remains read-only;
- reconstructed Ground remains provenance-separated from measured Ground.

## Default conservative parameters

- seed cell: 12 m;
- initial normal distance: 0.32 m;
- growth normal distance: 0.48 m;
- high-cell block threshold: 0.58 m;
- max local normal angle: 38 degrees;
- max growth iterations: 24;
- Ground erosion around hard elevated objects: 0.60 m.

These are experimental defaults for field testing, not final universal values.

## Ground Complete

R20.7 keeps the R20.6.2 reconstruction channel.

Measured Ground is classified first. If a cell is blocked as vegetation/object
or Ground is otherwise not observed, the reconstructed layer may fill terrain
from the preserved mantle / nearby accepted Ground while retaining
GroundSource=2.

## MDT

FAST MDT remains enabled. The preview is created from the already solved
Ground model instead of reclassifying the complete LAS/LAZ again.

Observation states remain:

- 0 = no Ground support;
- 1 = measured Ground;
- 2 = reconstructed Ground;
- 3 = raster-only interpolation.

## Comparator

`Universal Ground R20.6.2` remains selectable for A/B field comparison.

## Validation

1. BUILD + TESTS in LocalBuildManager.
2. On the same P1 camera view compare R20.6.2 and R20.7 FINAL GROUND.
3. Inspect vineyards/shrubs/objects above terrace surfaces.
4. Confirm steep true terrain is not cut.
5. Confirm reconstructed Ground fills rejected/no-observation zones below
   objects rather than reproducing object height.
6. Create MDT PREVIEW and inspect Elevation/Hillshade/Observation.
7. Repeat on L3.

R20.7 remains experimental until field-validated.
