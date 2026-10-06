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


## Experimental R20 inverted Ground mantle

See [R20_LEIA_ME.md](R20_LEIA_ME.md) for the isolated R20 engine, separate inferred-surface diagnostic LAZ, source distinctions and mandatory field validation. Use branch `r20-inverted-mantle` and the integrated `START_BUILD_MANAGER.bat`.


## R20.1 — Mantle-guided object veto (experimental)

See [R20_1_LEIA_ME.md](R20_1_LEIA_ME.md). The separate branch `r20-1-mantle-veto` preserves the R20 cloth and adds a post-classification elevated-object veto for roof/canopy candidates. The integrated `START_BUILD_MANAGER.bat` selects the authoritative local profile revision 38.


## R20.2 — Ground Continuity Recovery (experimental)

See [R20_2_LEIA_ME.md](R20_2_LEIA_ME.md). Branch `r20-2-continuity-recovery` preserves the R20/R20.1 engines, the original inverted cloth, roof/canopy veto, EPSG:3763, viewer camera and measured-only Ground export. Open `START_BUILD_MANAGER.bat` and run BUILD + TESTES using integrated profile revision 39.


## R20.4 — Universal Ground + MDT (experimental)

Branch `r20-4-universal-mdt` promotes one geometry-first pipeline for P1,
L3/LiDAR and unknown LAS/LAZ sources. Sensor identity no longer blocks the
R20.3 mantle/veto/continuity procedure. Return metadata is optional evidence
only when the complete dense grid proves a meaningful multi-return population.

After classification the application can create an EPSG:3763 MDT plus a
separate observation-state GeoTIFF distinguishing measured Ground from
raster-only interpolation. See [R20_4_LEIA_ME.md](R20_4_LEIA_ME.md).


## R20.5 — Elevated-object veto + MDT preview (experimental)

Branch `r20-5-ground-veto-mdt-preview` keeps one universal P1/L3/UNKNOWN
classification pipeline but adds a measured-neighbour multiscale veto for
suspended/elevated surfaces that can otherwise ride the PTD/mantle. The new
veto is applied before continuity recovery and never uses original class 2 as
terrain authority.

The MDT workflow is now review-first: **CRIAR MDT (PREVIEW)** computes the
terrain in memory and shows a 3D mesh with Elevation, Hillshade and Observation
modes. **EXPORTAR MDT VALIDADO** is enabled only after a preview exists and
writes the exact reviewed MDT. See `R20_5_LEIA_ME.md`.


## R20.5.1 — Preserve Mantle hotfix (experimental)

Branch `r20-5-1-preserve-mantle` fixes a design error in the first R20.5:
the elevated-object mask no longer mutates `mantle.veto_guard`. The MANTO is
again built with the exact R20.4 builder and remains untouched. A separate
R20.5.1 veto blocks elevated islands only in continuity/FINAL GROUND. MDT
preview-before-export is retained. See `R20_5_1_LEIA_ME.md`.


## R20.6 — Ground Complete (experimental)

Branch `r20-6-ground-complete` keeps the preserved R20 mantle and separates
measured Ground from reconstructed Ground. FINAL GROUND now combines accepted
measured returns with synthetic class-2 points generated only for explicit
`NO_GROUND_OBSERVATION` mantle cells. Reconstructed points are marked with
`GroundSource=2` and R20.6 `GroundMethod=13`; they are not physical
observations. MDT preview now distinguishes measured, reconstructed and
raster-only interpolation states. See `R20_6_LEIA_ME.md`.
