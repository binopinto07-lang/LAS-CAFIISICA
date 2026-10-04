# LAS-CAFIISICA R20.4 — UNIVERSAL GROUND + MDT

## Estado

**EXPERIMENTAL — requer BUILD + TESTES no LocalBuildManager e validação de campo.**

Branch: `r20-4-universal-mdt`

Fonte: `LAS_CAFIISICA_GROUND_V2_2026_10_R20_4`

## Alteração principal

R20.4 mantém a geometria R20.3 e remove a restrição de entrada L3.

P1, L3 e fontes LAS/LAZ desconhecidas executam o mesmo pipeline:

```text
PTD
 -> dense spatial evidence
 -> inverted Ground mantle
 -> roof/canopy/elevated-object veto
 -> breakline-safe measured Ground continuity
 -> FINAL GROUND
 -> MDT
```

Não existe um classificador P1 alternativo.

Os campos de retorno são opcionais. Só contribuem como evidência quando a própria
nuvem contém uma população multi-return mensurável. Uma população fotogramétrica
1/1 fica neutra e não é tratada como prova de penetração até ao terreno.

## P1: Ground não observado

O mesmo procedimento não altera a física da aquisição. Se a P1 não observou
fisicamente o terreno sob vegetação, R20.4 não cria um ponto LAS Ground fictício.

O MDT pode preencher apenas pequenas lacunas configuradas e grava separadamente:

- 0 = NO_GROUND_OBSERVATION;
- 1 = MEASURED_GROUND;
- 2 = INTERPOLATED_MDT.

Estado 2 pertence ao raster; não é reclassificado como Ground medido no LAS.

## MDT

A UI inclui **CRIAR MDT DO GROUND** depois de executar o motor.

Defaults experimentais:
- resolução: 0.25 m;
- preenchimento máximo: 0.75 m;
- CRS: EPSG:3763.

Outputs:
- `*_MDT_R20_4.tif`;
- `*_MDT_R20_4_OBSERVATION_STATE.tif`;
- relatório JSON.

## Compatibilidade

R20.3 continua presente como comparador. `main` não é alterada.

## Validação obrigatória

1. START_BUILD_MANAGER.bat -> BUILD + TESTES.
2. Confirmar pytest completo, não apenas os testes R20.4.
3. Testar `cloud0.las`/L3 e uma nuvem P1 real.
4. Comparar ORIGINAL, MANTO, FINAL GROUND e MDT.
5. Verificar externamente o GeoTIFF EPSG:3763 e o LAS Ground exportado.
