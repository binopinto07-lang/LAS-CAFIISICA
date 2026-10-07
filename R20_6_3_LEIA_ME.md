# LAS-CAFIISICA R20.6.3 — CRS-SAFE MDT

## Estado

Patch de robustez sobre a R20.6.2. O motor Ground, os thresholds, o manto,
os vetos e a reconstrução não são alterados nesta revisão.

Branch: `r20-6-3-crs-safe-mdt`

Source revision:

`LAS_CAFIISICA_GROUND_V2_2026_10_R20_6_3`

## Problema observado em campo

Numa nuvem P1 com 110 523 522 pontos, o Ground R20.6.2 terminou e o viewer
carregou FINAL GROUND e MANTO. O preview MDT falhou depois ao executar
`LasHeader.parse_crs()`.

O header da LAS originou:

`Invalid projection: EPSG:11108`

embora a aplicação já tivesse estabelecido o CRS de trabalho como
`EPSG:3763`.

## Correção R20.6.3

O CRS autoritativo do LAS-CAFIISICA continua a ser:

`WORKING_CRS = EPSG:3763`

O CRS declarado no header LAS passa a ser apenas diagnóstico no fluxo do MDT.

Regras:

1. o MDT usa sempre o WORKING_CRS validado pela aplicação;
2. `parse_crs()` fica protegido contra GeoKeys inválidos;
3. CRS ausente, inválido ou diferente não aborta o preview;
4. o GeoTIFF e o mapa de observação são escritos explicitamente em EPSG:3763;
5. o LAS de origem nunca é modificado;
6. o log distingue WORKING_CRS de CRS do header.

Mensagens esperadas:

- `MDT_CRS_SOURCE=WORKING_CRS MDT_CRS=EPSG:3763`
- `LAS_HEADER_CRS_OK=EPSG:3763`, quando válido;
- `LAS_HEADER_CRS_INVALID=...`, quando o GeoKey é inválido;
- `LAS_HEADER_CRS_IGNORED=...`, quando é diferente.

## Teste de regressão

Foi acrescentado um teste que força `LasHeader.parse_crs()` a lançar uma
exceção equivalente ao erro real EPSG:11108.

O teste exige que:

- o preview MDT continue a ser produzido;
- existam elevações válidas;
- o log confirme EPSG:3763 como WORKING_CRS;
- o erro EPSG:11108 seja apenas diagnóstico.

## Validação recomendada no Windows

1. executar `START_BUILD_MANAGER.bat`;
2. confirmar:
   `OK SOURCE: LAS_CAFIISICA_GROUND_V2_2026_10_R20_6_3`;
3. executar BUILD + TESTES;
4. abrir a mesma nuvem P1;
5. classificar com Universal Ground R20.6.2;
6. criar MDT Preview;
7. confirmar que o preview aparece mesmo com o GeoKey problemático;
8. confirmar no log:
   `MDT_CRS_SOURCE=WORKING_CRS MDT_CRS=EPSG:3763`;
9. só depois exportar o MDT validado.

Não alterar thresholds Ground nesta revisão.
