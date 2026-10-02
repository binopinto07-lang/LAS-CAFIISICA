# LAS-CAFIISICA R20 — Inverted Ground Mantle (EXPERIMENTAL)

Branch: `r20-inverted-mantle`
Version: `0.11.0-dev`
SOURCE GUARD: `LAS_CAFIISICA_GROUND_V2_2026_10_R20`
CRS: **ETRS89 / Portugal TM06 — EPSG:3763**

## Objetivo

Recuperar **retornos físicos medidos** que foram rejeitados pelo TIN/PTD em taludes e socalcos. Construir um manto invertido a partir do envelope baixo dos retornos **sem inventar Ground medido**.

## Fluxo implementado

1. Mantém a base R19.1 com `DenseSpatialEvidenceGrid` construída em chunks sobre todos os retornos medidos elegíveis.
2. Obtém `min_z` por célula, separadamente do estado de validade geométrica PTD.
3. Faz inversão virtual `Zinv = Zref - Z`, reduz espículas baixas locais sem atravessar saltos grandes, e desce um manto com colisão sobre o envelope invertido.
4. Devolve o manto ao Z original e calcula a inclinação local.
5. Só recupera pontos LAS **existentes** quando estão próximos do manto em distância normal (0,12 m acima; 0,08 m abaixo) e quando há observações suficientes, proximidade a evidência PTD e ausência de salto marcado.
6. A decisão `GroundDecision=7`, `GroundSource=5`, `PROV_INVERTED_MANTLE` indica uma recuperação de retorno medido; não é sintético.
7. Identifica separadamente células `observed/reliable`, `inferred`, `ambiguous` e `possible_no_ground_observation`. Estas últimas são **hipóteses**, não confirmação da inexistência de Ground.

## Exportações distintas

- **EXPORT GROUND ONLY**: somente retornos LAS medidos classificados como Ground, `synthetic=0` no motor R20. NÃO inclui o manto inferido.
- **EXPORT MANTO (LAZ)**: nova camada de diagnóstico, toda LAS classe 0, com `synthetic=1`, EPSG:3763 e ExtraBytes `MantleState`:
  - 1: envelope observado com suporte;
  - 2: pequena célula sem retorno interpolada (não-observada);
  - 3: envelope observado mas ambíguo;
  - 4: suspeita de terreno não observado abaixo de retornos superiores.
- Campo `MantleAnchorM` indica distância ao suporte PTD.
- Não utilizar o manto diagnóstico como nuvem Ground medida.

## Como testar

1. Descarregar ZIP da branch e extrair para uma pasta nova.
2. Abrir `START_BUILD_MANAGER.bat` (Local Build Manager integrado V0.1.11).
3. Executar **BUILD + TESTES**; verificar `tests/test_r20_inverted_mantle.py` e `tests/test_r20_ground_decision.py`.
4. Abrir `cloud0.las`. Escolher `L3 Inverted Ground R20`, Balanced, Mountain / Talude, Auto detect; sintético fica desativado.
5. Comparar ORIGINAL, R19.1 e R20 a partir da mesma câmara. Guardar estatísticas e linhas `R20_INVERTED_MANTLE` / `R19_REJECT_REASONS` (o prefixo de log mantém-se por compatibilidade).
6. Na viewport clicar **MANTO R20** para gerar, quando solicitado, a nuvem Potree colorida; esta vista é independente e mantém a câmara. O modo COMPARAR sobrepõe apenas ORIGINAL e FINAL GROUND, nunca o manto inferido.
7. Opcionalmente exportar `_R20_MANTO_DIAGNOSTICO.laz` para inspeção externa; escolher RGB e ler a dimensão extra `MantleState`.
8. Validar separadamente vegetação, socalcos, cristas/pés, zonas sem pontos, densidade e desempenho.

## Limitações reconhecidas

- Primeira implementação experimental **2.5-D**: faces quase verticais/overhangs exigirão superfície local 3D e tratamento de duas faces antes de uma classificação segura.
- O `min_z` não é uma prova de Ground; canopy/objetos poderão limitar a recuperação. Os requisitos de densidade, proximidade de âncoras e proteção de descontinuidades pretendem reduzir falsos positivos, não garantir ausência deles.
- O manto inferido serve para diagnóstico de oclusões, **não** é preenchido/exportado no Ground Only.
- A R19.1 permanece intacta na branch `ground-engine-v2`, e R18 continua selecionável no motor histórico; `main` não é alterado.
- Ainda é obrigatória validação através de BUILD, testes e inspeção visual no `cloud0.las`.

## Comparadores e referências de código

- `src/las_classifier/classifiers/csf_engine.py` (ideia do pano sobre nuvem invertida; R20 não chama a classificação CSF como verdade).
- `src/las_classifier/terrain/dense_spatial_evidence.py`, `ground_evidence.py` e `ground_debug.py`.
- Princípios de módulos de responsabilidade única e nomenclatura explícita de clean-code-python, algoritmos determinísticos e padrões Python; sem copiar software proprietário.
