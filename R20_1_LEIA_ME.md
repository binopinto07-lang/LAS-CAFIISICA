# LAS-CAFIISICA R20.1 — Mantle-guided Object Veto

**Estado:** EXPERIMENTAL, por validar com `cloud0.las` em Windows  
**Branch:** `r20-1-mantle-veto`  
**Versão:** `0.11.1-dev`  
**Source Guard:** `LAS_CAFIISICA_GROUND_V2_2026_10_R20_1`  
**CRS:** ETRS89 / Portugal TM06 (**EPSG:3763**)

## Problema real observado pelo utilizador

Comparação R20 ORIGINAL / FINAL GROUND / MANTO mostrou **telhados e copas altas classificados como Ground**. O próprio manto pode seguir um telhado sem observação por baixo, logo distância do ponto ao manto **não basta** para vetar edifícios. O manto R20 original tem utilidade para os socalcos e deve ser preservado.

## Implementação isolada

- `src/las_classifier/terrain/mantle_veto.py`: `MantleVeto` sobre o manto R20, sem alterar a geometria do cloth.
- `src/las_classifier/classifiers/l3_mantle_veto.py`: motor `L3 Inverted Ground R20.1`; R20, R19.1 e R18 permanecem disponíveis na UI.
- **Height veto**: retorna veto para pontos já aceites a mais de **0,42 m de distância normal acima** de uma célula com referência de manto fiável (inclinação tangente local compensada). Este limiar é inicial e exige teste de campo.
- **Roof-candidate veto**: deteta células elevadas relativamente a retornos medidos em pelo menos **3 das 4 direções cardeais**, em várias distâncias, com superfície relativamente plana, dispersão vertical contida e continuidade. Alarga apenas a patamares locais com elevação medida semelhante. Esta é uma hipótese geométrica de cobertura, **não** deteção semântica certificada de edifícios.
- **Canopy-candidate veto**: células elevadas face ao terreno em 3 direções e dispersão vertical elevada. Pontos acima de Ground quando existe retorno inferior são apanhados preferencialmente pelo height veto.
- **Pós-decisão obrigatório**: o veto atua *após* PTD/R18 e recuperação R20, incluindo pontos antigos da classe 2 incorretamente aceites, e nunca classifica inexistentes como Ground.
- `GroundRejectReason`: códigos adicionais `MANTLE_HEIGHT_VETO=11`, `ROOF_CANDIDATE_VETO=12`, `CANOPY_CANDIDATE_VETO=13`, com prioridade no diagnóstico exclusivo.
- `GroundProvenance`: bit `PROV_MANTLE_VETO` para rastrear alterações internas.
- Contadores na UI/log `R20_1_VETO` para pontos retirados e células candidatas.

## MANTO R20.1 na viewport e no LAZ diagnóstico

O cálculo original do manto é o mesmo da R20. A camada diagnóstica `MantleState` acrescenta:
- 1: superfície observada com suporte (verde)
- 2: pequeno vazio inferido, não medido (azul)
- 3: ambíguo (cinzento)
- 4: possível ausência de observação Ground (roxo)
- **5: hipótese de cobertura elevada (vermelho)**
- **6: hipótese de copa elevada (laranja)**

Todos os pontos de `EXPORT MANTO (LAZ)` continuam **LAS class 0**, `synthetic=1`; nunca contam como Ground real. `EXPORT GROUND ONLY` continua exclusivamente com retornos medidos aceites (synthetic=0), sem manto artificial. A câmara Potree não deve ser reenquadrada ao mudar de modo.

## Testes incluídos

- `tests/test_r20_1_mantle_veto.py`: telhado elevado mesmo se o PTD o aceitou; copa sem retorno inferior; vegetação acima de retorno Ground; talude natural; socalco longitudinal; fora da nuvem; invariância da ordem das consultas; aviso na legenda do manto.
- `tests/test_r20_1_decision_veto.py`: rejeição efetiva após decisão PTD/classe 2, sem reaceitar e sem mexer em inválidos.
- Preservar e executar todos os testes R20/R19 anteriores, testes Viewer/câmara, Loader, export, build e SOURCE GUARD.

## Protocolo de campo

1. Descarregar `r20-1-mantle-veto.zip` e extrair em pasta NOVA.
2. Executar `START_BUILD_MANAGER.bat` → **BUILD + TESTES**; confirmar `0.11.1-dev` e source guard acima.
3. Abrir `cloud0.las`; engine `L3 Inverted Ground R20.1`, `Balanced`, `Mountain / Talude`, `Auto detect`, sintético OFF.
4. Comparar a **mesma câmara** em ORIGINAL / FINAL GROUND / MANTO R20.1 e comparar com a R20 antiga: edifícios, árvores, patamares, crista e pé de talude.
5. Na viewport MANTO R20.1, verificar se as áreas candidatas aparecem vermelhas/laranja; enviar screenshots e log incluindo `R20_1_MANTLE_VETO`, `R20_1_VETO`, `R20.1_REJECT_REASONS`.
6. Verificar o LAZ diagnóstico e o LAZ Ground Only num visualizador externo (EPSG, contagens, RGB, dimensões/proveniência).
7. **Critérios críticos:** diminuir telhados/copas erradamente aceites SEM voltar a abrir buracos nos socalcos. Se o veto eliminar terreno legítimo em descontinuidades, não relaxar globalmente nem aprovar esta versão para produção.

## Limitações e guardrails

- Regras conservadoras de candidatos geométricos 2.5D; edifícios encostados a montes, coberturas muito extensas, rocha em patamares isolados e árvores sem retorno Ground podem continuar ambíguos. Nunca declarar identificação 100%.
- A presença de classe 2, o last/only return ou um TIN que acompanha um telhado **não prova Ground**.
- NÃO criar Ground sintético, NÃO alterar a R20 original nem `main`, NÃO depender de GitHub Actions e NÃO reescrever Loader/Viewer/build.
- Esta versão **não foi validada no cloud0.las**; os ficheiros foram integrados no GitHub e aguardam BUILD/TESTS e inspeção de campo.
