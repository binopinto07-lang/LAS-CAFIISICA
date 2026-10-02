# LAS-CAFIISICA — R20.2 Measured Ground Continuity Recovery

**Versão:** `0.11.2-dev`  
**Branch:** `r20-2-continuity-recovery`  
**Estado:** experimental, necessita BUILD + TESTES e ensaio visual no `cloud0.las`  
**Source Guard:** `LAS_CAFIISICA_GROUND_V2_2026_10_R20_2`  
**CRS:** ETRS89 / Portugal TM06, EPSG:3763  

## Objetivo aceite

Minimizar faixas pretas que sejam **Ground fisicamente medido mas rejeitado**, sobretudo em socalcos, faces inclinadas, pés e cristas, preservando a exclusão de edifícios/copas da R20.1. O resultado Ground Only **não contém pontos sintéticos/inferidos**. Se não houver qualquer medição real de terreno, essa zona não é falsamente considerada retorno Ground.

## Nova arquitetura, sem substituir componentes estáveis

- `src/las_classifier/terrain/ground_continuity.py`: grafo raster de células com pontos reais (nunca células inferidas). Partida em âncoras PTD fortes em células fiáveis do manto, excluindo candidatos a roof/canopy da R20.1. Expansão iterativa limitada por 8 vizinhos medidos, ângulo entre normais locais, continuidade **3D por distância ponto-plano nos dois sentidos** e máximo salto vertical por aresta. Não há amostragem por ordem LAS para decidir conectividade.
- `src/las_classifier/classifiers/l3_ground_continuity.py`: motor novo `L3 Ground Continuity R20.2`, mantendo na seleção `L3 Inverted Ground R20.1`, `L3 Inverted Ground R20`, R19/R18 e os restantes.
- A nova recuperação de retornos só é aplicada DEPOIS da decisão PTD, recuperação pelo manto R20 e pós-veto R20.1.
- Cada retorno real candidato deve estar dentro de célula ligada, não bloqueada por roof/canopy, passar **novamente pelo veto R20.1** e ficar perto do plano tangente local medido segundo distância normal (limites de partida 0,36 m acima, 0,22 m abaixo; +0,07 m apenas para face inclinada). Estes são parâmetros experimentais a validar no `cloud0.las`, não parâmetros universais.
- Preserva o manto R20 inalterado (leitura em R20.2); não altera limiares/class2/last-return para forçar uma percentagem artificial de Ground.

## Metadados e estatísticas

- `GroundDecision = 8`: `L3_GROUND_CONTINUITY_RECOVERED`.
- `GroundSource = 6`: retorno físico real recuperado via continuidade R20.2. NÃO significa inferido/sintético.
- `GroundMethod = 8` e novo bit `PROV_GROUND_CONTINUITY`.
- UI `R20.2 MEASURED GROUND CONTINUITY` apresenta retornos recuperados, classe2 recuperada, células iniciais, ligadas e expandidas, células bloqueadas por roof/canopy.
- Logs: `R20_2_CONTINUITY`, `R20_2_RECOVERY`, `R20.2_REJECT_REASONS`, `R20_1_VETO`.
- No `GROUND ONLY` são escritos exclusivamente pontos do ficheiro de origem aceites pelas decisões, nunca o manto inferido; o LAZ diagnóstico MANTO continua separado e assinalado como artificial.

## Testes sintéticos adicionados

`tests/test_r20_2_ground_continuity.py` abrange:
1. uma faixa com retornos medidos que a R20 antiga rejeita por ausência de PTD local e a R20.2 liga a âncoras;
2. uma fila sem pontos físicos (não se cria Ground e não se atravessa a falha);
3. máscara roof/canopy bloqueia a propagação mesmo com classe2/PTD errados;
4. ponto Ground perto da superfície recuperado, copa um metro acima excluída;
5. talude acentuado tratado segundo distância **normal 3D**, não só diferença Z;
6. invariância da ordem dos retornos consultados;
7. integração pós-veto, incluindo inválidos e rastreio GroundDecision/GroundSource/proveniência.

Manter regressões R20.1: telhado, copa, vegetação, socalco longitudinal, Viewer/câmara, LAS/LAZ, build/source guard e exportação.

## Procedimento de avaliação de campo

1. Descarregar o ZIP da branch e **extrair numa nova pasta**, para não reutilizar caches/código R20.1.
2. Abrir `START_BUILD_MANAGER.bat` → executar **BUILD + TESTES**; guardar o log completo.
3. Abrir `cloud0.las`, escolher `L3 Ground Continuity R20.2`, Balanced, Mountain / Talude, Auto detect, sintético OFF.
4. Comparar na mesma posição de câmara ORIGINAL → FINAL GROUND → MANTO R20.2. Registar percentagem Ground R20.1 versus R20.2 **e as regiões onde aumentou**.
5. Verificar visualmente: socalcos e faces reais recuperados; telhado continua excluído; copas altas não voltam a Ground; não aparecem pontes fictícias em áreas sem retorno LiDAR.
6. Exportar Ground Only e abrir externamente para confirmar EPSG:3763, número de retornos, `GroundSource=6`, `GroundMethod=8` e `synthetic=0`.
7. Enviar o log `R20_2_CONTINUITY` / `R20_2_RECOVERY` / `R20.2_REJECT_REASONS` e três imagens equivalentes às da R20.1. Sem estes testes NÃO declarar motor validado.

## Limitações assumidas

Continuidade de geometria de grelha 2.5D com comparação 3D de planos locais, não uma malha volumétrica multi-superfície completa. Uma parede verdadeiramente vertical/multi-Z na mesma célula continua um desafio distinto; não forçar toda a cobertura a verde nem aceitar telhados para fechar vazios. Um vazio de observação não é recuperável através de retornos físicos inexistentes.
