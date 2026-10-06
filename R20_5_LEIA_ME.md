# LAS-CAFIISICA R20.5 — OBJECT VETO + MDT PREVIEW

## Estado

**EXPERIMENTAL — requer BUILD + TESTES no LocalBuildManager e validação de campo P1/L3.**

Branch: `r20-5-ground-veto-mdt-preview`

Fonte esperada pelo LocalBuildManager:

`LAS_CAFIISICA_GROUND_V2_2026_10_R20_5`

Versão: `0.12.5-dev`

## Problema observado na R20.4

Em P1 de alta densidade, algumas superfícies acima do terreno (vegetação,
arbustos, telhados/objetos ou pequenas ilhas elevadas) podiam acompanhar o
próprio manto/PTD e permanecer no FINAL GROUND.

A R20.5 não trata P1 com outro classificador. O pipeline continua universal.
A correção acrescenta uma prova geométrica independente do manto para poder
REJEITAR uma superfície elevada mesmo quando o manto passa sobre ela.

## R20.5 — veto multiescala de ilhas elevadas

Novo módulo:

`src/las_classifier/terrain/elevated_surface_guard.py`

Princípios:

- usa apenas células que contêm retornos/pontos realmente medidos;
- não usa a classe 2 original como verdade;
- não depende do nome do sensor;
- compara o envelope baixo da célula com planos tangentes de vizinhos medidos;
- usa distâncias multiescala;
- exige suporte em várias direções e pelo menos um par de direções opostas;
- protege breaklines;
- a máscara resultante entra no veto ANTES da continuidade Ground;
- uma célula bloqueada não pode servir de ponte de recuperação.

O objetivo é rejeitar uma "ilha" elevada cercada por terreno mais baixo sem
confundir uma face inclinada/talude real com um objeto.

## Pipeline universal

```text
LAS/LAZ
  -> PTD
  -> evidência espacial densa (todos os pontos)
  -> manto invertido
  -> veto R20.1
  -> NOVO veto geométrico R20.5 de superfícies elevadas
  -> continuidade Ground breakline-safe
  -> FINAL GROUND medido
  -> MDT em memória
  -> PREVIEW 3D
  -> exportação explícita apenas após validação
```

P1, L3 e UNKNOWN continuam no mesmo procedimento.

## MDT: alteração de fluxo

Na R20.4, o botão de MDT começava por pedir o caminho de exportação.

Na R20.5:

1. **CRIAR MDT (PREVIEW)** calcula o MDT em memória.
2. A viewport ganha **MDT 3D**.
3. Pode alternar:
   - ELEVAÇÃO MDT;
   - HILLSHADE;
   - OBSERVAÇÃO.
4. Só depois fica disponível **EXPORTAR MDT VALIDADO**.

A exportação usa exatamente o objeto MDT que foi pré-visualizado; não volta a
classificar ou a recalcular terreno ao gravar.

Estado de observação:

- 0 = NO_GROUND_OBSERVATION;
- 1 = MEASURED_GROUND;
- 2 = INTERPOLATED_MDT.

O estado 2 existe apenas no raster. Nunca é promovido a ponto LAS Ground medido.

## Comparador

`Universal Ground R20.4` permanece no seletor para comparação A/B de campo.
A opção predefinida passa a ser `Universal Ground R20.5`.

## Testes acrescentados

- objeto/roof elevado sobre superfície inclinada;
- talude plano inclinado não pode ser vetado;
- degrau unilateral de socalco não pode ser confundido com ilha suspensa;
- breakline protegida;
- MDT preview não grava TIFF/JSON;
- exportação só acontece explicitamente;
- estado interpolado continua diferente de Ground medido;
- mudança para MDT não altera automaticamente a câmara.

## Validação de campo obrigatória

Para a `soalheira.las` P1 usada no diagnóstico:

1. executar R20.4 e guardar screenshot FINAL GROUND;
2. executar R20.5 na mesma câmara;
3. verificar os objetos que apareciam acima do solo;
4. comparar MANTO (não é autoridade Ground);
5. criar MDT PREVIEW;
6. ver ELEVAÇÃO / HILLSHADE / OBSERVAÇÃO;
7. só se estiver correto exportar GeoTIFF;
8. repetir numa L3 para garantir que os taludes reais não ficaram excessivamente cortados.

Não considerar R20.5 validada apenas por pytest.
