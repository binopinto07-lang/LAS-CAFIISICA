# LAS-CAFIISICA R20.5.1 — PRESERVE MANTLE

## Motivo

No primeiro R20.5, o novo veto de superfícies elevadas foi ligado por mutação
direta:

`mantle.veto_guard.roof_candidate |= mask`

Isso alterava o próprio objeto do MANTO. O resultado visual podia ficar cheio
de estados vermelhos/recortes e deixava de representar o manto R20.4 que tinha
sido aprovado visualmente.

## Correção

R20.5.1 separa três conceitos:

1. **MANTO** — construído exatamente por `_mantle_with_guard`, como R20.4.
2. **VETO FINAL R20.5.1** — objeto separado, calculado a partir de vizinhos medidos.
3. **CONTINUIDADE** — usa uma cópia temporária do guard apenas para impedir que
   uma ilha elevada seja ponte de propagação.

O `mantle.veto_guard` original NÃO é alterado.

## Fluxo

```text
PTD
 -> dense spatial evidence
 -> MANTO R20.4 PRESERVADO
 -> R20.1 original veto
 -> R20.5.1 elevated-surface guard (SEPARADO)
 -> continuity com bloqueio temporário
 -> final elevated-object veto
 -> FINAL GROUND
 -> MDT PREVIEW
 -> EXPORTAR MDT VALIDADO
```

## Consequência esperada

Ao visualizar **MANTO R20.5.1**, a geometria e os estados do manto devem regressar
ao comportamento R20.4. O novo filtro deve aparecer apenas no **FINAL GROUND**.

Isto permite manter a boa continuidade do manto e continuar a tentar rejeitar
vegetação, telhados e outros objetos elevados.

## Regressão automática

`tests/test_r20_5_1_preserve_mantle.py` verifica que:

- o roof/canopy guard original do manto permanece inalterado;
- a continuidade recebe uma cópia combinada;
- o veto novo fica separado;
- um ponto Ground em célula bloqueada é rejeitado no FINAL GROUND;
- um ponto que já era non-ground não é reclassificado.

## Validação de campo

Na `soalheira.las`:

1. abrir R20.4 e observar MANTO;
2. executar R20.5.1;
3. confirmar que MANTO R20.5.1 voltou a ter a continuidade visual do R20.4;
4. abrir FINAL GROUND e verificar se os objetos elevados são rejeitados;
5. criar MDT PREVIEW;
6. validar Elevação, Hillshade e Observação;
7. só depois exportar.

Não considerar R20.5.1 validada apenas porque pytest passa.
