# LAS-CAFIISICA

Primeiro marco funcional do classificador de nuvens LAS/LAZ.

Funcionalidades atualmente implementadas:

- aplicação desktop PySide6 executável com `las-cafiisica`;
- abertura de ficheiros `.las` e `.laz` com `laspy`;
- carregamento real de XYZ;
- preservação da classificação de entrada em `original_class`;
- classificação de entrada ignorada como estado de trabalho: `working_class` começa sempre como UNKNOWN (0);
- estatísticas reais da cloud: versão LAS, point format, número de pontos, limites XYZ, amplitude Z, CRS, dimensões, histograma da classificação original, densidade XY aproximada e espaçamento aproximado;
- testes unitários do loader e das estatísticas.

## Instalação de desenvolvimento

```bash
python -m pip install -e ".[dev]"
```

## Executar

```bash
las-cafiisica
```

## Testes

```bash
pytest
```

Ainda não existem motores SMRF, CSF, PTD ou Fusion neste marco.
