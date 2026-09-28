# LAS-CAFIISICA

Aplicação desktop Windows em desenvolvimento para leitura e futura reclassificação de nuvens LAS/LAZ.

## Estado funcional atual

- abre LAS e LAZ com `laspy`;
- carrega XYZ real;
- preserva a classificação de entrada em `original_class`;
- inicia `working_class` em UNKNOWN, sem usar a classificação de entrada como decisão;
- mostra estatísticas reais da cloud;
- cria logs locais;
- inclui testes LAS/LAZ;
- inclui self-test Python e self-test do executável;
- inclui perfil autoritativo para Local Build Manager V0.1.8 com SOURCE_GUARD;
- build Windows por PyInstaller em caminho curto `%LOCALAPPDATA%\LBM\LASCAFIISICA`.

Ainda não existem motores SMRF, CSF, PTD ou Fusion.

## Desenvolvimento

```text
python -m pip install -e ".[dev]"
pytest
las-cafiisica
```

## Local Build Manager

Selecionar a pasta/cloned repository `LAS-CAFIISICA` no Local Build Manager V0.1.8.
O gestor descobre `localbuild/las_cafiisica.json` diretamente no repositório.

Pipelines disponíveis:

- `test`: dependências, compileall, pytest e self-test Python;
- `build`: PyInstaller, verificação do EXE, self-test do EXE e ZIP;
- `full`: TEST + BUILD numa única execução.

O SOURCE_GUARD exige:

```text
LAS_CAFIISICA_BUILD_SOURCE_2026_09_R1
```

Saída esperada:

```text
builds/latest/LAS_CAFIISICA_Windows_x64.zip
builds/archive/LAS_CAFIISICA_<TIMESTAMP>.zip
```

O projeto não depende de GitHub Actions para testar ou compilar.

## Runtime local

Para máquinas sem Python 3.12 instalado, usar Local Build Manager V0.1.9 ou superior. O perfil ativa `auto_bootstrap_python` e o gestor prepara um runtime privado em `%LOCALAPPDATA%\\LBM\\py312` antes dos ambientes TEST/BUILD.
