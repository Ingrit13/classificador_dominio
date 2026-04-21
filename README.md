# Classificador de domínios (tabelas)

Modelo de classificação automática de domínios de tabelas baseado em **Support Vector Machine (SVM)** (TF-IDF + SVM). Integração opcional com OpenMetadata.

## Layout do projeto (pacote Python)

Código da aplicação em **`src/classificador_dominio/`** (layout *src* recomendado pela [PyPA](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/)):

| Caminho | Conteúdo |
|---------|----------|
| `src/classificador_dominio/` | Pacote importável: `pipeline_core`, `openmetadata_client`, `api`, GUIs |
| `data/` | CSVs de exemplo (`dataset_V9.csv`, etc.) |
| `scripts/` | Scripts executados por caminho (`run_pipeline.py`) |
| `tests/` | Testes (`pytest`) |

Na **raiz** ficam `pyproject.toml`, `README.md`, `.env.example`, artefatos gerados (`modelo_svm_treinado.pkl`, `matriz_confusao.png`, `resultados_api/`, …).

### Instalação

Na raiz do repositório:

```bash
pip install -e .
```

Com API e Jupyter (opcional):

```bash
pip install -e ".[api,jupyter]"
```

## Formas de uso

### 1. GUI principal (Treino / Previsão / Baixar dados)

```bash
python -m classificador_dominio.gui_app
```

- **Treino**: CSV com `schema`, `nome_tabela`, `qtd_colunas`, `nome_colunas`, `dominio`.
- **Previsão**: CSV sem `dominio`; saída com `predicted_domain` e `confidence`.

O processo usa o diretório **raiz do repositório** como cwd (modelo `.pkl` e `.env` na raiz).

### 2. GUI só OpenMetadata

```bash
python -m classificador_dominio.gui_openmetadata
```

### 3. API REST

```bash
pip install -e ".[api]"
uvicorn classificador_dominio.api:app --reload --host 0.0.0.0 --port 8000
```

- **Treinar**: `POST /treinar` (form-data, campo `arquivo`).
- **Prever**: `POST /prever` — resposta CSV com `predicted_domain` e `confidence`.

Documentação: http://localhost:8000/docs

### 4. Pipeline em linha de comando (script)

```bash
py scripts/run_pipeline.py
```

Usa `data/dataset_V9.csv` e grava artefatos na **raiz** do repositório (`modelo_svm_treinado.pkl`, `matriz_confusao.png`, etc.).

### 5. Testes

```bash
pip install -e ".[dev]"
pytest
```

### 6. Executável (Windows, PyInstaller)

Após `pip install -e .`, crie um script de entrada na raiz (por exemplo `launcher_gui.py`) com `from classificador_dominio.gui_app import main` e `main()`, e rode o PyInstaller sobre esse arquivo, usando `--collect-submodules classificador_dominio` se necessário. O `.exe` deve ser distribuído junto com `.env` (ou variáveis de ambiente) e `modelo_svm_treinado.pkl` na pasta de trabalho desejada.

## CSV esperado

- **Treino**: `schema`, `nome_tabela`, `qtd_colunas`, `nome_colunas`, `dominio`.
- **Previsão**: sem `dominio`; saída com `predicted_domain` e `confidence`.

## OpenMetadata

| Variável | Descrição |
|----------|-----------|
| `OPENMETADATA_URL` | URL base da API |
| `OPENMETADATA_TOKEN` | Token JWT ou PAT (Bearer) |

- **API**: `GET /openmetadata/domains`; `POST /openmetadata/aplicar-dominios` (CSV ou JSON `itens`).
- **GUI OpenMetadata**: fluxo catálogo → previsões → aplicar só linhas marcadas na coluna **Enviar**.

Copie `.env.example` para `.env` na raiz do projeto.

## Dependências

Definidas em **`pyproject.toml`**. O arquivo `requirements.txt` contém apenas `-e .` para instalar o projeto localmente.
