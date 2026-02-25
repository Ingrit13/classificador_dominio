# Classificador de domínios (tabelas)

Modelo de classificação automática de domínios de tabelas baseado em **Support Vector Machine (SVM)** (TF-IDF + SVM). Integração opcional com OpenMetadata para tags de domínio.

## Formas de uso (informar CSV)

### 1. Tela pop-up (GUI)

Interface gráfica para escolher o CSV e treinar ou prever:

```bash
py gui_app.py
```

- **Treino**: botão "Selecionar CSV…" → escolha o arquivo com colunas `schema`, `nome_tabela`, `qtd_colunas`, `nome_colunas`, `dominio`. Depois clique em "Treinar modelo".
- **Previsão**: na aba "Previsão", selecione um CSV (sem coluna `dominio`), informe onde salvar o resultado e clique em "Gerar previsões".

### 2. API REST

Servir a API e enviar o CSV por upload:

```bash
pip install fastapi uvicorn python-multipart
py -m uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

- **Treinar**: `POST /treinar` com o CSV no body (form-data, campo `arquivo`). Resposta: acurácia, relatório, etc.
- **Prever**: `POST /prever` com o CSV no body. Resposta: arquivo CSV com colunas `predicted_domain` e `confidence`.

Documentação interativa: http://localhost:8000/docs

### 3. Executável

Gerar um `.exe` da GUI (Windows) com PyInstaller:

```bash
pip install pyinstaller
pyinstaller --onefile --windowed --name ClassificadorDominios gui_app.py
```

O executável ficará em `dist/ClassificadorDominios.exe`. Ao rodar, use "Selecionar CSV…" e depois "Treinar modelo" ou "Gerar previsões".

**Observação**: o primeiro uso pode ser lento (carregar modelo/treino). Para incluir o modelo já treinado no executável, coloque `modelo_svm_treinado.pkl` na mesma pasta do script antes de rodar o `pyinstaller` e use `--add-data "modelo_svm_treinado.pkl;."` (Windows) para embutir.

## CSV esperado

- **Treino**: colunas `schema`, `nome_tabela`, `qtd_colunas`, `nome_colunas`, `dominio`.
- **Previsão**: mesmas colunas, exceto `dominio`. O resultado terá `predicted_domain` e `confidence`.

## Integração OpenMetadata (etapa 3)

Para listar domínios e aplicar domínios às tabelas no catálogo, configure as **variáveis de ambiente**:

| Variável | Descrição |
|----------|-----------|
| `OPENMETADATA_URL` | URL base da API (ex.: `https://catalogo.cge.mt.gov.br`) |
| `OPENMETADATA_TOKEN` | Token JWT ou PAT (Bearer) para autenticação |

Exemplo no Windows (PowerShell):
```powershell
$env:OPENMETADATA_URL = "https://catalogo.cge.mt.gov.br"
$env:OPENMETADATA_TOKEN = "seu_token_jwt_aqui"
```

Exemplo no Linux/macOS:
```bash
export OPENMETADATA_URL="https://catalogo.cge.mt.gov.br"
export OPENMETADATA_TOKEN="seu_token_jwt_aqui"
```

- **API**: `GET /openmetadata/domains` lista domínios; `POST /openmetadata/aplicar-dominios` envia CSV (colunas `table_fqn`, `predicted_domain`) ou JSON `{"itens": [...]}` para aplicar domínios às tabelas.
- **GUI**: aba "OpenMetadata" — selecione o CSV de previsões (com `table_fqn` e `predicted_domain`) e use "Aplicar domínios no OpenMetadata". O CSV gerado na aba Previsão já inclui `table_fqn` quando o CSV de entrada tem `schema` e `nome_tabela`.

Há um `.env.example` no projeto; copie para `.env` e preencha (se usar biblioteca que carrega `.env`).

## Dependências

```bash
pip install -r requirements.txt
```

Para só treino/previsão (sem API): pandas, numpy, scikit-learn, matplotlib, joblib. Para OpenMetadata: requests.
