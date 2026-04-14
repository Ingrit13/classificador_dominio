"""
API REST para treino, previsão e integração OpenMetadata.
Uso: py -m uvicorn api:app --reload
Variáveis de ambiente para OpenMetadata: OPENMETADATA_URL, OPENMETADATA_TOKEN
"""
import io
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
import pandas as pd

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

from pipeline_core import treinar_com_csv, prever_csv, MODELO_PADRAO, MODELOS_DISPONIVEIS
from openmetadata_client import (
    configurado as om_configurado,
    listar_dominios,
    aplicar_dominios,
)

app = FastAPI(title="Classificador de Domínios", version="1.0")
DIR_RESULTADOS = Path(__file__).resolve().parent / "resultados_api"
DIR_RESULTADOS.mkdir(exist_ok=True)


@app.get("/")
def root():
    return {
        "api": "Classificador de Domínios",
        "endpoints": {
            "treinar": "POST /treinar — envia CSV com colunas: schema, nome_tabela, qtd_colunas, nome_colunas, dominio",
            "prever": "POST /prever — envia CSV (sem dominio); retorna CSV com predicted_domain e confidence",
            "saude": "GET /saude",
            "modelos": "GET /modelos — lista modelos disponíveis para treino (ex.: svm)",
            "openmetadata_dominios": "GET /openmetadata/domains — lista domínios (requer OPENMETADATA_URL e OPENMETADATA_TOKEN)",
            "openmetadata_aplicar": "POST /openmetadata/aplicar-dominios — aplica domínios às tabelas (JSON ou CSV)",
        },
    }


@app.get("/saude")
def saude():
    return {"status": "ok", "modelo_padrao_existe": Path(MODELO_PADRAO).exists()}


@app.get("/modelos")
def listar_modelos():
    """Modelos disponíveis para treino (por enquanto apenas svm)."""
    return {"modelos": MODELOS_DISPONIVEIS}


@app.post("/treinar")
async def treinar(
    arquivo: UploadFile = File(..., description="CSV com coluna 'dominio' para treino"),
    modelo: str = "svm",
):
    """
    modelo: classificador a usar. Por enquanto apenas "svm".
    """
    if not arquivo.filename or not arquivo.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Envie um arquivo .csv")
    modelo = (modelo or "svm").strip().lower()
    if modelo not in MODELOS_DISPONIVEIS:
        raise HTTPException(400, f"Modelo '{modelo}' não disponível. Opções: {MODELOS_DISPONIVEIS}")
    try:
        conteudo = await arquivo.read()
        csv_path = DIR_RESULTADOS / "upload_treino.csv"
        csv_path.write_bytes(conteudo)
        modelo_path = DIR_RESULTADOS / "modelo_treinado.pkl"
        matriz_path = DIR_RESULTADOS / "matriz_confusao.png"
        resultado = treinar_com_csv(
            str(csv_path),
            salvar_modelo_em=str(modelo_path),
            salvar_matriz_em=str(matriz_path),
            modelo=modelo,
        )
        return {
            "mensagem": "Treino concluído.",
            "modelo": modelo,
            "acuracia": resultado["acuracia"],
            "n_treino": resultado["n_treino"],
            "n_teste": resultado["n_teste"],
            "classes": resultado["classes"],
            "relatorio": resultado["relatorio_texto"],
        }
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/prever")
async def prever(
    arquivo: UploadFile = File(..., description="CSV com schema, nome_tabela, qtd_colunas, nome_colunas (sem dominio)"),
):
    if not arquivo.filename or not arquivo.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Envie um arquivo .csv")
    base = Path(__file__).resolve().parent
    modelo_path = base / "resultados_api" / "modelo_treinado.pkl"
    if not modelo_path.exists():
        modelo_path = base / MODELO_PADRAO
    if not modelo_path.exists():
        raise HTTPException(400, "Nenhum modelo encontrado. Treine antes em POST /treinar.")
    try:
        conteudo = await arquivo.read()
        csv_path = DIR_RESULTADOS / "upload_prever.csv"
        csv_path.write_bytes(conteudo)
        saida_path = DIR_RESULTADOS / "previsoes.csv"
        df = prever_csv(str(csv_path), modelo_path=str(modelo_path), salvar_resultado_em=str(saida_path))
        buffer = io.BytesIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        return StreamingResponse(
            buffer,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=previsoes.csv"},
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


# ---------- OpenMetadata (variáveis de ambiente: OPENMETADATA_URL, OPENMETADATA_TOKEN) ----------

@app.get("/openmetadata/domains")
def openmetadata_dominios():
    """Lista domínios do catálogo. Requer OPENMETADATA_URL e OPENMETADATA_TOKEN."""
    if not om_configurado():
        raise HTTPException(
            503,
            "OpenMetadata não configurado. Defina OPENMETADATA_URL e OPENMETADATA_TOKEN nas variáveis de ambiente.",
        )
    return {"dominios": listar_dominios()}


class AplicarDominiosBody(BaseModel):
    itens: List[dict]  # [ {"table_fqn": "...", "predicted_domain": "..." }, ... ]


@app.post("/openmetadata/aplicar-dominios")
async def openmetadata_aplicar(
    arquivo: Optional[UploadFile] = File(None, description="CSV com table_fqn e predicted_domain"),
    body: Optional[AplicarDominiosBody] = Body(None, description="JSON: { \"itens\": [ { \"table_fqn\": \"...\", \"predicted_domain\": \"...\" } ] }"),
):
    """
    Aplica domínios às tabelas no OpenMetadata.
    Envie um CSV (table_fqn, predicted_domain) OU JSON body: { "itens": [ { "table_fqn": "...", "predicted_domain": "..." } ] }.
    """
    if not om_configurado():
        raise HTTPException(
            503,
            "OpenMetadata não configurado. Defina OPENMETADATA_URL e OPENMETADATA_TOKEN nas variáveis de ambiente.",
        )
    lista_itens = []
    if arquivo and arquivo.filename and arquivo.filename.lower().endswith(".csv"):
        content = await arquivo.read()
        df = pd.read_csv(io.BytesIO(content))
        for col in ["table_fqn", "predicted_domain"]:
            if col not in df.columns:
                raise HTTPException(400, f"CSV deve conter coluna '{col}'.")
        lista_itens = df[["table_fqn", "predicted_domain"]].to_dict("records")
    elif body and body.itens:
        lista_itens = body.itens
    else:
        raise HTTPException(400, "Envie um arquivo CSV ou um JSON com 'itens'.")
    if not lista_itens:
        return {"total": 0, "sucesso": 0, "falhas": 0, "logs": []}
    try:
        resultado = aplicar_dominios(lista_itens)
        return resultado
    except Exception as e:
        raise HTTPException(500, str(e))
