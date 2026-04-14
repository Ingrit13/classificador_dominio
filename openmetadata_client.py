"""
Cliente OpenMetadata para listar domínios e aplicar domínio em tabelas.
Configuração via variáveis de ambiente:
  - OPENMETADATA_URL: host/base URL da API (ex: https://catalogo.cge.mt.gov.br)
  - OPENMETADATA_TOKEN: token JWT/PAT (Bearer) para autenticação
"""
import json
import os
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

# Sem valor padrão para o token (segurança). URL pode ficar vazia para desabilitar.
OPENMETADATA_URL = os.environ.get("OPENMETADATA_URL", "").rstrip("/")
OPENMETADATA_TOKEN = os.environ.get("OPENMETADATA_TOKEN", "")

FALLBACK_DOMAINS = [
    "pessoas", "financeiro", "folha_pagamento", "contratos", "compras",
    "receita", "despesa", "logistica", "corregedoria", "saude", "educacao", "normas", "entidade",
]


def _auth_headers() -> Dict[str, str]:
    if not OPENMETADATA_TOKEN:
        return {}
    return {"Authorization": f"Bearer {OPENMETADATA_TOKEN}"}


def configurado() -> bool:
    """Retorna True se URL e token estiverem definidos."""
    return bool(OPENMETADATA_URL and OPENMETADATA_TOKEN)


def listar_bases_dados(limit: int = 500) -> List[Dict[str, Any]]:
    """
    Lista bases de dados (entidade Database) disponíveis no catálogo.
    Retorna lista de dicts com name, fullyQualifiedName, id.
    """
    if not OPENMETADATA_URL or not OPENMETADATA_TOKEN:
        raise RuntimeError("OPENMETADATA_URL e OPENMETADATA_TOKEN devem estar definidos.")
    url = f"{OPENMETADATA_URL}/api/v1/databases"
    r = requests.get(
        url,
        headers=_auth_headers(),
        params={"limit": limit, "fields": "service"},
        timeout=60,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Falha ao listar databases ({r.status_code}): {r.text[:300]}")
    data = r.json()
    out = []
    for d in data.get("data", []):
        fqn = d.get("fullyQualifiedName")
        if fqn:
            out.append({"name": d.get("name"), "fullyQualifiedName": fqn, "id": d.get("id")})
    return out


def listar_tabelas_por_database(database_fqn: str, limit: int = 1000) -> List[Dict[str, Any]]:
    """
    Lista tabelas cujo parent database corresponde ao FQN informado.
    database_fqn: fullyQualifiedName da base (ex.: postgres.production).
    """
    if not OPENMETADATA_URL or not OPENMETADATA_TOKEN:
        raise RuntimeError("OPENMETADATA_URL e OPENMETADATA_TOKEN devem estar definidos.")
    url = f"{OPENMETADATA_URL}/api/v1/tables"
    r = requests.get(
        url,
        headers=_auth_headers(),
        params={
            "database": database_fqn,
            "limit": limit,
            "fields": "columns,databaseSchema,database",
        },
        timeout=120,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Falha ao listar tabelas ({r.status_code}): {r.text[:400]}")
    return r.json().get("data", [])


def tabelas_openmetadata_para_dataframe(tabelas: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Converte a lista de tabelas retornada pela API em DataFrame para o modelo
    (schema, nome_tabela, qtd_colunas, nome_colunas, table_fqn).
    """
    linhas = []
    for t in tabelas:
        fqn = t.get("fullyQualifiedName") or ""
        nome = t.get("name") or ""
        cols = t.get("columns") or []
        nomes_col = [c.get("name", "") for c in cols if isinstance(c, dict) and c.get("name")]
        qtd = len(nomes_col)
        nome_colunas_str = str(set(nomes_col)) if nomes_col else "{}"
        ds = t.get("databaseSchema")
        if isinstance(ds, dict) and ds.get("name"):
            schema_txt = str(ds.get("name"))
        else:
            partes = fqn.split(".")
            schema_txt = partes[-2] if len(partes) >= 2 else ""
        linhas.append(
            {
                "schema": schema_txt,
                "nome_tabela": nome,
                "qtd_colunas": qtd,
                "nome_colunas": nome_colunas_str,
                "table_fqn": fqn,
            }
        )
    return pd.DataFrame(linhas)


def listar_dominios(limit: int = 1000) -> List[Dict[str, Any]]:
    """
    Lista domínios disponíveis no OpenMetadata.
    Retorna lista de {"name": str, "id": str}. Em falha de API, retorna fallback com id=None.
    """
    if not OPENMETADATA_URL:
        return [{"name": d, "id": None} for d in FALLBACK_DOMAINS]
    url = f"{OPENMETADATA_URL}/api/v1/domains?limit={limit}"
    try:
        r = requests.get(url, headers=_auth_headers(), timeout=30)
        if r.status_code != 200:
            return [{"name": d, "id": None} for d in FALLBACK_DOMAINS]
        data = r.json()
        domains = data.get("data", [])
        return [{"name": d.get("name"), "id": d.get("id")} for d in domains if d.get("name")]
    except Exception:
        return [{"name": d, "id": None} for d in FALLBACK_DOMAINS]


def get_table_by_fqn(table_fqn: str) -> Dict[str, Any]:
    """Busca tabela pelo FQN (fullyQualifiedName). Levanta RuntimeError em falha."""
    if not OPENMETADATA_URL or not OPENMETADATA_TOKEN:
        raise RuntimeError("OPENMETADATA_URL e OPENMETADATA_TOKEN devem estar definidos nas variáveis de ambiente.")
    url = f"{OPENMETADATA_URL}/api/v1/tables/name/{table_fqn}?fields=domain"
    r = requests.get(url, headers=_auth_headers(), timeout=30)
    if r.status_code == 200:
        return r.json()
    raise RuntimeError(f"Falha ao buscar tabela '{table_fqn}' ({r.status_code}): {r.text[:200]}")


def patch_table_domain(table_id: str, domain_id: str) -> Dict[str, Any]:
    """Atribui domínio à tabela via PATCH. Levanta RuntimeError em falha."""
    if not OPENMETADATA_URL or not OPENMETADATA_TOKEN:
        raise RuntimeError("OPENMETADATA_URL e OPENMETADATA_TOKEN devem estar definidos nas variáveis de ambiente.")
    url = f"{OPENMETADATA_URL}/api/v1/tables/{table_id}"
    payload = {"domain": {"id": domain_id, "type": "domain"}}
    headers = {**_auth_headers(), "Content-Type": "application/json"}
    r = requests.patch(url, headers=headers, data=json.dumps(payload), timeout=30)
    if r.status_code in (200, 201):
        return r.json()
    raise RuntimeError(f"PATCH domínio falhou ({r.status_code}): {r.text[:500]}")


def aplicar_dominios(
    itens: List[Dict[str, Any]],
    *,
    col_fqn: str = "table_fqn",
    col_dominio: str = "predicted_domain",
    delay_segundos: float = 0.2,
) -> Dict[str, Any]:
    """
    Para cada item, busca a tabela pelo FQN, resolve o domain_id pelo nome e aplica PATCH.
    itens: lista de dicts com pelo menos col_fqn e col_dominio.
    Retorna: {"total": N, "sucesso": n, "falhas": n, "logs": [...]}
    """
    if not configurado():
        return {"total": 0, "sucesso": 0, "falhas": 0, "logs": ["OPENMETADATA_URL e OPENMETADATA_TOKEN não configurados."]}
    domains = listar_dominios()
    domain_name_to_id = {d["name"]: d["id"] for d in domains if d.get("name") and d.get("id")}
    total, sucesso, falhas = 0, 0, 0
    logs = []
    for row in itens:
        table_fqn = row.get(col_fqn)
        domain_name = row.get(col_dominio)
        if not table_fqn or not domain_name:
            continue
        total += 1
        try:
            table_obj = get_table_by_fqn(table_fqn)
            table_id = table_obj.get("id")
            if not table_id:
                raise RuntimeError("Resposta sem 'id' da tabela.")
            domain_id = domain_name_to_id.get(domain_name)
            if not domain_id:
                domains = listar_dominios()
                domain_name_to_id = {d["name"]: d["id"] for d in domains if d.get("name") and d.get("id")}
                domain_id = domain_name_to_id.get(domain_name)
            if not domain_id:
                raise RuntimeError(f"Domínio '{domain_name}' não encontrado na API.")
            patch_table_domain(table_id, domain_id)
            sucesso += 1
            logs.append(f"OK {table_fqn} <- {domain_name}")
        except Exception as e:
            falhas += 1
            logs.append(f"ERRO {table_fqn}: {e}")
        if delay_segundos > 0:
            time.sleep(delay_segundos)
    return {"total": total, "sucesso": sucesso, "falhas": falhas, "logs": logs}
