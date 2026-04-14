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

# Lê o ambiente em cada uso (permite load_dotenv antes/depois do import sem ficar com URL vazia).
def _om_url() -> str:
    return os.environ.get("OPENMETADATA_URL", "").rstrip("/")


def _om_token() -> str:
    return os.environ.get("OPENMETADATA_TOKEN", "")


def _auth_headers() -> Dict[str, str]:
    t = _om_token()
    if not t:
        return {}
    return {"Authorization": f"Bearer {t}"}


def configurado() -> bool:
    """Retorna True se URL e token estiverem definidos."""
    return bool(_om_url() and _om_token())


def _get_paginated(url: str, params_base: Dict[str, Any], page_limit: int = 500) -> List[Dict[str, Any]]:
    """Acumula todas as páginas (cursor `after`) de um GET JSON com `data` e `paging`."""
    items: List[Dict[str, Any]] = []
    after: Optional[str] = None
    while True:
        params = {**params_base, "limit": min(page_limit, 1000)}
        if after:
            params["after"] = after
        r = requests.get(url, headers=_auth_headers(), params=params, timeout=120)
        if r.status_code != 200:
            raise RuntimeError(f"Falha em {url} ({r.status_code}): {r.text[:400]}")
        data = r.json()
        items.extend(data.get("data") or [])
        after = (data.get("paging") or {}).get("after")
        if not after:
            break
    return items


def _database_dict_from_entity(
    d: Dict[str, Any], *, service_pai_explicito: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Monta o dict usado na GUI a partir de uma entidade Database (listagem REST ou _source do search).
    Garante FQN: a API às vezes omite fullyQualifiedName mas traz service + name.
    """
    fqn = (d.get("fullyQualifiedName") or "").strip()
    if not fqn:
        svc_ref = d.get("service")
        svc_n = ""
        if isinstance(svc_ref, str):
            svc_n = svc_ref.strip()
        elif isinstance(svc_ref, dict):
            svc_n = (svc_ref.get("fullyQualifiedName") or svc_ref.get("name") or "").strip()
        nm = (d.get("name") or "").strip()
        if svc_n and nm:
            fqn = f"{svc_n}.{nm}"
    if not fqn:
        return None
    svc_pai = service_pai_explicito
    if svc_pai is None:
        sr = d.get("service")
        if isinstance(sr, str):
            svc_pai = sr.strip() or None
        elif isinstance(sr, dict):
            svc_pai = (sr.get("fullyQualifiedName") or sr.get("name") or "").strip() or None
    return {
        "name": d.get("name"),
        "fullyQualifiedName": fqn,
        "id": d.get("id"),
        "service_pai": svc_pai,
    }


def _hits_from_search_response(body: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extrai lista de _source (ou documento plano) dos hits da API de search do OpenMetadata."""
    hits_root = body.get("hits")
    inner: List[Any] = []
    if isinstance(hits_root, dict):
        inner = list(hits_root.get("hits") or [])
    elif isinstance(hits_root, list):
        inner = list(hits_root)
    out: List[Dict[str, Any]] = []
    for h in inner:
        if not isinstance(h, dict):
            continue
        src = h.get("_source")
        if isinstance(src, dict):
            out.append(src)
        else:
            out.append(h)
    return out


def listar_databases_via_search(limit: int = 500) -> List[Dict[str, Any]]:
    """
    Lista databases via GET /api/v1/search/query (mesmo índice que a UI costuma usar).
    Útil quando GET /databases?service=... retorna vazio mas o catálogo tem bases indexadas.
    """
    if not _om_url() or not _om_token():
        raise RuntimeError("OPENMETADATA_URL e OPENMETADATA_TOKEN devem estar definidos.")
    url = f"{_om_url()}/api/v1/search/query"
    agregado: List[Dict[str, Any]] = []
    vistos: set = set()
    page_size = min(100, limit)
    for index in ("database", "database_search_index"):
        from_ = 0
        while len(agregado) < limit:
            r = requests.get(
                url,
                headers=_auth_headers(),
                params={
                    "q": "*",
                    "index": index,
                    "from": from_,
                    "size": page_size,
                    "fetch_source": "true",
                },
                timeout=120,
            )
            if r.status_code != 200:
                break
            body = r.json()
            fontes = _hits_from_search_response(body)
            if not fontes:
                break
            for src in fontes:
                ent = dict(src)
                raw_et = ent.get("entityType") or ent.get("entity_type")
                if raw_et is not None and str(raw_et).lower() != "database":
                    continue
                row = _database_dict_from_entity(ent)
                if not row:
                    continue
                fqn = row["fullyQualifiedName"]
                if fqn in vistos:
                    continue
                vistos.add(fqn)
                agregado.append(row)
                if len(agregado) >= limit:
                    break
            hits_root = body.get("hits")
            total_val = 0
            if isinstance(hits_root, dict):
                tot = hits_root.get("total")
                if isinstance(tot, dict):
                    total_val = int(tot.get("value") or 0)
                elif isinstance(tot, (int, float)):
                    total_val = int(tot)
            from_ += page_size
            if from_ >= total_val or len(fontes) < page_size:
                break
        if agregado:
            break
    return agregado


def listar_servicos_database(limit: int = 500) -> List[Dict[str, Any]]:
    """Lista Database Services (conexões de banco) — GET /api/v1/services/databaseServices."""
    if not _om_url() or not _om_token():
        raise RuntimeError("OPENMETADATA_URL e OPENMETADATA_TOKEN devem estar definidos.")
    url = f"{_om_url()}/api/v1/services/databaseServices"
    try:
        raw = _get_paginated(url, {"limit": limit, "include": "all"}, page_limit=limit)
    except RuntimeError:
        raw = _get_paginated(url, {"limit": limit}, page_limit=limit)
    out = []
    for s in raw:
        fqn = s.get("fullyQualifiedName") or s.get("name")
        if fqn:
            out.append(
                {
                    "name": s.get("name"),
                    "fullyQualifiedName": fqn,
                    "id": s.get("id"),
                    "serviceType": s.get("serviceType"),
                }
            )
    return out


def listar_databases_por_servico(service_id: str, limit: int = 500) -> List[Dict[str, Any]]:
    """
    Lista entidades Database ligadas a um Database Service (parâmetro `service`).
    Algumas versões do OpenMetadata rejeitam `include=all`; outras esperam `name` em vez de FQN.
    """
    if not _om_url() or not _om_token():
        raise RuntimeError("OPENMETADATA_URL e OPENMETADATA_TOKEN devem estar definidos.")
    url = f"{_om_url()}/api/v1/databases"
    param_variants = [
        {"service": service_id, "fields": "service", "include": "all", "limit": limit},
        {"service": service_id, "fields": "service", "limit": limit},
        {"service": service_id, "limit": limit},
    ]
    raw: List[Dict[str, Any]] = []
    last_err: Optional[RuntimeError] = None
    for params in param_variants:
        try:
            raw = _get_paginated(url, params, page_limit=limit)
            last_err = None
            break
        except RuntimeError as e:
            last_err = e
    if last_err is not None:
        raise last_err
    out = []
    for d in raw:
        row = _database_dict_from_entity(d if isinstance(d, dict) else {})
        if row:
            out.append(row)
    return out


def listar_bases_dados(limit: int = 500) -> List[Dict[str, Any]]:
    """
    Lista bases de dados (Database) no catálogo.

    Estratégia (OpenMetadata costuma indexar databases por serviço):
    1) Lista cada Database Service em /api/v1/services/databaseServices
    2) Para cada serviço, busca databases com GET /api/v1/databases?service=<nome>
       (a API documenta filtro por *nome* do serviço; tentamos name antes do FQN)
    3) Se ainda vazio, tenta listagem global GET /api/v1/databases
    4) Se ainda vazio, tenta GET /api/v1/search/query (índice database)

    Retorna lista de dicts com name, fullyQualifiedName, id e opcionalmente service_pai.
    """
    if not _om_url() or not _om_token():
        raise RuntimeError("OPENMETADATA_URL e OPENMETADATA_TOKEN devem estar definidos.")

    vistos: set = set()
    out: List[Dict[str, Any]] = []

    servicos = listar_servicos_database(limit=limit)

    for svc in servicos:
        svc_fqn = (svc.get("fullyQualifiedName") or "").strip()
        svc_name = (svc.get("name") or "").strip()
        chaves_servico: List[str] = []
        if svc_name:
            chaves_servico.append(svc_name)
        if svc_fqn and svc_fqn not in chaves_servico:
            chaves_servico.append(svc_fqn)

        dbs: List[Dict[str, Any]] = []
        for chave in chaves_servico:
            try:
                dbs = listar_databases_por_servico(chave, limit=limit)
            except RuntimeError:
                dbs = []
            if dbs:
                break
        for d in dbs:
            row = _database_dict_from_entity(d if isinstance(d, dict) else {}, service_pai_explicito=svc_fqn or None)
            if not row:
                continue
            fqn = row["fullyQualifiedName"]
            if fqn in vistos:
                continue
            vistos.add(fqn)
            out.append(row)

    if not out:
        url = f"{_om_url()}/api/v1/databases"
        try:
            raw = _get_paginated(
                url,
                {"fields": "service", "include": "all", "limit": limit},
                page_limit=limit,
            )
        except RuntimeError:
            raw = _get_paginated(url, {"fields": "service", "limit": limit}, page_limit=limit)
        for d in raw:
            row = _database_dict_from_entity(d if isinstance(d, dict) else {})
            if not row:
                continue
            fqn = row["fullyQualifiedName"]
            if fqn in vistos:
                continue
            vistos.add(fqn)
            out.append(row)

    if not out:
        try:
            for row in listar_databases_via_search(limit=limit):
                fqn = row.get("fullyQualifiedName") or ""
                if not fqn or fqn in vistos:
                    continue
                vistos.add(fqn)
                out.append(row)
        except (RuntimeError, requests.RequestException, ValueError, KeyError, TypeError):
            pass

    return out


def listar_tabelas_por_database(database_fqn: str, limit: int = 1000) -> List[Dict[str, Any]]:
    """
    Lista tabelas cujo parent database corresponde ao FQN informado.
    database_fqn: fullyQualifiedName da base (ex.: postgres.production).
    """
    if not _om_url() or not _om_token():
        raise RuntimeError("OPENMETADATA_URL e OPENMETADATA_TOKEN devem estar definidos.")
    url = f"{_om_url()}/api/v1/tables"
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
    Lista domínios cadastrados no OpenMetadata (GET /api/v1/domains), com paginação quando a API enviar `paging.after`.
    Retorna lista de {"name": str, "id": str}.
    """
    if not _om_url() or not _om_token():
        raise RuntimeError("OPENMETADATA_URL e OPENMETADATA_TOKEN devem estar definidos para listar domínios.")
    url = f"{_om_url()}/api/v1/domains"
    resultado: List[Dict[str, Any]] = []
    after: Optional[str] = None
    while True:
        params: Dict[str, Any] = {"limit": min(limit, 1000)}
        if after:
            params["after"] = after
        r = requests.get(url, headers=_auth_headers(), params=params, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"Falha ao listar domínios ({r.status_code}): {r.text[:400]}")
        data = r.json()
        for d in data.get("data", []):
            nome = d.get("name")
            if nome:
                resultado.append({"name": nome, "id": d.get("id")})
        paging = data.get("paging") or {}
        after = paging.get("after")
        if not after:
            break
    return resultado


def get_table_by_fqn(table_fqn: str) -> Dict[str, Any]:
    """Busca tabela pelo FQN (fullyQualifiedName). Levanta RuntimeError em falha."""
    if not _om_url() or not _om_token():
        raise RuntimeError("OPENMETADATA_URL e OPENMETADATA_TOKEN devem estar definidos nas variáveis de ambiente.")
    url = f"{_om_url()}/api/v1/tables/name/{table_fqn}?fields=domain"
    r = requests.get(url, headers=_auth_headers(), timeout=30)
    if r.status_code == 200:
        return r.json()
    raise RuntimeError(f"Falha ao buscar tabela '{table_fqn}' ({r.status_code}): {r.text[:200]}")


def patch_table_domain(table_id: str, domain_id: str) -> Dict[str, Any]:
    """Atribui domínio à tabela via PATCH. Levanta RuntimeError em falha."""
    if not _om_url() or not _om_token():
        raise RuntimeError("OPENMETADATA_URL e OPENMETADATA_TOKEN devem estar definidos nas variáveis de ambiente.")
    url = f"{_om_url()}/api/v1/tables/{table_id}"
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
