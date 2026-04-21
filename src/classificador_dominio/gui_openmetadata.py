"""
Janela só com o fluxo OpenMetadata (catálogo, previsão em memória, aplicar domínios).
Requer o mesmo modelo treinado que a GUI completa (MODELO_PADRAO em pipeline_core).

Uso (na raiz do repositório)::

    python -m classificador_dominio.gui_openmetadata
"""
from __future__ import annotations

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

from .paths import repo_root

_REPO = repo_root()
os.chdir(_REPO)

try:
    from dotenv import load_dotenv

    load_dotenv(_REPO / ".env")
except ImportError:
    pass

try:
    import pandas as pd
    from .pipeline_core import prever_dataframe, MODELO_PADRAO
except ImportError:
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror("Erro", "Instale as dependências: pip install pandas numpy scikit-learn")
    sys.exit(1)

try:
    from .openmetadata_client import (
        configurado as om_configurado,
        listar_dominios,
        aplicar_dominios,
        listar_bases_dados,
        listar_tabelas_por_database,
        tabelas_openmetadata_para_dataframe,
    )
except ImportError:
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror("Erro", "Módulo openmetadata_client não encontrado.")
    sys.exit(1)


# --- i18n ---
LANG_DISPLAY = ("Português", "English")
LANG_CODE = {"Português": "pt", "English": "en"}
LANG_LABEL = {"pt": "Português", "en": "English"}

I18N: Dict[str, Dict[str, str]] = {
    "pt": {
        "title": "OpenMetadata — Classificador de domínios",
        "language": "Idioma:",
        "status_prefix": "Status:",
        "status_ok": "Configurado (OPENMETADATA_URL e OPENMETADATA_TOKEN definidos)",
        "status_bad": "Não configurado — defina OPENMETADATA_URL e OPENMETADATA_TOKEN nas variáveis de ambiente ou no .env",
        "flow": (
            "Fluxo: 1) Carregar lista do catálogo; 2) escolher uma base; "
            "3) Buscar tabelas e gerar previsões — coluna «Enviar» vem todas como Sim; clique na célula para alternar Sim/Não; "
            "4) Aplicar domínios envia só as linhas com Enviar=Sim. "
            "«Listar domínios» substitui a tabela (para aplicar domínios, busque previsões de novo). "
            "O modelo deve existir (treine com: python -m classificador_dominio.gui_app — aba Treino — ou copie o .pkl)."
        ),
        "base_db": "Base (database):",
        "btn_load": "Carregar lista do catálogo",
        "pred_none": "Nenhuma previsão em memória.",
        "pred_memory": "Previsões em memória: {n} tabela(s).",
        "btn_search": "Buscar tabelas e gerar previsões",
        "frame_results": "Resultados",
        "frame_tables": "Tabelas e previsões",
        "frame_domains": "Domínios",
        "log_caption": "Log (aplicar domínios e avisos):",
        "btn_domains": "Listar domínios",
        "btn_apply": "Aplicar domínios no OpenMetadata",
        "th_name": "Nome",
        "th_id": "ID",
        "th_send": "Enviar",
        "th_table": "Nome",
        "th_domain": "Domínio",
        "th_conf": "Confiança (%)",
        "mark_yes": "Sim",
        "mark_no": "Não",
        "service_word": "serviço",
        "env_hint": "Defina OPENMETADATA_URL e OPENMETADATA_TOKEN.",
        "env_warn": "Variáveis de ambiente não configuradas.",
        "env_log": "Defina OPENMETADATA_URL e OPENMETADATA_TOKEN nas variáveis de ambiente.\n",
        "warn_title": "Aviso",
        "err_title": "Erro",
        "om_title": "OpenMetadata",
        "done_title": "Concluído",
        "warn_select_base": "Use «Carregar lista do catálogo» acima e selecione uma base no combobox.",
        "warn_model": "Modelo não encontrado em:\n{path}\n\nTreine com: python -m classificador_dominio.gui_app (aba Treino) ou copie o .pkl para a raiz do projeto.",
        "warn_not_pred": "A tabela atual não é a de previsões (coluna Enviar). Execute «Buscar tabelas e gerar previsões» para exibir a grade e escolher o que enviar.",
        "warn_run_first": "Execute primeiro «Buscar tabelas e gerar previsões».",
        "warn_no_send": "Nenhuma linha com «Enviar» = Sim. Marque ao menos uma tabela.",
        "err_col": "Faltando coluna '{col}' nas previsões.",
        "done_msg": "Sucesso: {ok} | Falhas: {fail}",
        "st_loaded": "Carregadas {n} base(s) (databases).",
        "st_no_db": "Nenhuma database encontrada. Verifique ingestão no OpenMetadata e permissões do token.",
        "st_err_bases": "Erro ao listar bases: {e}",
        "st_fetch": "Buscando tabelas da base: {db}…",
        "st_gen": "Gerando previsões para {n} tabela(s)…",
        "st_no_tbl": "Nenhuma tabela retornada pela API para esta base.",
        "st_after_pred": "{n} tabela(s); todas com Enviar=Sim. Clique na coluna Enviar para excluir da aplicação no catálogo.",
        "st_domains": "{n} domínio(s) listados. (Previsões continuam em memória; reaplique a busca para ver a grade com Enviar.)",
        "st_err": "Erro: {e}",
        "log_apply": "Aplicando {sel} linha(s) selecionada(s) (de {tot} no total) no OpenMetadata…\n\n",
        "log_tot": "Total: {tot} | Sucesso: {ok} | Falhas: {fail}\n\n",
    },
    "en": {
        "title": "OpenMetadata — Domain classifier",
        "language": "Language:",
        "status_prefix": "Status:",
        "status_ok": "Configured (OPENMETADATA_URL and OPENMETADATA_TOKEN are set)",
        "status_bad": "Not configured — set OPENMETADATA_URL and OPENMETADATA_TOKEN in the environment or in .env",
        "flow": (
            "Flow: 1) Load catalog list; 2) pick a database; "
            "3) Fetch tables and run predictions — the «Send» column starts as Yes; click the cell to toggle Yes/No; "
            "4) Apply domains sends only rows with Send=Yes. "
            "«List domains» replaces the table (run Fetch again to apply domains). "
            "The model file must exist (train with: python -m classificador_dominio.gui_app — Train tab — or copy the .pkl)."
        ),
        "base_db": "Database:",
        "btn_load": "Load catalog list",
        "pred_none": "No predictions in memory.",
        "pred_memory": "Predictions in memory: {n} table(s).",
        "btn_search": "Fetch tables and predict",
        "frame_results": "Results",
        "frame_tables": "Tables and predictions",
        "frame_domains": "Domains",
        "log_caption": "Log (apply domains and notices):",
        "btn_domains": "List domains",
        "btn_apply": "Apply domains to OpenMetadata",
        "th_name": "Name",
        "th_id": "ID",
        "th_send": "Send",
        "th_table": "Name",
        "th_domain": "Domain",
        "th_conf": "Confidence (%)",
        "mark_yes": "Yes",
        "mark_no": "No",
        "service_word": "service",
        "env_hint": "Set OPENMETADATA_URL and OPENMETADATA_TOKEN.",
        "env_warn": "Environment variables are not configured.",
        "env_log": "Set OPENMETADATA_URL and OPENMETADATA_TOKEN in the environment.\n",
        "warn_title": "Warning",
        "err_title": "Error",
        "om_title": "OpenMetadata",
        "done_title": "Done",
        "warn_select_base": "Use «Load catalog list» above and select a database in the combobox.",
        "warn_model": "Model not found at:\n{path}\n\nTrain with: python -m classificador_dominio.gui_app (Train tab) or copy the .pkl to the project root.",
        "warn_not_pred": "The current table is not predictions (Send column). Run «Fetch tables and predict» to show the grid and choose what to send.",
        "warn_run_first": "Run «Fetch tables and predict» first.",
        "warn_no_send": "No row with «Send» = Yes. Mark at least one table.",
        "err_col": "Missing column '{col}' in predictions.",
        "done_msg": "Success: {ok} | Failures: {fail}",
        "st_loaded": "Loaded {n} database(s).",
        "st_no_db": "No databases found. Check OpenMetadata ingestion and token permissions.",
        "st_err_bases": "Error listing databases: {e}",
        "st_fetch": "Fetching tables for database: {db}…",
        "st_gen": "Generating predictions for {n} table(s)…",
        "st_no_tbl": "No tables returned by the API for this database.",
        "st_after_pred": "{n} table(s); all with Send=Yes. Click the Send column to exclude rows from the catalog update.",
        "st_domains": "{n} domain(s) listed. (Predictions stay in memory; run Fetch again to see the Send grid.)",
        "st_err": "Error: {e}",
        "log_apply": "Applying {sel} selected row(s) (out of {tot}) to OpenMetadata…\n\n",
        "log_tot": "Total: {tot} | Success: {ok} | Failures: {fail}\n\n",
    },
}


def _t(lang: str, key: str, **kwargs: Any) -> str:
    s = I18N.get(lang, I18N["pt"]).get(key, key)
    if kwargs:
        return s.format(**kwargs)
    return s


def _tree_clear(tree: ttk.Treeview) -> None:
    tree.delete(*tree.get_children())


def _tree_configure(
    tree: ttk.Treeview,
    columns: Sequence[str],
    headings: Sequence[str],
    widths: Sequence[int],
) -> None:
    tree["show"] = "headings"
    tree["columns"] = tuple(columns)
    for col, head, w in zip(columns, headings, widths):
        tree.heading(col, text=head)
        tree.column(col, width=w, minwidth=48, stretch=True)


def _preencher_dominios(tree: ttk.Treeview, dominios: List[Dict[str, Any]], lang: str) -> None:
    _tree_clear(tree)
    _tree_configure(
        tree,
        ("nome", "id"),
        (_t(lang, "th_name"), _t(lang, "th_id")),
        (220, 320),
    )
    for d in dominios:
        nome = str(d.get("name") or "")
        did = str(d.get("id") or "")
        tree.insert("", tk.END, values=(nome, did))


def _nome_linha_previsao(row: pd.Series) -> str:
    if "nome_tabela" in row.index and pd.notna(row.get("nome_tabela")):
        return str(row["nome_tabela"])
    fqn = str(row.get("table_fqn") or "")
    if fqn and "." in fqn:
        return fqn.rsplit(".", 1)[-1]
    return fqn


def _preencher_previsoes(tree: ttk.Treeview, df: pd.DataFrame, lang: str) -> None:
    _tree_clear(tree)
    _tree_configure(
        tree,
        ("enviar", "nome", "dominio", "confianca"),
        (_t(lang, "th_send"), _t(lang, "th_table"), _t(lang, "th_domain"), _t(lang, "th_conf")),
        (56, 240, 150, 100),
    )
    yes = _t(lang, "mark_yes")
    for i, (_, row) in enumerate(df.iterrows()):
        nome = _nome_linha_previsao(row)
        dom = str(row.get("predicted_domain", "") or "")
        conf = row.get("confidence")
        if conf is None or (isinstance(conf, float) and pd.isna(conf)):
            conf_txt = ""
        else:
            c = float(conf)
            pct = c * 100 if c <= 1.0 else c
            conf_txt = f"{pct:.2f}%"
        tree.insert("", tk.END, iid=str(i), values=(yes, nome, dom, conf_txt))


def _migrate_send_column(tree: ttk.Treeview, old_lang: str, new_lang: str) -> None:
    y_old, n_old = _t(old_lang, "mark_yes"), _t(old_lang, "mark_no")
    y_new, n_new = _t(new_lang, "mark_yes"), _t(new_lang, "mark_no")
    for iid in tree.get_children():
        vals = list(tree.item(iid, "values"))
        if not vals:
            continue
        if vals[0] == y_old:
            vals[0] = y_new
        elif vals[0] == n_old:
            vals[0] = n_new
        tree.item(iid, values=tuple(vals))


def _om_label_base(b: dict, lang: str) -> str:
    svc = b.get("service_pai") or "—"
    sw = _t(lang, "service_word")
    return f"{b.get('name')} | {b.get('fullyQualifiedName')}  ({sw}: {svc})"


def main() -> None:
    root = tk.Tk()
    state: Dict[str, Any] = {
        "lang": "pt",
        "bases": [],
        "df_pred": None,
        "modo_tabela": "",
        "pred_n": None,
    }

    def lang() -> str:
        return str(state.get("lang") or "pt")

    def tr(key: str, **kwargs: Any) -> str:
        return _t(lang(), key, **kwargs)

    frame = ttk.Frame(root, padding=10)
    frame.pack(fill=tk.BOTH, expand=True)

    row_lang = ttk.Frame(frame)
    row_lang.pack(fill=tk.X, pady=(0, 4))
    lbl_lang_pick = ttk.Label(row_lang, text="")
    lbl_lang_pick.pack(side=tk.LEFT)
    cb_lang = ttk.Combobox(row_lang, width=14, state="readonly", values=LANG_DISPLAY)
    cb_lang.pack(side=tk.LEFT, padx=(8, 0))
    cb_lang.set(LANG_LABEL["pt"])

    lbl_om_status = ttk.Label(frame, text="", font=("", 9))
    lbl_om_status.pack(anchor=tk.W)
    lbl_flow = ttk.Label(frame, text="", wraplength=720)
    lbl_flow.pack(anchor=tk.W, pady=(0, 4))

    lbl_status = ttk.Label(frame, text="", font=("", 9))
    lbl_status.pack(anchor=tk.W, pady=(0, 4))

    row_om1 = ttk.Frame(frame)
    row_om1.pack(fill=tk.X, pady=4)
    lbl_base = ttk.Label(row_om1, text="")
    lbl_base.pack(side=tk.LEFT, padx=(0, 8))
    btn_om_carregar_bases = ttk.Button(row_om1, text="")
    btn_om_carregar_bases.pack(side=tk.LEFT, padx=(0, 8))
    cb_base_fqn = ttk.Combobox(row_om1, width=48, state="readonly")
    cb_base_fqn.pack(side=tk.LEFT, fill=tk.X, expand=True)

    row_om2 = ttk.Frame(frame)
    row_om2.pack(fill=tk.X, pady=4)
    lbl_om_pred = ttk.Label(row_om2, text="")
    lbl_om_pred.pack(side=tk.LEFT)
    btn_om_buscar = ttk.Button(row_om2, text="")
    btn_om_buscar.pack(side=tk.RIGHT)

    lf_result = ttk.LabelFrame(frame, text="", padding=6)
    lf_result.pack(fill=tk.BOTH, expand=True, pady=(4, 6))

    tree_frm = ttk.Frame(lf_result)
    tree_frm.pack(fill=tk.BOTH, expand=True)
    tree_results = ttk.Treeview(tree_frm, height=14)
    scroll_y = ttk.Scrollbar(tree_frm, orient=tk.VERTICAL, command=tree_results.yview)
    scroll_x = ttk.Scrollbar(tree_frm, orient=tk.HORIZONTAL, command=tree_results.xview)
    tree_results.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
    tree_results.grid(row=0, column=0, sticky="nsew")
    scroll_y.grid(row=0, column=1, sticky="ns")
    scroll_x.grid(row=1, column=0, sticky="ew")
    tree_frm.rowconfigure(0, weight=1)
    tree_frm.columnconfigure(0, weight=1)

    lbl_log = ttk.Label(frame, text="", font=("", 9))
    lbl_log.pack(anchor=tk.W)
    log_om = scrolledtext.ScrolledText(frame, height=7, wrap=tk.WORD, font=("Consolas", 9))
    log_om.pack(fill=tk.BOTH, expand=False, pady=(0, 6))

    btn_row_om = ttk.Frame(frame)
    btn_row_om.pack(pady=4)
    btn_list_domains = ttk.Button(btn_row_om, text="")
    btn_list_domains.pack(side=tk.LEFT, padx=(0, 8))
    btn_apply = ttk.Button(btn_row_om, text="")
    btn_apply.pack(side=tk.LEFT)

    def _refresh_status_line() -> None:
        body = tr("status_ok") if om_configurado() else tr("status_bad")
        lbl_om_status.config(text=f"{tr('status_prefix')} {body}")

    def _refresh_combobox_bases() -> None:
        bases: List[dict] = state.get("bases") or []
        labels = [_om_label_base(b, lang()) for b in bases]
        cb_base_fqn.configure(state="normal")
        cur = cb_base_fqn.get()
        cb_base_fqn.set("")
        cb_base_fqn["values"] = labels
        if labels:
            if cur in labels:
                cb_base_fqn.set(cur)
            else:
                cb_base_fqn.set(labels[0])
        cb_base_fqn.configure(state="readonly")
        cb_base_fqn.update_idletasks()

    def apply_language(previous_lang_code: str | None = None) -> None:
        lg = lang()
        modo = state.get("modo_tabela") or ""
        if (
            previous_lang_code is not None
            and previous_lang_code != lg
            and modo == "previsoes"
            and state.get("df_pred") is not None
            and not state["df_pred"].empty
        ):
            _migrate_send_column(tree_results, previous_lang_code, lg)

        root.title(tr("title"))
        lbl_lang_pick.config(text=tr("language"))
        lbl_flow.config(text=tr("flow"))
        lbl_base.config(text=tr("base_db"))
        btn_om_carregar_bases.config(text=tr("btn_load"))
        btn_om_buscar.config(text=tr("btn_search"))
        lbl_log.config(text=tr("log_caption"))
        btn_list_domains.config(text=tr("btn_domains"))
        btn_apply.config(text=tr("btn_apply"))
        _refresh_status_line()

        if modo == "previsoes":
            lf_result.config(text=tr("frame_tables"))
            df = state.get("df_pred")
            if df is not None and not df.empty:
                _tree_configure(
                    tree_results,
                    ("enviar", "nome", "dominio", "confianca"),
                    (tr("th_send"), tr("th_table"), tr("th_domain"), tr("th_conf")),
                    (56, 240, 150, 100),
                )
                n = state.get("pred_n")
                if n is not None:
                    lbl_om_pred.config(text=tr("pred_memory", n=n))
            else:
                lf_result.config(text=tr("frame_results"))
                lbl_om_pred.config(text=tr("pred_none"))
        elif modo == "dominios":
            lf_result.config(text=tr("frame_domains"))
            _tree_configure(
                tree_results,
                ("nome", "id"),
                (tr("th_name"), tr("th_id")),
                (220, 320),
            )
        else:
            lf_result.config(text=tr("frame_results"))
            lbl_om_pred.config(text=tr("pred_none"))

        _refresh_combobox_bases()

    def _set_status(msg: str) -> None:
        lbl_status.config(text=msg)

    def _on_lang_selected(_event: tk.Event | None = None) -> None:
        disp = cb_lang.get()
        new_code = LANG_CODE.get(disp, "pt")
        old_code = state.get("lang", "pt")
        if new_code == old_code:
            return
        state["lang"] = new_code
        apply_language(previous_lang_code=old_code)

    cb_lang.bind("<<ComboboxSelected>>", _on_lang_selected)

    def _on_tree_click_previsoes(event: tk.Event) -> None:
        if state.get("modo_tabela") != "previsoes":
            return
        if tree_results.identify_region(event.x, event.y) != "cell":
            return
        if tree_results.identify_column(event.x) != "#1":
            return
        rowid = tree_results.identify_row(event.y)
        if not rowid:
            return
        vals = list(tree_results.item(rowid, "values"))
        if len(vals) < 4:
            return
        yes, no = tr("mark_yes"), tr("mark_no")
        vals[0] = no if vals[0] == yes else yes
        tree_results.item(rowid, values=tuple(vals))

    tree_results.bind("<Button-1>", _on_tree_click_previsoes)

    def om_carregar_bases() -> None:
        _set_status("")
        if not om_configurado():
            _set_status(tr("env_hint"))
            messagebox.showwarning(tr("om_title"), tr("env_warn"))
            return
        try:
            bases = listar_bases_dados()
            state["bases"] = bases
            _refresh_combobox_bases()
            _set_status(tr("st_loaded", n=len(bases)))
            if not bases:
                _set_status(tr("st_no_db"))
        except Exception as e:
            _set_status(tr("st_err_bases", e=e))
            messagebox.showerror(tr("om_title"), str(e))

    btn_om_carregar_bases.config(command=om_carregar_bases)

    def _om_fqn_base_selecionada() -> str:
        texto = (cb_base_fqn.get() or "").strip()
        bases: List[dict] = state.get("bases") or []
        if not texto or not bases:
            return ""
        for b in bases:
            if _om_label_base(b, lang()) == texto:
                return b.get("fullyQualifiedName") or ""
        idx = cb_base_fqn.current()
        if 0 <= idx < len(bases):
            return bases[idx].get("fullyQualifiedName") or ""
        return ""

    def om_buscar_tabelas_e_prever_worker(fqn_db: str) -> None:
        lg = lang()

        def on_ok(df_pred: pd.DataFrame, n: int) -> None:
            df_pred = df_pred.reset_index(drop=True)
            state["df_pred"] = df_pred
            state["modo_tabela"] = "previsoes"
            state["pred_n"] = n
            _preencher_previsoes(tree_results, df_pred, lang())
            lf_result.config(text=tr("frame_tables"))
            lbl_om_pred.config(text=tr("pred_memory", n=n))
            _set_status(tr("st_after_pred", n=n))

        def on_empty(msg: str) -> None:
            state["df_pred"] = None
            state["modo_tabela"] = ""
            state["pred_n"] = None
            _tree_clear(tree_results)
            lbl_om_pred.config(text=tr("pred_none"))
            _set_status(msg)

        def on_err(e: BaseException) -> None:
            state["df_pred"] = None
            state["modo_tabela"] = ""
            state["pred_n"] = None
            _tree_clear(tree_results)
            lbl_om_pred.config(text=tr("pred_none"))
            _set_status(tr("st_err", e=e))
            messagebox.showerror(tr("err_title"), str(e))

        try:
            root.after(0, lambda: btn_om_buscar.config(state="disabled"))
            root.after(0, lambda db=fqn_db: _set_status(tr("st_fetch", db=db)))
            tabelas = listar_tabelas_por_database(fqn_db)
            if not tabelas:
                root.after(0, lambda: on_empty(tr("st_no_tbl")))
                return
            df_in = tabelas_openmetadata_para_dataframe(tabelas)
            root.after(0, lambda n=len(df_in): _set_status(tr("st_gen", n=n)))
            df_pred = prever_dataframe(df_in, modelo_path=MODELO_PADRAO, salvar_resultado_em=None)
            n = len(df_pred)
            root.after(0, lambda d=df_pred, c=n: on_ok(d, c))
        except Exception as e:
            root.after(0, lambda err=e: on_err(err))
        finally:
            root.after(0, lambda: btn_om_buscar.config(state="normal"))

    def om_buscar_tabelas_e_prever() -> None:
        if not om_configurado():
            _set_status(tr("env_hint"))
            messagebox.showwarning(tr("om_title"), tr("env_warn"))
            return
        fqn_db = _om_fqn_base_selecionada()
        if not fqn_db:
            messagebox.showwarning(tr("warn_title"), tr("warn_select_base"))
            return
        if not Path(MODELO_PADRAO).exists():
            messagebox.showwarning(tr("warn_title"), tr("warn_model", path=MODELO_PADRAO))
            return
        threading.Thread(target=om_buscar_tabelas_e_prever_worker, args=(fqn_db,), daemon=True).start()

    btn_om_buscar.config(command=om_buscar_tabelas_e_prever)

    def om_listar() -> None:
        if not om_configurado():
            _set_status(tr("env_hint"))
            messagebox.showwarning(tr("om_title"), tr("env_warn"))
            return
        try:
            doms = listar_dominios()
            state["modo_tabela"] = "dominios"
            _preencher_dominios(tree_results, doms, lang())
            lf_result.config(text=tr("frame_domains"))
            _set_status(tr("st_domains", n=len(doms)))
        except Exception as e:
            _tree_clear(tree_results)
            _set_status(tr("st_err", e=e))
            messagebox.showerror(tr("err_title"), str(e))

    def om_aplicar() -> None:
        log_om.delete("1.0", tk.END)
        if not om_configurado():
            log_om.insert(tk.END, tr("env_log"))
            messagebox.showwarning(tr("om_title"), tr("env_warn"))
            return
        if state.get("modo_tabela") != "previsoes":
            messagebox.showwarning(tr("warn_title"), tr("warn_not_pred"))
            return
        df = state.get("df_pred")
        if df is None or df.empty:
            messagebox.showwarning(tr("warn_title"), tr("warn_run_first"))
            return
        try:
            for col in ["table_fqn", "predicted_domain"]:
                if col not in df.columns:
                    messagebox.showerror(tr("err_title"), tr("err_col", col=col))
                    return
            yes = tr("mark_yes")
            idx_marcados: List[int] = []
            for iid in tree_results.get_children():
                vals = tree_results.item(iid, "values")
                if vals and vals[0] == yes:
                    try:
                        idx_marcados.append(int(iid))
                    except ValueError:
                        continue
            if not idx_marcados:
                messagebox.showwarning(tr("warn_title"), tr("warn_no_send"))
                return
            df_sel = df.iloc[sorted(set(idx_marcados))].copy()
            itens = df_sel[["table_fqn", "predicted_domain"]].to_dict("records")
            log_om.insert(tk.END, tr("log_apply", sel=len(itens), tot=len(df)))
            log_om.update()
            resultado = aplicar_dominios(itens)
            log_om.insert(
                tk.END,
                tr("log_tot", tot=resultado["total"], ok=resultado["sucesso"], fail=resultado["falhas"]),
            )
            for line in resultado["logs"]:
                log_om.insert(tk.END, line + "\n")
            messagebox.showinfo(
                tr("done_title"),
                tr("done_msg", ok=resultado["sucesso"], fail=resultado["falhas"]),
            )
        except Exception as e:
            log_om.insert(tk.END, tr("st_err", e=e) + "\n")
            messagebox.showerror(tr("err_title"), str(e))

    btn_list_domains.config(command=om_listar)
    btn_apply.config(command=om_aplicar)

    root.geometry("780x620")
    root.minsize(560, 480)
    apply_language()

    root.mainloop()


if __name__ == "__main__":
    main()
