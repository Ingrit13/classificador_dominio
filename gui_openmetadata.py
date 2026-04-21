"""
Janela só com o fluxo OpenMetadata (catálogo, previsão em memória, aplicar domínios).
Requer o mesmo modelo treinado que a GUI completa (MODELO_PADRAO em pipeline_core).

Uso: py gui_openmetadata.py
"""
from __future__ import annotations

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from pathlib import Path
from typing import Any, Dict, List, Sequence

_script_dir = Path(__file__).resolve().parent
os.chdir(_script_dir)

try:
    from dotenv import load_dotenv

    load_dotenv(_script_dir / ".env")
except ImportError:
    pass

try:
    import pandas as pd
    from pipeline_core import prever_dataframe, MODELO_PADRAO
except ImportError:
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror("Erro", "Instale as dependências: pip install pandas numpy scikit-learn")
    sys.exit(1)

try:
    from openmetadata_client import (
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


def _preencher_dominios(tree: ttk.Treeview, dominios: List[Dict[str, Any]]) -> None:
    _tree_clear(tree)
    _tree_configure(tree, ("nome", "id"), ("Nome", "ID"), (220, 320))
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


_MARK_SIM = "Sim"
_MARK_NAO = "Não"


def _preencher_previsoes(tree: ttk.Treeview, df: pd.DataFrame) -> None:
    """Primeira coluna «Enviar»: Sim = incluir no PATCH ao OpenMetadata (padrão em todas as linhas)."""
    _tree_clear(tree)
    _tree_configure(
        tree,
        ("enviar", "nome", "dominio", "confianca"),
        ("Enviar", "Nome", "Domínio", "Confiança (%)"),
        (56, 240, 150, 100),
    )
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
        tree.insert("", tk.END, iid=str(i), values=(_MARK_SIM, nome, dom, conf_txt))


def main() -> None:
    root = tk.Tk()
    root.title("OpenMetadata — Classificador de domínios")
    root.geometry("780x620")
    root.minsize(560, 480)

    frame = ttk.Frame(root, padding=10)
    frame.pack(fill=tk.BOTH, expand=True)

    om_status = (
        "Configurado (OPENMETADATA_URL e OPENMETADATA_TOKEN definidos)"
        if om_configurado()
        else "Não configurado — defina OPENMETADATA_URL e OPENMETADATA_TOKEN nas variáveis de ambiente ou no .env"
    )
    ttk.Label(frame, text=f"Status: {om_status}", font=("", 9)).pack(anchor=tk.W)
    ttk.Label(
        frame,
        text=(
            "Fluxo: 1) Carregar lista do catálogo; 2) escolher uma base; "
            "3) Buscar tabelas e gerar previsões — coluna «Enviar» vem todas como Sim; clique na célula para alternar Sim/Não; "
            "4) Aplicar domínios envia só as linhas com Enviar=Sim. "
            "«Listar domínios» substitui a tabela (para aplicar domínios, busque previsões de novo). "
            "O modelo deve existir (treine com gui_app.py na aba Treino ou copie o .pkl esperado)."
        ),
        wraplength=720,
    ).pack(anchor=tk.W, pady=(0, 4))

    lbl_status = ttk.Label(frame, text="", font=("", 9))
    lbl_status.pack(anchor=tk.W, pady=(0, 4))

    om_state: dict = {"bases": [], "df_pred": None, "modo_tabela": ""}

    def _set_status(msg: str) -> None:
        lbl_status.config(text=msg)

    def _om_label_base(b: dict) -> str:
        svc = b.get("service_pai") or "—"
        return f"{b.get('name')} | {b.get('fullyQualifiedName')}  (serviço: {svc})"

    row_om1 = ttk.Frame(frame)
    row_om1.pack(fill=tk.X, pady=4)
    ttk.Label(row_om1, text="Base (database):").pack(side=tk.LEFT, padx=(0, 8))
    btn_om_carregar_bases = ttk.Button(row_om1, text="Carregar lista do catálogo")
    btn_om_carregar_bases.pack(side=tk.LEFT, padx=(0, 8))
    cb_base_fqn = ttk.Combobox(row_om1, width=48, state="readonly")
    cb_base_fqn.pack(side=tk.LEFT, fill=tk.X, expand=True)

    row_om2 = ttk.Frame(frame)
    row_om2.pack(fill=tk.X, pady=4)
    lbl_om_pred = ttk.Label(row_om2, text="Nenhuma previsão em memória.")
    lbl_om_pred.pack(side=tk.LEFT)
    btn_om_buscar = ttk.Button(row_om2, text="Buscar tabelas e gerar previsões")
    btn_om_buscar.pack(side=tk.RIGHT)

    lf_result = ttk.LabelFrame(frame, text="Resultados", padding=6)
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

    def _on_tree_click_previsoes(event: tk.Event) -> None:
        if om_state.get("modo_tabela") != "previsoes":
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
        vals[0] = _MARK_NAO if vals[0] == _MARK_SIM else _MARK_SIM
        tree_results.item(rowid, values=tuple(vals))

    tree_results.bind("<Button-1>", _on_tree_click_previsoes)

    ttk.Label(frame, text="Log (aplicar domínios e avisos):", font=("", 9)).pack(anchor=tk.W)
    log_om = scrolledtext.ScrolledText(frame, height=7, wrap=tk.WORD, font=("Consolas", 9))
    log_om.pack(fill=tk.BOTH, expand=False, pady=(0, 6))

    def om_carregar_bases() -> None:
        _set_status("")
        if not om_configurado():
            _set_status("Defina OPENMETADATA_URL e OPENMETADATA_TOKEN.")
            messagebox.showwarning("OpenMetadata", "Variáveis de ambiente não configuradas.")
            return
        try:
            bases = listar_bases_dados()
            om_state["bases"] = bases
            labels = [_om_label_base(b) for b in bases]
            cb_base_fqn.configure(state="normal")
            cb_base_fqn.set("")
            cb_base_fqn["values"] = labels
            if labels:
                cb_base_fqn.set(labels[0])
            cb_base_fqn.configure(state="readonly")
            cb_base_fqn.update_idletasks()
            _set_status(f"Carregadas {len(bases)} base(s) (databases).")
            if not bases:
                _set_status(
                    "Nenhuma database encontrada. Verifique ingestão no OpenMetadata e permissões do token."
                )
        except Exception as e:
            _set_status(f"Erro ao listar bases: {e}")
            messagebox.showerror("OpenMetadata", str(e))

    btn_om_carregar_bases.config(command=om_carregar_bases)

    def _om_fqn_base_selecionada() -> str:
        texto = (cb_base_fqn.get() or "").strip()
        if not texto or not om_state["bases"]:
            return ""
        for b in om_state["bases"]:
            if _om_label_base(b) == texto:
                return b.get("fullyQualifiedName") or ""
        idx = cb_base_fqn.current()
        if 0 <= idx < len(om_state["bases"]):
            return om_state["bases"][idx].get("fullyQualifiedName") or ""
        return ""

    def om_buscar_tabelas_e_prever_worker(fqn_db: str) -> None:
        def on_ok(df_pred: pd.DataFrame, n: int) -> None:
            df_pred = df_pred.reset_index(drop=True)
            om_state["df_pred"] = df_pred
            om_state["modo_tabela"] = "previsoes"
            _preencher_previsoes(tree_results, df_pred)
            lf_result.config(text="Tabelas e previsões")
            lbl_om_pred.config(text=f"Previsões em memória: {n} tabela(s).")
            _set_status(
                f"{n} tabela(s); todas com Enviar=Sim. Clique na coluna Enviar para excluir da aplicação no catálogo."
            )

        def on_empty(msg: str) -> None:
            om_state["df_pred"] = None
            om_state["modo_tabela"] = ""
            _tree_clear(tree_results)
            lbl_om_pred.config(text="Nenhuma previsão em memória.")
            _set_status(msg)

        def on_err(e: BaseException) -> None:
            om_state["df_pred"] = None
            om_state["modo_tabela"] = ""
            _tree_clear(tree_results)
            lbl_om_pred.config(text="Nenhuma previsão em memória.")
            _set_status(f"Erro: {e}")
            messagebox.showerror("Erro", str(e))

        try:
            root.after(0, lambda: btn_om_buscar.config(state="disabled"))
            root.after(0, lambda db=fqn_db: _set_status(f"Buscando tabelas da base: {db}…"))
            tabelas = listar_tabelas_por_database(fqn_db)
            if not tabelas:
                root.after(0, lambda: on_empty("Nenhuma tabela retornada pela API para esta base."))
                return
            df_in = tabelas_openmetadata_para_dataframe(tabelas)
            root.after(0, lambda n=len(df_in): _set_status(f"Gerando previsões para {n} tabela(s)…"))
            df_pred = prever_dataframe(df_in, modelo_path=MODELO_PADRAO, salvar_resultado_em=None)
            n = len(df_pred)
            root.after(0, lambda d=df_pred, c=n: on_ok(d, c))
        except Exception as e:
            root.after(0, lambda err=e: on_err(err))
        finally:
            root.after(0, lambda: btn_om_buscar.config(state="normal"))

    def om_buscar_tabelas_e_prever() -> None:
        if not om_configurado():
            _set_status("Defina OPENMETADATA_URL e OPENMETADATA_TOKEN.")
            messagebox.showwarning("OpenMetadata", "Variáveis de ambiente não configuradas.")
            return
        fqn_db = _om_fqn_base_selecionada()
        if not fqn_db:
            messagebox.showwarning(
                "Aviso",
                "Use «Carregar lista do catálogo» acima e selecione uma base no combobox.",
            )
            return
        if not Path(MODELO_PADRAO).exists():
            messagebox.showwarning(
                "Aviso",
                f"Modelo não encontrado em:\n{MODELO_PADRAO}\n\n"
                "Treine com gui_app.py (aba Treino) ou copie o arquivo do modelo para esse caminho.",
            )
            return
        threading.Thread(target=om_buscar_tabelas_e_prever_worker, args=(fqn_db,), daemon=True).start()

    btn_om_buscar.config(command=om_buscar_tabelas_e_prever)

    def om_listar() -> None:
        if not om_configurado():
            _set_status("Defina OPENMETADATA_URL e OPENMETADATA_TOKEN.")
            messagebox.showwarning("OpenMetadata", "Variáveis de ambiente não configuradas.")
            return
        try:
            doms = listar_dominios()
            om_state["modo_tabela"] = "dominios"
            _preencher_dominios(tree_results, doms)
            lf_result.config(text="Domínios")
            _set_status(f"{len(doms)} domínio(s) listados. (Previsões continuam em memória; reaplique a busca para ver a grade com Enviar.)")
        except Exception as e:
            _tree_clear(tree_results)
            _set_status(f"Erro: {e}")
            messagebox.showerror("Erro", str(e))

    def om_aplicar() -> None:
        log_om.delete("1.0", tk.END)
        if not om_configurado():
            log_om.insert(tk.END, "Defina OPENMETADATA_URL e OPENMETADATA_TOKEN nas variáveis de ambiente.\n")
            messagebox.showwarning("OpenMetadata", "Variáveis de ambiente não configuradas.")
            return
        if om_state.get("modo_tabela") != "previsoes":
            messagebox.showwarning(
                "Aviso",
                "A tabela atual não é a de previsões (coluna Enviar). "
                "Execute «Buscar tabelas e gerar previsões» para exibir a grade e escolher o que enviar.",
            )
            return
        df = om_state.get("df_pred")
        if df is None or df.empty:
            messagebox.showwarning("Aviso", "Execute primeiro 'Buscar tabelas e gerar previsões'.")
            return
        try:
            for col in ["table_fqn", "predicted_domain"]:
                if col not in df.columns:
                    messagebox.showerror("Erro", f"Faltando coluna '{col}' nas previsões.")
                    return
            idx_marcados: List[int] = []
            for iid in tree_results.get_children():
                vals = tree_results.item(iid, "values")
                if vals and vals[0] == _MARK_SIM:
                    try:
                        idx_marcados.append(int(iid))
                    except ValueError:
                        continue
            if not idx_marcados:
                messagebox.showwarning("Aviso", "Nenhuma linha com «Enviar» = Sim. Marque ao menos uma tabela.")
                return
            df_sel = df.iloc[sorted(set(idx_marcados))].copy()
            itens = df_sel[["table_fqn", "predicted_domain"]].to_dict("records")
            log_om.insert(tk.END, f"Aplicando {len(itens)} linha(s) selecionada(s) (de {len(df)} no total) no OpenMetadata…\n\n")
            log_om.update()
            resultado = aplicar_dominios(itens)
            log_om.insert(
                tk.END,
                f"Total: {resultado['total']} | Sucesso: {resultado['sucesso']} | Falhas: {resultado['falhas']}\n\n",
            )
            for line in resultado["logs"]:
                log_om.insert(tk.END, line + "\n")
            messagebox.showinfo("Concluído", f"Sucesso: {resultado['sucesso']} | Falhas: {resultado['falhas']}")
        except Exception as e:
            log_om.insert(tk.END, f"Erro: {e}\n")
            messagebox.showerror("Erro", str(e))

    btn_row_om = ttk.Frame(frame)
    btn_row_om.pack(pady=4)
    ttk.Button(btn_row_om, text="Listar domínios", command=om_listar).pack(side=tk.LEFT, padx=(0, 8))
    ttk.Button(btn_row_om, text="Aplicar domínios no OpenMetadata", command=om_aplicar).pack(side=tk.LEFT)

    root.mainloop()


if __name__ == "__main__":
    main()
