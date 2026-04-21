"""
Interface gráfica para treino e previsão do classificador de domínios.
O usuário escolhe o arquivo CSV (para treino ou para previsão) e executa o fluxo.

Uso (na raiz do repositório)::

    python -m classificador_dominio.gui_app
"""
from __future__ import annotations

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
from typing import Any, Dict

from .paths import repo_root

_REPO = repo_root()
os.chdir(_REPO)

try:
    from dotenv import load_dotenv

    load_dotenv(_REPO / ".env")
except ImportError:
    pass

try:
    from .pipeline_core import treinar_com_csv, prever_csv, MODELO_PADRAO, MODELOS_DISPONIVEIS
    import pandas as pd
except ImportError:
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror("Erro / Error", "Instale as dependências: pip install pandas numpy scikit-learn")
    sys.exit(1)
try:
    from .openmetadata_client import configurado as om_configurado, _om_url, _auth_headers
except ImportError:
    om_configurado = lambda: False
    _om_url = lambda: ""
    _auth_headers = lambda: {}

# --- i18n ---
LANG_DISPLAY = ("Português", "English")
LANG_CODE = {"Português": "pt", "English": "en"}
LANG_LABEL = {"pt": "Português", "en": "English"}

I18N: Dict[str, Dict[str, str]] = {
    "pt": {
        "title": "Classificador de Domínios — CSV",
        "language": "Idioma:",
        "tab_download": "Baixar Dados",
        "tab_train": "Treino",
        "tab_predict": "Previsão",
        "baixar_desc": "Baixar tabelas do catálogo OpenMetadata e salvar como CSV.",
        "rb_all_tables": "Baixar todos os dados",
        "rb_domain_only": "Baixar somente dados com domínio (catálogo preenchido)",
        "save_csv_to": "Salvar CSV em:",
        "choose_btn": "Escolher…",
        "btn_download": "Baixar dados",
        "treino_model_label": "Modelo:",
        "treino_csv_hint": (
            "CSV para treino (colunas: schema, nome_tabela, qtd_colunas, nome_colunas, dominio):"
        ),
        "select_csv_train": "Selecionar CSV…",
        "btn_train": "Treinar modelo",
        "prever_csv_hint": "CSV para previsão (sem coluna dominio):",
        "select_csv_predict": "Selecionar CSV…",
        "save_result_to": "Salvar resultado em:",
        "btn_predict": "Gerar previsões",
        "dlg_save_csv_title": "Onde salvar o CSV",
        "dlg_open_train_title": "Selecione o CSV de treino",
        "dlg_open_predict_title": "Selecione o CSV para previsão",
        "dlg_save_predict_title": "Onde salvar as previsões",
        "warn_header": "Aviso",
        "warn_om_title": "OpenMetadata",
        "warn_om_msg": "Defina OPENMETADATA_URL e OPENMETADATA_TOKEN nas variáveis de ambiente.",
        "warn_save_path": "Informe onde salvar o arquivo CSV.",
        "warn_select_csv": "Selecione um arquivo CSV válido.",
        "warn_invalid_model": "Modelo inválido. Opções: {opts}",
        "warn_predict_csv": "Selecione um arquivo CSV válido para previsão.",
        "warn_predict_output": "Informe onde salvar o resultado.",
        "log_fetching_catalog": "Buscando tabelas no catálogo…\n",
        "log_tables_domain_progress": "  {n} tabelas com domínio obtidas até agora…\n",
        "log_tables_progress": "  {n} tabelas obtidas até agora…\n",
        "log_total_obtained": "Total obtido: {n} tabelas.\n",
        "log_exported_n": "\n{n} tabelas exportadas → {path}\n",
        "msg_success_export_title": "Sucesso",
        "msg_success_export_body": "{n} tabelas exportadas para:\n{path}",
        "msg_err_download_title": "Erro ao baixar dados",
        "log_error": "\nErro: {e}\n",
        "log_training_start": "Carregando e treinando com: {path}\nModelo: {model}\n\n",
        "log_accuracy": "Acurácia: {pct:.2f}%\n",
        "log_train_test": "Treino: {n_train} | Teste: {n_test}\n",
        "log_classes": "Classes: {classes}\n\n",
        "log_model_saved": "\n\nModelo salvo em: {path}\n",
        "msg_train_done_title": "Sucesso",
        "msg_train_done_body": "Treino concluído. Modelo salvo.",
        "msg_train_err_title": "Erro no treino",
        "log_predict_start": "Previsão: {path}\n",
        "err_no_model": "Nenhum modelo encontrado. Treine antes (aba Treino).",
        "log_total_lines": "Total de linhas: {n}\n",
        "log_result_saved": "Resultado salvo em: {path}\n",
        "msg_predict_done_title": "Sucesso",
        "msg_predict_done_body": "Previsões salvas em:\n{path}",
        "msg_predict_err_title": "Erro na previsão",
    },
    "en": {
        "title": "Domain Classifier — CSV",
        "language": "Language:",
        "tab_download": "Download Data",
        "tab_train": "Training",
        "tab_predict": "Prediction",
        "baixar_desc": "Download tables from the OpenMetadata catalog and save as CSV.",
        "rb_all_tables": "Download all data",
        "rb_domain_only": "Download only rows with a domain (filled catalog)",
        "save_csv_to": "Save CSV to:",
        "choose_btn": "Choose…",
        "btn_download": "Download data",
        "treino_model_label": "Model:",
        "treino_csv_hint": (
            "Training CSV (columns: schema, nome_tabela, qtd_colunas, nome_colunas, dominio):"
        ),
        "select_csv_train": "Select CSV…",
        "btn_train": "Train model",
        "prever_csv_hint": "Prediction CSV (without dominio column):",
        "select_csv_predict": "Select CSV…",
        "save_result_to": "Save result to:",
        "btn_predict": "Run predictions",
        "dlg_save_csv_title": "Where to save the CSV",
        "dlg_open_train_title": "Select the training CSV",
        "dlg_open_predict_title": "Select the prediction CSV",
        "dlg_save_predict_title": "Where to save predictions",
        "warn_header": "Warning",
        "warn_om_title": "OpenMetadata",
        "warn_om_msg": "Set OPENMETADATA_URL and OPENMETADATA_TOKEN in the environment.",
        "warn_save_path": "Choose where to save the CSV file.",
        "warn_select_csv": "Select a valid CSV file.",
        "warn_invalid_model": "Invalid model. Options: {opts}",
        "warn_predict_csv": "Select a valid CSV file for prediction.",
        "warn_predict_output": "Choose where to save the output.",
        "log_fetching_catalog": "Fetching tables from the catalog…\n",
        "log_tables_domain_progress": "  {n} tables with domain fetched so far…\n",
        "log_tables_progress": "  {n} tables fetched so far…\n",
        "log_total_obtained": "Total fetched: {n} tables.\n",
        "log_exported_n": "\n{n} tables exported → {path}\n",
        "msg_success_export_title": "Success",
        "msg_success_export_body": "{n} tables exported to:\n{path}",
        "msg_err_download_title": "Download error",
        "log_error": "\nError: {e}\n",
        "log_training_start": "Loading and training with: {path}\nModel: {model}\n\n",
        "log_accuracy": "Accuracy: {pct:.2f}%\n",
        "log_train_test": "Train: {n_train} | Test: {n_test}\n",
        "log_classes": "Classes: {classes}\n\n",
        "log_model_saved": "\n\nModel saved to: {path}\n",
        "msg_train_done_title": "Success",
        "msg_train_done_body": "Training finished. Model saved.",
        "msg_train_err_title": "Training error",
        "log_predict_start": "Prediction: {path}\n",
        "err_no_model": "No model found. Train first (Training tab).",
        "log_total_lines": "Total rows: {n}\n",
        "log_result_saved": "Result saved to: {path}\n",
        "msg_predict_done_title": "Success",
        "msg_predict_done_body": "Predictions saved to:\n{path}",
        "msg_predict_err_title": "Prediction error",
    },
}


def _t(lang: str, key: str, **kwargs: Any) -> str:
    s = I18N.get(lang, I18N["pt"]).get(key, key)
    if kwargs:
        return s.format(**kwargs)
    return s


def baixar_dados_worker(
    saida_path: str,
    somente_com_dominio: bool,
    log_widget: scrolledtext.ScrolledText,
    btn: ttk.Button,
    lang_code: str,
):
    tr = lambda k, **kw: _t(lang_code, k, **kw)
    try:
        import requests as _req

        btn.config(state="disabled")
        log_widget.delete("1.0", tk.END)
        base_url = f"{_om_url()}/api/v1"
        headers = {**_auth_headers(), "accept": "application/json"}
        _req.packages.urllib3.disable_warnings()
        log_widget.insert(tk.END, tr("log_fetching_catalog"))
        log_widget.update()

        def montar_linha(t):
            fqn = t.get("fullyQualifiedName", "")
            partes = fqn.split(".") if fqn else []
            colunas = [c["name"] for c in t.get("columns", []) if c.get("name")]
            return {
                "schema": partes[0] if partes else "",
                "nome_tabela": partes[-1] if partes else "",
                "qtd_colunas": len(colunas),
                "nome_colunas": "{" + ",".join(f"'{c}'" for c in colunas) + "}",
                "dominio": (t.get("domains") or [{}])[0].get("displayName", ""),
                "texto": t.get("description") or "",
            }

        rows = []
        if somente_com_dominio:
            offset = 0
            size = 100
            while True:
                resp = _req.get(
                    f"{base_url}/search/query",
                    headers=headers,
                    params={"q": "domains:*", "index": "table_search_index", "from": offset, "size": size},
                    verify=False,
                    timeout=120,
                )
                resp.raise_for_status()
                hits = resp.json().get("hits", {}).get("hits", [])
                if not hits:
                    break
                for hit in hits:
                    rows.append(montar_linha(hit["_source"]))
                log_widget.insert(tk.END, tr("log_tables_domain_progress", n=len(rows)))
                log_widget.update()
                offset += size
        else:
            after = None
            while True:
                params = {"limit": 100, "fields": "columns,tags,domains"}
                if after:
                    params["after"] = after
                resp = _req.get(
                    f"{base_url}/tables",
                    headers=headers,
                    params=params,
                    verify=False,
                    timeout=120,
                )
                resp.raise_for_status()
                payload = resp.json()
                for t in payload["data"]:
                    rows.append(montar_linha(t))
                after = payload.get("paging", {}).get("after")
                log_widget.insert(tk.END, tr("log_tables_progress", n=len(rows)))
                log_widget.update()
                if not after:
                    break

        log_widget.insert(tk.END, tr("log_total_obtained", n=len(rows)))
        log_widget.update()
        df = pd.DataFrame(rows)
        df.to_csv(saida_path, index=False, encoding="utf-8-sig")
        log_widget.insert(tk.END, tr("log_exported_n", n=len(df), path=saida_path))
        messagebox.showinfo(
            tr("msg_success_export_title"),
            tr("msg_success_export_body", n=len(df), path=saida_path),
        )
    except Exception as e:
        log_widget.insert(tk.END, tr("log_error", e=e))
        messagebox.showerror(tr("msg_err_download_title"), str(e))
    finally:
        btn.config(state="normal")


def treinar_worker(csv_path: str, modelo: str, log_widget: scrolledtext.ScrolledText, btn: ttk.Button, lang_code: str):
    tr = lambda k, **kw: _t(lang_code, k, **kw)
    try:
        btn.config(state="disabled")
        log_widget.delete("1.0", tk.END)
        log_widget.insert(tk.END, tr("log_training_start", path=csv_path, model=modelo))
        log_widget.update()
        resultado = treinar_com_csv(
            csv_path,
            salvar_modelo_em=MODELO_PADRAO,
            salvar_matriz_em="matriz_confusao.png",
            modelo=modelo,
        )
        log_widget.insert(tk.END, tr("log_accuracy", pct=resultado["acuracia"] * 100))
        log_widget.insert(tk.END, tr("log_train_test", n_train=resultado["n_treino"], n_test=resultado["n_teste"]))
        log_widget.insert(tk.END, tr("log_classes", classes=resultado["classes"]))
        log_widget.insert(tk.END, resultado["relatorio_texto"])
        log_widget.insert(tk.END, tr("log_model_saved", path=MODELO_PADRAO))
        messagebox.showinfo(tr("msg_train_done_title"), tr("msg_train_done_body"))
    except Exception as e:
        log_widget.insert(tk.END, tr("log_error", e=e))
        messagebox.showerror(tr("msg_train_err_title"), str(e))
    finally:
        btn.config(state="normal")


def prever_worker(csv_path: str, saida_path: str, log_widget: scrolledtext.ScrolledText, btn: ttk.Button, lang_code: str):
    tr = lambda k, **kw: _t(lang_code, k, **kw)
    try:
        btn.config(state="disabled")
        log_widget.delete("1.0", tk.END)
        log_widget.insert(tk.END, tr("log_predict_start", path=csv_path))
        if not Path(MODELO_PADRAO).exists():
            raise FileNotFoundError(tr("err_no_model"))
        df = prever_csv(csv_path, modelo_path=MODELO_PADRAO, salvar_resultado_em=saida_path)
        log_widget.insert(tk.END, tr("log_total_lines", n=len(df)))
        log_widget.insert(tk.END, tr("log_result_saved", path=saida_path))
        cols = [c for c in ["predicted_domain", "confidence", "table_fqn"] if c in df.columns]
        log_widget.insert(tk.END, df[cols].head(10).to_string() if cols else df.head(10).to_string())
        messagebox.showinfo(tr("msg_predict_done_title"), tr("msg_predict_done_body", path=saida_path))
    except Exception as e:
        log_widget.insert(tk.END, tr("log_error", e=e))
        messagebox.showerror(tr("msg_predict_err_title"), str(e))
    finally:
        btn.config(state="normal")


def main():
    root = tk.Tk()
    state: Dict[str, Any] = {"lang": "pt"}

    def lang() -> str:
        return str(state.get("lang") or "pt")

    def tr(key: str, **kwargs: Any) -> str:
        return _t(lang(), key, **kwargs)

    root.geometry("640x520")
    root.minsize(500, 400)

    row_lang = ttk.Frame(root, padding=(8, 8, 8, 0))
    row_lang.pack(fill=tk.X)
    lbl_lang_pick = ttk.Label(row_lang, text="")
    lbl_lang_pick.pack(side=tk.LEFT)
    cb_lang = ttk.Combobox(row_lang, width=14, state="readonly", values=LANG_DISPLAY)
    cb_lang.pack(side=tk.LEFT, padx=(8, 0))
    cb_lang.set(LANG_LABEL["pt"])

    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

    # ---- Aba Baixar Dados ----
    frame_baixar = ttk.Frame(notebook, padding=10)
    notebook.add(frame_baixar, text="")

    lbl_baixar_desc = ttk.Label(frame_baixar, text="", font=("", 9))
    lbl_baixar_desc.pack(anchor=tk.W, pady=(0, 6))

    opcao_download = tk.StringVar(value="todos")
    rb_todos = ttk.Radiobutton(frame_baixar, text="", variable=opcao_download, value="todos")
    rb_todos.pack(anchor=tk.W)
    rb_com_dom = ttk.Radiobutton(frame_baixar, text="", variable=opcao_download, value="com_dominio")
    rb_com_dom.pack(anchor=tk.W, pady=(0, 8))

    lbl_save_csv = ttk.Label(frame_baixar, text="")
    lbl_save_csv.pack(anchor=tk.W)
    path_baixar_saida = tk.StringVar(value=str(_REPO / "tabelas_openmetadata.csv"))
    row_baixar = ttk.Frame(frame_baixar)
    row_baixar.pack(fill=tk.X, pady=4)
    entry_baixar = ttk.Entry(row_baixar, textvariable=path_baixar_saida, width=60)
    entry_baixar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))

    def escolher_saida_baixar():
        p = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            title=tr("dlg_save_csv_title"),
        )
        if p:
            path_baixar_saida.set(p)

    btn_escolher_baixar = ttk.Button(row_baixar, text="", command=escolher_saida_baixar)
    btn_escolher_baixar.pack(side=tk.RIGHT)

    log_baixar = scrolledtext.ScrolledText(frame_baixar, height=14, wrap=tk.WORD, font=("Consolas", 9))
    log_baixar.pack(fill=tk.BOTH, expand=True, pady=8)

    def iniciar_download():
        if not om_configurado():
            messagebox.showwarning(tr("warn_om_title"), tr("warn_om_msg"))
            return
        saida = path_baixar_saida.get().strip()
        if not saida:
            messagebox.showwarning(tr("warn_header"), tr("warn_save_path"))
            return
        somente_com_dominio = opcao_download.get() == "com_dominio"
        threading.Thread(
            target=baixar_dados_worker,
            args=(saida, somente_com_dominio, log_baixar, btn_baixar, lang()),
            daemon=True,
        ).start()

    btn_baixar = ttk.Button(frame_baixar, text="", command=iniciar_download)
    btn_baixar.pack(pady=4)

    # ---- Aba Treino ----
    frame_treino = ttk.Frame(notebook, padding=10)
    notebook.add(frame_treino, text="")

    lbl_modelo = ttk.Label(frame_treino, text="")
    lbl_modelo.pack(anchor=tk.W)
    modelo_treino = tk.StringVar(value=MODELOS_DISPONIVEIS[0] if MODELOS_DISPONIVEIS else "svm")
    row_modelo = ttk.Frame(frame_treino)
    row_modelo.pack(fill=tk.X, pady=2)
    cb_modelo = ttk.Combobox(row_modelo, textvariable=modelo_treino, values=MODELOS_DISPONIVEIS, state="readonly", width=15)
    cb_modelo.pack(side=tk.LEFT)
    if MODELOS_DISPONIVEIS and modelo_treino.get() not in MODELOS_DISPONIVEIS:
        modelo_treino.set(MODELOS_DISPONIVEIS[0])

    lbl_treino_csv = ttk.Label(frame_treino, text="")
    lbl_treino_csv.pack(anchor=tk.W)
    path_treino = tk.StringVar()
    row1 = ttk.Frame(frame_treino)
    row1.pack(fill=tk.X, pady=4)
    entry_treino = ttk.Entry(row1, textvariable=path_treino, width=60)
    entry_treino.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))

    def escolher_treino():
        p = filedialog.askopenfilename(
            filetypes=[("CSV", "*.csv"), ("Todos", "*.*")],
            title=tr("dlg_open_train_title"),
        )
        if p:
            path_treino.set(p)

    btn_sel_treino = ttk.Button(row1, text="", command=escolher_treino)
    btn_sel_treino.pack(side=tk.RIGHT)

    log_treino = scrolledtext.ScrolledText(frame_treino, height=14, wrap=tk.WORD, font=("Consolas", 9))
    log_treino.pack(fill=tk.BOTH, expand=True, pady=8)

    def iniciar_treino():
        p = path_treino.get().strip()
        if not p or not Path(p).exists():
            messagebox.showwarning(tr("warn_header"), tr("warn_select_csv"))
            return
        m = modelo_treino.get().strip().lower() or "svm"
        if m not in MODELOS_DISPONIVEIS:
            messagebox.showwarning(tr("warn_header"), tr("warn_invalid_model", opts=MODELOS_DISPONIVEIS))
            return
        threading.Thread(target=treinar_worker, args=(p, m, log_treino, btn_treino, lang()), daemon=True).start()

    btn_treino = ttk.Button(frame_treino, text="", command=iniciar_treino)
    btn_treino.pack(pady=4)

    # ---- Aba Previsão ----
    frame_prever = ttk.Frame(notebook, padding=10)
    notebook.add(frame_prever, text="")

    lbl_prever_csv = ttk.Label(frame_prever, text="")
    lbl_prever_csv.pack(anchor=tk.W)
    path_prever = tk.StringVar()
    row2 = ttk.Frame(frame_prever)
    row2.pack(fill=tk.X, pady=4)
    entry_prever = ttk.Entry(row2, textvariable=path_prever, width=60)
    entry_prever.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))

    def escolher_prever():
        p = filedialog.askopenfilename(
            filetypes=[("CSV", "*.csv"), ("Todos", "*.*")],
            title=tr("dlg_open_predict_title"),
        )
        if p:
            path_prever.set(p)

    btn_sel_prever = ttk.Button(row2, text="", command=escolher_prever)
    btn_sel_prever.pack(side=tk.RIGHT)

    lbl_save_result = ttk.Label(frame_prever, text="")
    lbl_save_result.pack(anchor=tk.W)
    path_saida = tk.StringVar(value=str(_REPO / "previsoes_resultado.csv"))
    row3 = ttk.Frame(frame_prever)
    row3.pack(fill=tk.X, pady=4)
    entry_saida = ttk.Entry(row3, textvariable=path_saida, width=60)
    entry_saida.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))

    def escolher_saida():
        p = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            title=tr("dlg_save_predict_title"),
        )
        if p:
            path_saida.set(p)

    btn_escolher_saida = ttk.Button(row3, text="", command=escolher_saida)
    btn_escolher_saida.pack(side=tk.RIGHT)

    log_prever = scrolledtext.ScrolledText(frame_prever, height=10, wrap=tk.WORD, font=("Consolas", 9))
    log_prever.pack(fill=tk.BOTH, expand=True, pady=8)

    def iniciar_prever():
        p = path_prever.get().strip()
        s = path_saida.get().strip()
        if not p or not Path(p).exists():
            messagebox.showwarning(tr("warn_header"), tr("warn_predict_csv"))
            return
        if not s:
            messagebox.showwarning(tr("warn_header"), tr("warn_predict_output"))
            return
        threading.Thread(target=prever_worker, args=(p, s, log_prever, btn_prever, lang()), daemon=True).start()

    btn_prever = ttk.Button(frame_prever, text="", command=iniciar_prever)
    btn_prever.pack(pady=4)

    def apply_language() -> None:
        lg = lang()
        root.title(tr("title"))
        lbl_lang_pick.config(text=tr("language"))
        cb_lang.set(LANG_LABEL.get(lg, LANG_LABEL["pt"]))
        notebook.tab(frame_baixar, text=tr("tab_download"))
        notebook.tab(frame_treino, text=tr("tab_train"))
        notebook.tab(frame_prever, text=tr("tab_predict"))
        lbl_baixar_desc.config(text=tr("baixar_desc"))
        rb_todos.config(text=tr("rb_all_tables"))
        rb_com_dom.config(text=tr("rb_domain_only"))
        lbl_save_csv.config(text=tr("save_csv_to"))
        btn_escolher_baixar.config(text=tr("choose_btn"))
        btn_baixar.config(text=tr("btn_download"))
        lbl_modelo.config(text=tr("treino_model_label"))
        lbl_treino_csv.config(text=tr("treino_csv_hint"))
        btn_sel_treino.config(text=tr("select_csv_train"))
        btn_treino.config(text=tr("btn_train"))
        lbl_prever_csv.config(text=tr("prever_csv_hint"))
        btn_sel_prever.config(text=tr("select_csv_predict"))
        lbl_save_result.config(text=tr("save_result_to"))
        btn_escolher_saida.config(text=tr("choose_btn"))
        btn_prever.config(text=tr("btn_predict"))

    def _on_lang_selected(_event: tk.Event | None = None) -> None:
        disp = cb_lang.get()
        new_code = LANG_CODE.get(disp, "pt")
        if new_code == state.get("lang", "pt"):
            return
        state["lang"] = new_code
        apply_language()

    cb_lang.bind("<<ComboboxSelected>>", _on_lang_selected)

    apply_language()
    root.mainloop()


if __name__ == "__main__":
    main()
