"""
Interface gráfica para treino e previsão do classificador de domínios.
O usuário escolhe o arquivo CSV (para treino ou para previsão) e executa o fluxo.

Uso (na raiz do repositório)::

    python -m classificador_dominio.gui_app
"""
import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path

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
    messagebox.showerror("Erro", "Instale as dependências: pip install pandas numpy scikit-learn")
    sys.exit(1)
try:
    from .openmetadata_client import configurado as om_configurado, _om_url, _auth_headers
except ImportError:
    om_configurado = lambda: False
    _om_url = lambda: ""
    _auth_headers = lambda: {}


def baixar_dados_worker(
    saida_path: str,
    somente_com_dominio: bool,
    log_widget: scrolledtext.ScrolledText,
    btn: ttk.Button,
):
    try:
        import requests as _req
        btn.config(state="disabled")
        log_widget.delete("1.0", tk.END)
        base_url = f"{_om_url()}/api/v1"
        headers = {**_auth_headers(), "accept": "application/json"}
        _req.packages.urllib3.disable_warnings()
        log_widget.insert(tk.END, "Buscando tabelas no catálogo…\n")
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
                log_widget.insert(tk.END, f"  {len(rows)} tabelas com domínio obtidas até agora…\n")
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
                log_widget.insert(tk.END, f"  {len(rows)} tabelas obtidas até agora…\n")
                log_widget.update()
                if not after:
                    break

        log_widget.insert(tk.END, f"Total obtido: {len(rows)} tabelas.\n")
        log_widget.update()
        df = pd.DataFrame(rows)
        df.to_csv(saida_path, index=False, encoding="utf-8-sig")
        log_widget.insert(tk.END, f"\n{len(df)} tabelas exportadas → {saida_path}\n")
        messagebox.showinfo("Sucesso", f"{len(df)} tabelas exportadas para:\n{saida_path}")
    except Exception as e:
        log_widget.insert(tk.END, f"\nErro: {e}\n")
        messagebox.showerror("Erro ao baixar dados", str(e))
    finally:
        btn.config(state="normal")


def treinar_worker(csv_path: str, modelo: str, log_widget: scrolledtext.ScrolledText, btn: ttk.Button):
    try:
        btn.config(state="disabled")
        log_widget.delete("1.0", tk.END)
        log_widget.insert(tk.END, f"Carregando e treinando com: {csv_path}\nModelo: {modelo}\n\n")
        log_widget.update()
        resultado = treinar_com_csv(
            csv_path,
            salvar_modelo_em=MODELO_PADRAO,
            salvar_matriz_em="matriz_confusao.png",
            modelo=modelo,
        )
        log_widget.insert(tk.END, f"Acurácia: {resultado['acuracia']*100:.2f}%\n")
        log_widget.insert(tk.END, f"Treino: {resultado['n_treino']} | Teste: {resultado['n_teste']}\n")
        log_widget.insert(tk.END, f"Classes: {resultado['classes']}\n\n")
        log_widget.insert(tk.END, resultado["relatorio_texto"])
        log_widget.insert(tk.END, f"\n\nModelo salvo em: {MODELO_PADRAO}\n")
        messagebox.showinfo("Sucesso", "Treino concluído. Modelo salvo.")
    except Exception as e:
        log_widget.insert(tk.END, f"\nErro: {e}\n")
        messagebox.showerror("Erro no treino", str(e))
    finally:
        btn.config(state="normal")


def prever_worker(csv_path: str, saida_path: str, log_widget: scrolledtext.ScrolledText, btn: ttk.Button):
    try:
        btn.config(state="disabled")
        log_widget.delete("1.0", tk.END)
        log_widget.insert(tk.END, f"Previsão: {csv_path}\n")
        if not Path(MODELO_PADRAO).exists():
            raise FileNotFoundError("Nenhum modelo encontrado. Treine antes (aba Treino).")
        df = prever_csv(csv_path, modelo_path=MODELO_PADRAO, salvar_resultado_em=saida_path)
        log_widget.insert(tk.END, f"Total de linhas: {len(df)}\n")
        log_widget.insert(tk.END, f"Resultado salvo em: {saida_path}\n")
        cols = [c for c in ["predicted_domain", "confidence", "table_fqn"] if c in df.columns]
        log_widget.insert(tk.END, df[cols].head(10).to_string() if cols else df.head(10).to_string())
        messagebox.showinfo("Sucesso", f"Previsões salvas em:\n{saida_path}")
    except Exception as e:
        log_widget.insert(tk.END, f"\nErro: {e}\n")
        messagebox.showerror("Erro na previsão", str(e))
    finally:
        btn.config(state="normal")


def main():
    root = tk.Tk()
    root.title("Classificador de Domínios — CSV")
    root.geometry("640x520")
    root.minsize(500, 400)

    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

    # ---- Aba Baixar Dados ----
    frame_baixar = ttk.Frame(notebook, padding=10)
    notebook.add(frame_baixar, text="Baixar Dados")

    ttk.Label(
        frame_baixar,
        text="Baixar tabelas do catálogo OpenMetadata e salvar como CSV.",
        font=("", 9),
    ).pack(anchor=tk.W, pady=(0, 6))

    opcao_download = tk.StringVar(value="todos")
    ttk.Radiobutton(
        frame_baixar,
        text="Baixar todos os dados",
        variable=opcao_download,
        value="todos",
    ).pack(anchor=tk.W)
    ttk.Radiobutton(
        frame_baixar,
        text="Baixar somente dados com domínio (catálogo preenchido)",
        variable=opcao_download,
        value="com_dominio",
    ).pack(anchor=tk.W, pady=(0, 8))

    ttk.Label(frame_baixar, text="Salvar CSV em:").pack(anchor=tk.W)
    path_baixar_saida = tk.StringVar(value=str(_REPO / "tabelas_openmetadata.csv"))
    row_baixar = ttk.Frame(frame_baixar)
    row_baixar.pack(fill=tk.X, pady=4)
    ttk.Entry(row_baixar, textvariable=path_baixar_saida, width=60).pack(
        side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8)
    )

    def escolher_saida_baixar():
        p = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            title="Onde salvar o CSV",
        )
        if p:
            path_baixar_saida.set(p)

    ttk.Button(row_baixar, text="Escolher…", command=escolher_saida_baixar).pack(side=tk.RIGHT)

    log_baixar = scrolledtext.ScrolledText(frame_baixar, height=14, wrap=tk.WORD, font=("Consolas", 9))
    log_baixar.pack(fill=tk.BOTH, expand=True, pady=8)

    def iniciar_download():
        if not om_configurado():
            messagebox.showwarning(
                "OpenMetadata",
                "Defina OPENMETADATA_URL e OPENMETADATA_TOKEN nas variáveis de ambiente.",
            )
            return
        saida = path_baixar_saida.get().strip()
        if not saida:
            messagebox.showwarning("Aviso", "Informe onde salvar o arquivo CSV.")
            return
        somente_com_dominio = opcao_download.get() == "com_dominio"
        threading.Thread(
            target=baixar_dados_worker,
            args=(saida, somente_com_dominio, log_baixar, btn_baixar),
            daemon=True,
        ).start()

    btn_baixar = ttk.Button(frame_baixar, text="Baixar dados", command=iniciar_download)
    btn_baixar.pack(pady=4)

    # ---- Aba Treino ----
    frame_treino = ttk.Frame(notebook, padding=10)
    notebook.add(frame_treino, text="Treino")

    ttk.Label(frame_treino, text="Modelo:").pack(anchor=tk.W)
    modelo_treino = tk.StringVar(value=MODELOS_DISPONIVEIS[0] if MODELOS_DISPONIVEIS else "svm")
    row_modelo = ttk.Frame(frame_treino)
    row_modelo.pack(fill=tk.X, pady=2)
    cb_modelo = ttk.Combobox(row_modelo, textvariable=modelo_treino, values=MODELOS_DISPONIVEIS, state="readonly", width=15)
    cb_modelo.pack(side=tk.LEFT)
    if MODELOS_DISPONIVEIS and modelo_treino.get() not in MODELOS_DISPONIVEIS:
        modelo_treino.set(MODELOS_DISPONIVEIS[0])

    ttk.Label(frame_treino, text="CSV para treino (colunas: schema, nome_tabela, qtd_colunas, nome_colunas, dominio):").pack(anchor=tk.W)
    path_treino = tk.StringVar()
    row1 = ttk.Frame(frame_treino)
    row1.pack(fill=tk.X, pady=4)
    ttk.Entry(row1, textvariable=path_treino, width=60).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
    def escolher_treino():
        p = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("Todos", "*.*")], title="Selecione o CSV de treino")
        if p:
            path_treino.set(p)
    ttk.Button(row1, text="Selecionar CSV…", command=escolher_treino).pack(side=tk.RIGHT)

    log_treino = scrolledtext.ScrolledText(frame_treino, height=14, wrap=tk.WORD, font=("Consolas", 9))
    log_treino.pack(fill=tk.BOTH, expand=True, pady=8)

    def iniciar_treino():
        p = path_treino.get().strip()
        if not p or not Path(p).exists():
            messagebox.showwarning("Aviso", "Selecione um arquivo CSV válido.")
            return
        m = modelo_treino.get().strip().lower() or "svm"
        if m not in MODELOS_DISPONIVEIS:
            messagebox.showwarning("Aviso", f"Modelo inválido. Opções: {MODELOS_DISPONIVEIS}")
            return
        threading.Thread(target=treinar_worker, args=(p, m, log_treino, btn_treino), daemon=True).start()

    btn_treino = ttk.Button(frame_treino, text="Treinar modelo", command=iniciar_treino)
    btn_treino.pack(pady=4)

    # ---- Aba Previsão ----
    frame_prever = ttk.Frame(notebook, padding=10)
    notebook.add(frame_prever, text="Previsão")

    ttk.Label(frame_prever, text="CSV para previsão (sem coluna dominio):").pack(anchor=tk.W)
    path_prever = tk.StringVar()
    row2 = ttk.Frame(frame_prever)
    row2.pack(fill=tk.X, pady=4)
    ttk.Entry(row2, textvariable=path_prever, width=60).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
    def escolher_prever():
        p = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("Todos", "*.*")], title="Selecione o CSV para previsão")
        if p:
            path_prever.set(p)
    ttk.Button(row2, text="Selecionar CSV…", command=escolher_prever).pack(side=tk.RIGHT)

    ttk.Label(frame_prever, text="Salvar resultado em:").pack(anchor=tk.W)
    path_saida = tk.StringVar(value=str(_REPO / "previsoes_resultado.csv"))
    row3 = ttk.Frame(frame_prever)
    row3.pack(fill=tk.X, pady=4)
    ttk.Entry(row3, textvariable=path_saida, width=60).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
    def escolher_saida():
        p = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")], title="Onde salvar as previsões")
        if p:
            path_saida.set(p)
    ttk.Button(row3, text="Escolher…", command=escolher_saida).pack(side=tk.RIGHT)

    log_prever = scrolledtext.ScrolledText(frame_prever, height=10, wrap=tk.WORD, font=("Consolas", 9))
    log_prever.pack(fill=tk.BOTH, expand=True, pady=8)

    def iniciar_prever():
        p = path_prever.get().strip()
        s = path_saida.get().strip()
        if not p or not Path(p).exists():
            messagebox.showwarning("Aviso", "Selecione um arquivo CSV válido para previsão.")
            return
        if not s:
            messagebox.showwarning("Aviso", "Informe onde salvar o resultado.")
            return
        threading.Thread(target=prever_worker, args=(p, s, log_prever, btn_prever), daemon=True).start()

    btn_prever = ttk.Button(frame_prever, text="Gerar previsões", command=iniciar_prever)
    btn_prever.pack(pady=4)

    root.mainloop()


if __name__ == "__main__":
    main()