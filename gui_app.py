"""
Interface gráfica para treino e previsão do classificador de domínios.
O usuário escolhe o arquivo CSV (para treino ou para previsão) e executa o fluxo.
Uso: py gui_app.py
"""
import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path

# Garante que o diretório do script é o atual
_script_dir = Path(__file__).resolve().parent
os.chdir(_script_dir)

try:
    from dotenv import load_dotenv

    load_dotenv(_script_dir / ".env")
except ImportError:
    pass

try:
    from pipeline_core import treinar_com_csv, prever_csv, prever_dataframe, MODELO_PADRAO, MODELOS_DISPONIVEIS
    import pandas as pd
except ImportError:
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
    om_configurado = lambda: False
    listar_dominios = lambda: []
    aplicar_dominios = lambda itens, **kw: {"total": 0, "sucesso": 0, "falhas": 0, "logs": ["Módulo openmetadata_client não disponível."]}
    listar_bases_dados = lambda: []
    listar_tabelas_por_database = lambda fqn: []
    tabelas_openmetadata_para_dataframe = lambda t: pd.DataFrame()


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
    path_saida = tk.StringVar(value=str(Path(__file__).resolve().parent / "previsoes_resultado.csv"))
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

    # ---- Aba OpenMetadata ----
    frame_om = ttk.Frame(notebook, padding=10)
    notebook.add(frame_om, text="OpenMetadata")

    om_status = "Configurado (OPENMETADATA_URL e OPENMETADATA_TOKEN definidos)" if om_configurado() else "Não configurado — defina OPENMETADATA_URL e OPENMETADATA_TOKEN nas variáveis de ambiente"
    ttk.Label(frame_om, text=f"Status: {om_status}", font=("", 9)).pack(anchor=tk.W)
    ttk.Label(
        frame_om,
        text=(
            "Fluxo: 1) Carregar lista do catálogo (databases) no combobox abaixo; "
            "2) escolher uma base; 3) Buscar tabelas e gerar previsões; "
            "4) Aplicar domínios no OpenMetadata. "
            "«Listar domínios» só mostra domínios no log — não preenche o combobox."
        ),
        wraplength=620,
    ).pack(anchor=tk.W, pady=(0, 6))

    om_state = {"bases": [], "df_pred": None}

    def _om_label_base(b: dict) -> str:
        svc = b.get("service_pai") or "—"
        return f"{b.get('name')} | {b.get('fullyQualifiedName')}  (serviço: {svc})"

    row_om1 = ttk.Frame(frame_om)
    row_om1.pack(fill=tk.X, pady=4)
    ttk.Label(row_om1, text="Base (database):").pack(side=tk.LEFT, padx=(0, 8))
    # Botão antes do Combobox: com expand no combo, empacotar o botão à direita o escondia fora da janela.
    btn_om_carregar_bases = ttk.Button(row_om1, text="Carregar lista do catálogo")
    btn_om_carregar_bases.pack(side=tk.LEFT, padx=(0, 8))
    cb_base_fqn = ttk.Combobox(row_om1, width=48, state="readonly")
    cb_base_fqn.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def om_carregar_bases():
        log_om.delete("1.0", tk.END)
        if not om_configurado():
            log_om.insert(tk.END, "Defina OPENMETADATA_URL e OPENMETADATA_TOKEN nas variáveis de ambiente.\n")
            messagebox.showwarning("OpenMetadata", "Variáveis de ambiente não configuradas.")
            return
        try:
            bases = listar_bases_dados()
            om_state["bases"] = bases
            labels = [_om_label_base(b) for b in bases]
            # No Windows, Combobox em state=readonly muitas vezes não aplica values/set; alternar estado corrige.
            cb_base_fqn.configure(state="normal")
            cb_base_fqn.set("")
            cb_base_fqn["values"] = labels
            if labels:
                cb_base_fqn.set(labels[0])
            cb_base_fqn.configure(state="readonly")
            cb_base_fqn.update_idletasks()
            log_om.insert(tk.END, f"Carregadas {len(bases)} base(s) (databases).\n")
            if not bases:
                log_om.insert(
                    tk.END,
                    "Nenhuma database encontrada. Verifique ingestão no OpenMetadata e permissões do token.\n",
                )
        except Exception as e:
            log_om.insert(tk.END, f"Erro ao listar bases: {e}\n")
            messagebox.showerror("OpenMetadata", str(e))

    btn_om_carregar_bases.config(command=om_carregar_bases)

    row_om2 = ttk.Frame(frame_om)
    row_om2.pack(fill=tk.X, pady=4)
    lbl_om_pred = ttk.Label(row_om2, text="Nenhuma previsão em memória.")
    lbl_om_pred.pack(side=tk.LEFT)

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

    btn_om_buscar = ttk.Button(row_om2, text="Buscar tabelas e gerar previsões")

    def om_buscar_tabelas_e_prever_worker():
        try:
            btn_om_buscar.config(state="disabled")
            log_om.delete("1.0", tk.END)
            if not om_configurado():
                log_om.insert(tk.END, "Defina OPENMETADATA_URL e OPENMETADATA_TOKEN.\n")
                return
            fqn_db = _om_fqn_base_selecionada()
            if not fqn_db:
                messagebox.showwarning(
                    "Aviso",
                    "Use «Carregar lista do catálogo» acima e selecione uma base no combobox.",
                )
                return
            if not Path(MODELO_PADRAO).exists():
                messagebox.showwarning("Aviso", "Treine o modelo na aba Treino antes.")
                return
            log_om.insert(tk.END, f"Buscando tabelas da base: {fqn_db}\n")
            log_om.update()
            tabelas = listar_tabelas_por_database(fqn_db)
            if not tabelas:
                log_om.insert(tk.END, "Nenhuma tabela retornada pela API para esta base.\n")
                om_state["df_pred"] = None
                lbl_om_pred.config(text="Nenhuma previsão em memória.")
                return
            df_in = tabelas_openmetadata_para_dataframe(tabelas)
            log_om.insert(tk.END, f"Tabelas obtidas: {len(df_in)}. Gerando previsões…\n")
            log_om.update()
            df_pred = prever_dataframe(df_in, modelo_path=MODELO_PADRAO, salvar_resultado_em=None)
            om_state["df_pred"] = df_pred
            lbl_om_pred.config(text=f"Previsões em memória: {len(df_pred)} tabela(s).")
            log_om.insert(tk.END, df_pred[["table_fqn", "predicted_domain", "confidence"]].head(15).to_string())
            if len(df_pred) > 15:
                log_om.insert(tk.END, f"\n… e mais {len(df_pred) - 15} linha(s).\n")
            log_om.insert(tk.END, "\nUse 'Aplicar domínios' para gravar no catálogo.\n")
        except Exception as e:
            om_state["df_pred"] = None
            lbl_om_pred.config(text="Nenhuma previsão em memória.")
            log_om.insert(tk.END, f"Erro: {e}\n")
            messagebox.showerror("Erro", str(e))
        finally:
            btn_om_buscar.config(state="normal")

    def om_buscar_tabelas_e_prever():
        threading.Thread(target=om_buscar_tabelas_e_prever_worker, daemon=True).start()

    btn_om_buscar.config(command=om_buscar_tabelas_e_prever)
    btn_om_buscar.pack(side=tk.RIGHT)

    log_om = scrolledtext.ScrolledText(frame_om, height=12, wrap=tk.WORD, font=("Consolas", 9))
    log_om.pack(fill=tk.BOTH, expand=True, pady=8)

    def om_listar():
        log_om.delete("1.0", tk.END)
        if not om_configurado():
            log_om.insert(tk.END, "Defina OPENMETADATA_URL e OPENMETADATA_TOKEN nas variáveis de ambiente.\n")
            return
        try:
            doms = listar_dominios()
            log_om.insert(tk.END, f"Domínios ({len(doms)}):\n")
            for d in doms:
                log_om.insert(tk.END, f"  - {d.get('name')} (id={d.get('id')})\n")
        except Exception as e:
            log_om.insert(tk.END, f"Erro: {e}\n")

    def om_aplicar():
        log_om.delete("1.0", tk.END)
        if not om_configurado():
            log_om.insert(tk.END, "Defina OPENMETADATA_URL e OPENMETADATA_TOKEN nas variáveis de ambiente.\n")
            messagebox.showwarning("OpenMetadata", "Variáveis de ambiente não configuradas.")
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
            itens = df[["table_fqn", "predicted_domain"]].to_dict("records")
            log_om.insert(tk.END, f"Aplicando {len(itens)} linha(s) no OpenMetadata…\n\n")
            log_om.update()
            resultado = aplicar_dominios(itens)
            log_om.insert(tk.END, f"Total: {resultado['total']} | Sucesso: {resultado['sucesso']} | Falhas: {resultado['falhas']}\n\n")
            for line in resultado["logs"]:
                log_om.insert(tk.END, line + "\n")
            messagebox.showinfo("Concluído", f"Sucesso: {resultado['sucesso']} | Falhas: {resultado['falhas']}")
        except Exception as e:
            log_om.insert(tk.END, f"Erro: {e}\n")
            messagebox.showerror("Erro", str(e))

    btn_row_om = ttk.Frame(frame_om)
    btn_row_om.pack(pady=4)
    ttk.Button(btn_row_om, text="Listar domínios", command=om_listar).pack(side=tk.LEFT, padx=(0, 8))
    ttk.Button(btn_row_om, text="Aplicar domínios no OpenMetadata", command=om_aplicar).pack(side=tk.LEFT)

    root.mainloop()


if __name__ == "__main__":
    main()
