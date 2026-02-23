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
os.chdir(Path(__file__).resolve().parent)

try:
    from pipeline_core import treinar_com_csv, prever_csv, MODELO_PADRAO, MODELOS_DISPONIVEIS
    import pandas as pd
except ImportError:
    messagebox.showerror("Erro", "Instale as dependências: pip install pandas numpy scikit-learn")
    sys.exit(1)
try:
    from openmetadata_client import configurado as om_configurado, listar_dominios, aplicar_dominios
except ImportError:
    om_configurado = lambda: False
    listar_dominios = lambda: []
    aplicar_dominios = lambda itens, **kw: {"total": 0, "sucesso": 0, "falhas": 0, "logs": ["Módulo openmetadata_client não disponível."]}


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
    ttk.Label(frame_om, text="CSV com colunas table_fqn e predicted_domain (ex.: resultado da Previsão):").pack(anchor=tk.W)
    path_om = tk.StringVar()
    row_om = ttk.Frame(frame_om)
    row_om.pack(fill=tk.X, pady=4)
    ttk.Entry(row_om, textvariable=path_om, width=60).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
    def escolher_om():
        p = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("Todos", "*.*")], title="CSV com table_fqn e predicted_domain")
        if p:
            path_om.set(p)
    ttk.Button(row_om, text="Selecionar CSV…", command=escolher_om).pack(side=tk.RIGHT)

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
        p = path_om.get().strip()
        if not p or not Path(p).exists():
            messagebox.showwarning("Aviso", "Selecione um CSV com table_fqn e predicted_domain.")
            return
        try:
            df = pd.read_csv(p)
            for col in ["table_fqn", "predicted_domain"]:
                if col not in df.columns:
                    messagebox.showerror("Erro", f"CSV deve conter coluna '{col}'.")
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
