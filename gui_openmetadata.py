"""
Janela só com o fluxo OpenMetadata (catálogo, previsão em memória, aplicar domínios).
Requer o mesmo modelo treinado que a GUI completa (MODELO_PADRAO em pipeline_core).

Uso: py gui_openmetadata.py
"""
import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
os.chdir(_script_dir)

try:
    from dotenv import load_dotenv

    load_dotenv(_script_dir / ".env")
except ImportError:
    pass

try:
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


def main() -> None:
    root = tk.Tk()
    root.title("OpenMetadata — Classificador de domínios")
    root.geometry("720x560")
    root.minsize(520, 420)

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
            "Fluxo: 1) Carregar lista do catálogo (databases) no combobox; "
            "2) escolher uma base; 3) Buscar tabelas e gerar previsões; "
            "4) Aplicar domínios no OpenMetadata. "
            "«Listar domínios» só mostra domínios no log. "
            "O modelo deve existir (treine com gui_app.py na aba Treino ou copie o .pkl esperado)."
        ),
        wraplength=660,
    ).pack(anchor=tk.W, pady=(0, 6))

    om_state: dict = {"bases": [], "df_pred": None}

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

    def om_carregar_bases() -> None:
        log_om.delete("1.0", tk.END)
        if not om_configurado():
            log_om.insert(tk.END, "Defina OPENMETADATA_URL e OPENMETADATA_TOKEN nas variáveis de ambiente.\n")
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

    row_om2 = ttk.Frame(frame)
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

    def om_buscar_tabelas_e_prever_worker() -> None:
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
                messagebox.showwarning(
                    "Aviso",
                    f"Modelo não encontrado em:\n{MODELO_PADRAO}\n\n"
                    "Treine com gui_app.py (aba Treino) ou copie o arquivo do modelo para esse caminho.",
                )
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

    def om_buscar_tabelas_e_prever() -> None:
        threading.Thread(target=om_buscar_tabelas_e_prever_worker, daemon=True).start()

    btn_om_buscar.config(command=om_buscar_tabelas_e_prever)
    btn_om_buscar.pack(side=tk.RIGHT)

    log_om = scrolledtext.ScrolledText(frame, height=14, wrap=tk.WORD, font=("Consolas", 9))
    log_om.pack(fill=tk.BOTH, expand=True, pady=8)

    def om_listar() -> None:
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

    def om_aplicar() -> None:
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
