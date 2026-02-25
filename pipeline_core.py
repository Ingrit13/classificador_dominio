"""
Núcleo do pipeline de classificação de domínios.
Funções reutilizáveis para treino (a partir de CSV) e previsão.
"""
import re
import math
import unicodedata
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

ROTULO = "dominio"
TEXT_COLS = ["schema", "nome_tabela", "qtd_colunas", "nome_colunas"]
TEST_SIZE = 0.30
RANDOM_STATE = 42
MODELO_PADRAO = "modelo_svm_treinado.pkl"

# Modelos disponíveis para treino (por enquanto apenas SVM; preparado para outros no futuro)
MODELOS_DISPONIVEIS = ["svm"]


def _criar_classificador(modelo: str):
    """
    Retorna (nome_etapa, estimador) para o Pipeline.
    modelo: "svm" (outros podem ser adicionados depois).
    """
    modelo = (modelo or "svm").strip().lower()
    if modelo not in MODELOS_DISPONIVEIS:
        raise ValueError(f"Modelo '{modelo}' não disponível. Opções: {MODELOS_DISPONIVEIS}")
    if modelo == "svm":
        return ("clf", SVC(kernel="linear", probability=True, class_weight="balanced", random_state=RANDOM_STATE))
    raise ValueError(f"Modelo '{modelo}' não implementado.")


def _preprocess_text(text) -> str:
    if text is None:
        return ""
    text = str(text).lower()
    text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("utf-8")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _limpar_listas(texto) -> str:
    texto = (texto or "").replace("{", " ").replace("}", " ").replace("[", " ").replace("]", " ")
    return texto.replace("_", " ")


def _preparar_texto(row) -> str:
    schema = _limpar_listas(row.get("schema", ""))
    nome_tabela = _limpar_listas(row.get("nome_tabela", ""))
    nome_colunas = _limpar_listas(row.get("nome_colunas", ""))
    qtd = str(row.get("qtd_colunas", "0"))
    return f"{schema}, {nome_tabela}, {qtd}, {nome_colunas}"


def carregar_e_preparar_csv(
    csv_path: str,
    exige_rotulo: bool = True,
) -> pd.DataFrame:
    """
    Carrega o CSV e aplica pré-processamento.
    Colunas esperadas: schema, nome_tabela, qtd_colunas, nome_colunas e (se exige_rotulo) dominio.
    """
    try:
        df = pd.read_csv(csv_path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(csv_path, sep=";", engine="python")

    colunas_obrigatorias = TEXT_COLS + ([ROTULO] if exige_rotulo else [])
    missing = [c for c in colunas_obrigatorias if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes no CSV: {missing}")

    for c in TEXT_COLS:
        if c == "qtd_colunas":
            df[c] = df[c].astype("Int64").astype("string").fillna("0")
        else:
            df[c] = df[c].astype("string").fillna("")

    if exige_rotulo:
        df[ROTULO] = df[ROTULO].astype("string")
        df = df[df[ROTULO].notna() & (df[ROTULO].str.strip() != "")]

    df["texto"] = df.apply(_preparar_texto, axis=1).astype("string").fillna("")
    df["texto"] = df["texto"].apply(_preprocess_text)
    df = df[df["texto"].str.strip() != ""]
    return df


def treinar_com_csv(
    csv_path: str,
    salvar_modelo_em: Optional[str] = None,
    salvar_matriz_em: Optional[str] = None,
    modelo: str = "svm",
):
    """
    Treina o modelo a partir de um CSV de treino (com coluna 'dominio').
    modelo: "svm" (por enquanto única opção; preparado para outros no futuro).
    Retorna: dict com pipeline, metricas (relatorio, acuracia), df usado.
    Opcionalmente salva modelo (.pkl) e figura da matriz de confusão.
    """
    df = carregar_e_preparar_csv(csv_path, exige_rotulo=True)

    counts = df[ROTULO].value_counts()
    min_test = math.ceil(1 / TEST_SIZE)
    min_train = math.ceil(1 / (1 - TEST_SIZE))
    min_per_class = max(min_test, min_train)
    valid_classes = counts[counts >= min_per_class].index
    dropped = sorted(set(counts.index) - set(valid_classes))
    if dropped:
        df = df[df[ROTULO].isin(valid_classes)]
    if df[ROTULO].nunique() < 2:
        raise ValueError("Após a limpeza, há menos de 2 classes. Ajuste o dataset ou agrupe classes raras.")

    X = df["texto"].to_numpy()
    y = df[ROTULO].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    nome_clf, estimador = _criar_classificador(modelo)
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_df=0.9, min_df=2, sublinear_tf=True)),
        (nome_clf, estimador),
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    acuracia = float(accuracy_score(y_test, y_pred))
    report_str = classification_report(y_test, y_pred, zero_division=0)

    if salvar_modelo_em:
        Path(salvar_modelo_em).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, salvar_modelo_em)
    if salvar_matriz_em:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        labels = pipeline.classes_
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.6), max(5, len(labels) * 0.6)))
        ax.imshow(cm)
        ax.set_title("Matriz de Confusão")
        ax.set_xlabel("Classe Predita")
        ax.set_ylabel("Classe Verdadeira")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center")
        fig.tight_layout()
        plt.savefig(salvar_matriz_em, dpi=150)
        plt.close()

    return {
        "pipeline": pipeline,
        "acuracia": acuracia,
        "relatorio_dict": report,
        "relatorio_texto": report_str,
        "n_treino": len(y_train),
        "n_teste": len(y_test),
        "classes": list(pipeline.classes_),
    }


def prever_csv(
    csv_path: str,
    modelo_path: Optional[str] = None,
    salvar_resultado_em: Optional[str] = None,
) -> pd.DataFrame:
    """
    Carrega modelo, aplica em um CSV (sem coluna dominio) e retorna DataFrame
    com colunas originais + predicted_domain + confidence.
    """
    modelo_path = modelo_path or MODELO_PADRAO
    pipeline = joblib.load(modelo_path)
    df = carregar_e_preparar_csv(csv_path, exige_rotulo=False)
    if df.empty:
        raise ValueError("Nenhuma linha válida após pré-processamento.")

    pred = pipeline.predict(df["texto"].to_numpy())
    proba = pipeline.predict_proba(df["texto"].to_numpy())
    conf = np.max(proba, axis=1)

    df = df.copy()
    df["predicted_domain"] = pred
    df["confidence"] = conf.astype(float)
    if "schema" in df.columns and "nome_tabela" in df.columns:
        df["table_fqn"] = df["schema"].astype(str) + "." + df["nome_tabela"].astype(str)

    if salvar_resultado_em:
        df.to_csv(salvar_resultado_em, index=False)
    return df
