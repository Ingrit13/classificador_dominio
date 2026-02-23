# Script para executar o pipeline (preparação + treino + avaliação) fora do Jupyter.
# Uso: py run_pipeline.py

import os
import re
import math
import unicodedata

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")  # backend sem display
import matplotlib.pyplot as plt

# --- Parâmetros ---
CSV_PATH = "dataset_V9.csv"
ROTULO = "dominio"
TEXT_COLS = ["schema", "nome_tabela", "qtd_colunas", "nome_colunas"]
TEST_SIZE = 0.30
RANDOM_STATE = 42

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("=== 1) Preparação dos Dados & Split ===\n")

    # Carregar dataset
    try:
        df = pd.read_csv(CSV_PATH, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(CSV_PATH, sep=";", engine="python")

    missing = [c for c in TEXT_COLS + [ROTULO] if c not in df.columns]
    if missing:
        raise KeyError(f"As colunas {missing} não foram encontradas no CSV.")

    for c in TEXT_COLS:
        if c == "qtd_colunas":
            df[c] = df[c].astype("Int64").astype("string").fillna("0")
        else:
            df[c] = df[c].astype("string").fillna("")
    df[ROTULO] = df[ROTULO].astype("string")
    df = df[df[ROTULO].notna()]
    df = df[df[ROTULO].str.strip() != ""]

    # Pré-processamento
    def preprocess_text(text):
        if text is None:
            return ""
        text = str(text).lower()
        text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("utf-8")
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def limpar_listas(texto):
        texto = (texto or "").replace("{", " ").replace("}", " ").replace("[", " ").replace("]", " ")
        texto = texto.replace("_", " ")
        return texto

    def preparar_texto(row):
        schema = limpar_listas(row.get("schema", ""))
        nome_tabela = limpar_listas(row.get("nome_tabela", ""))
        nome_colunas = limpar_listas(row.get("nome_colunas", ""))
        qtd = str(row.get("qtd_colunas", "0"))
        return f"{schema}, {nome_tabela}, {qtd}, {nome_colunas}"

    df["texto"] = df.apply(preparar_texto, axis=1).astype("string").fillna("")
    df["texto"] = df["texto"].apply(preprocess_text)
    df = df[df["texto"].str.strip() != ""]

    # Classes raras
    counts = df[ROTULO].value_counts()
    min_test = math.ceil(1 / TEST_SIZE)
    min_train = math.ceil(1 / (1 - TEST_SIZE))
    min_per_class = max(min_test, min_train)
    valid_classes = counts[counts >= min_per_class].index
    dropped = sorted(set(counts.index) - set(valid_classes))
    if dropped:
        print(f"[AVISO] Removendo classes raras (< {min_per_class} amostras): {dropped}")
        df = df[df[ROTULO].isin(valid_classes)]

    if df[ROTULO].nunique() < 2:
        raise ValueError("Após a limpeza, há menos de 2 classes.")

    X = df["texto"].to_numpy()
    y = df[ROTULO].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Split: train={len(y_train)} | test={len(y_test)}")
    np.save("X_train.npy", X_train)
    np.save("X_test.npy", X_test)
    np.save("y_train.npy", y_train)
    np.save("y_test.npy", y_test)
    df[["texto", ROTULO]].to_csv("texto_preprocessado.csv", index=False)
    print("Arquivos salvos: X_train.npy, X_test.npy, y_train.npy, y_test.npy, texto_preprocessado.csv\n")

    print("=== 2) Treino & Avaliação (TF-IDF + SVM) ===\n")
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_df=0.9, min_df=2, sublinear_tf=True)),
        ("svm", SVC(kernel="linear", probability=True, class_weight="balanced", random_state=42)),
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print("Relatório de Classificação:\n", classification_report(y_test, y_pred, zero_division=0))
    print(f"Acurácia: {accuracy_score(y_test, y_pred) * 100:.2f}%")

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
    plt.savefig("matriz_confusao.png", dpi=150)
    plt.close()
    print("Matriz de confusão salva em: matriz_confusao.png")

    joblib.dump(pipeline, "modelo_svm_treinado.pkl")
    print("Modelo salvo em: modelo_svm_treinado.pkl")
    print("\n=== Pipeline concluído com sucesso. ===")

if __name__ == "__main__":
    main()
