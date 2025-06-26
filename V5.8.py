import pandas as pd
import re
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# === 1. Carregar o dataset ===
df = pd.read_csv("dataset_V8.csv")

# === 2. Função de pré-processamento ===
def preprocess_text(text):
    text = text.lower()
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# === 3. Combinar colunas em uma só ===
def preparar_texto(row):
    schema = row['schema'].replace("_", " ")
    nome_tabela = row['nome_tabela'].replace("_", " ")
    nome_colunas = row['nome_colunas'].replace("{", "").replace("}", "").replace("_", " ")
    return f"{schema}, {nome_tabela}, {row['qtd_colunas']}, {nome_colunas}"

df['texto'] = df.apply(preparar_texto, axis=1)
df['texto'] = df['texto'].apply(preprocess_text)

# === 4. Definir variáveis ===
X = df['texto']
y = df['dominio']

# === 5. Dividir treino e teste ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# === 6. Pipeline com TF-IDF e SVM ===
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_df=0.9, min_df=2)),
    ('svm', SVC(kernel='linear', probability=True, class_weight='balanced'))
])

# === 7. Treinar modelo ===
pipeline.fit(X_train, y_train)

# === 8. Avaliação ===
y_pred = pipeline.predict(X_test)
print("Relatório de Classificação:\n", classification_report(y_test, y_pred))
print(f"Acurácia: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# === 9. Matriz de confusão ===
cm = confusion_matrix(y_test, y_pred, labels=pipeline.classes_)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=pipeline.classes_, yticklabels=pipeline.classes_)
plt.title("Matriz de Confusão")
plt.xlabel("Classe Predita")
plt.ylabel("Classe Verdadeira")
plt.tight_layout()
plt.savefig("matriz_confusao.png", dpi=300)
plt.show()

# === 10. Salvar modelo ===
joblib.dump(pipeline, "modelo_svm_treinado.pkl")
