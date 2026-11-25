import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

nltk.download("stopwords", quiet=True)

date = [
    ("Felicitări! Ai câștigat un premiu gratuit, apasă acum pe link!", "SPAM"),
    ("Ofertă specială gratuită doar astăzi, revendică premiul tău!", "SPAM"),
    ("Câștigă bani rapid fără efort, intră pe linkul de mai jos!", "SPAM"),
    ("Contul tău va fi blocat, confirmă parola urgent pe link!", "SPAM"),
    ("Ai câștigat un voucher cadou, completează formularul acum!", "SPAM"),
    ("Promoție exclusivă la telefon nou, ofertă limitată astăzi!", "SPAM"),
    ("Premiu gratuit disponibil astăzi, apasă linkul pentru detalii!", "SPAM"),
    ("Urgent: verifică contul bancar pe linkul securizat acum!", "SPAM"),
    ("Câștigă un bilet gratuit la concert, participă la concurs!", "SPAM"),
    ("Salut, trimit atașat raportul pentru ședința de mâine.", "LEGITIM"),
    ("Te rog să verifici documentul atașat și să îmi confirmi.", "LEGITIM"),
    ("Ne vedem la ora 10 în sala de seminar pentru laborator.", "LEGITIM"),
    ("Am actualizat notițele de curs și le-am urcat în folder.", "LEGITIM"),
    ("Poți să îmi trimiți prezentarea de la cursul de săptămâna trecută?", "LEGITIM"),
    ("Ședința de echipă a fost mutată mâine dimineață la ora nouă.", "LEGITIM"),
    ("Am trimis pe e-mail tema rezolvată și aștept feedback.", "LEGITIM"),
    ("Laboratorul de mâine începe la ora 14 în laboratorul de informatică.", "LEGITIM"),
    ("Mulțumesc pentru ajutorul cu configurarea mediului Python.", "LEGITIM"),
]

texte = [item[0] for item in date]
etichete = [item[1] for item in date]

negatii = {"nu", "nici", "fara", "fără"}
cuvinte_legatura = set(stopwords.words("romanian")) - negatii


def normalizeaza_diacritice(text: str) -> str:
    return (
        text.replace("ş", "ș")
        .replace("ţ", "ț")
        .replace("Ş", "Ș")
        .replace("Ţ", "Ț")
    )


def preproceseaza_text(text: str) -> str:
    text = normalizeaza_diacritice(text).lower()
    tokeni = re.findall(r"\b[a-zăâîșț]+\b", text)
    tokeni_curati = [t for t in tokeni if t not in cuvinte_legatura]
    return " ".join(tokeni_curati)


texte_curate = [preproceseaza_text(t) for t in texte]

X_train, X_test, y_train, y_test = train_test_split(
    texte_curate,
    etichete,
    test_size=0.33,
    random_state=42,
    stratify=etichete,
)

experimente = [
    ("MultinomialNB | Unigrame (1, 1)", (1, 1), MultinomialNB()),
    ("MultinomialNB | Uni+Bigram (1, 2)", (1, 2), MultinomialNB()),
    ("LogisticRegression | Unigrame (1, 1)", (1, 1), LogisticRegression(random_state=42)),
    ("LogisticRegression | Uni+Bigram (1, 2)", (1, 2), LogisticRegression(random_state=42)),
]

rezultate = []
for nume, ngram_range, clf in experimente:
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=ngram_range)),
        ("clf", clf),
    ])

    model.fit(X_train, y_train)
    predictii = model.predict(X_test)
    acuratete = accuracy_score(y_test, predictii)
    f1_m = f1_score(y_test, predictii, average="macro")
    rezultate.append({"Model": nume, "Accuracy": round(acuratete, 4), "F1_macro": round(f1_m, 4)})

    print(f"=== {nume} ===")
    print(f"Acuratete pe test: {acuratete:.2f} | F1-macro: {f1_m:.2f}")
    print("Matrice de confuzie (randuri = real, coloane = prezis):")
    print(confusion_matrix(y_test, predictii, labels=["SPAM", "LEGITIM"]))
    print("Raport clasificare:")
    print(classification_report(y_test, predictii, zero_division=0))

print("=== TABEL COMPARATIV FINAL ===")
print(pd.DataFrame(rezultate).to_string(index=False))