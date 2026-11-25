import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

date = [
    ("Felicitari! Ai castigat un premiu gratuit, apasa acum pe link!", "SPAM"),
    ("Oferta speciala gratuita doar astazi, revendica premiul tau!", "SPAM"),
    ("Castiga bani rapid fara efort, intra pe linkul de mai jos!", "SPAM"),
    ("Contul tau va fi blocat, confirma parola urgent pe link!", "SPAM"),
    ("Ai castigat un voucher cadou, completeaza formularul acum!", "SPAM"),
    ("Promotie exclusiva la telefon nou, oferta limitata astazi!", "SPAM"),
    ("Premiu gratuit disponibil astazi, apasa linkul pentru detalii!", "SPAM"),
    ("Urgent: verifica contul bancar pe linkul securizat acum!", "SPAM"),
    ("Castiga un bilet gratuit la concert, participa la concurs!", "SPAM"),
    ("Salut, trimit atasat raportul pentru sedinta de maine.", "LEGITIM"),
    ("Te rog sa verifici documentul atasat si sa imi confirmi.", "LEGITIM"),
    ("Ne vedem la ora 10 in sala de seminar pentru laborator.", "LEGITIM"),
    ("Am actualizat notitele de curs si le-am urcat in folder.", "LEGITIM"),
    ("Poti sa imi trimiti prezentarea de la cursul de saptamana trecuta?", "LEGITIM"),
    ("Sedinta de echipa a fost mutata maine dimineata la ora noua.", "LEGITIM"),
    ("Am trimis pe e-mail tema rezolvata si astept feedback.", "LEGITIM"),
    ("Laboratorul de maine incepe la ora 14 in laboratorul de informatica.", "LEGITIM"),
    ("Multumesc pentru ajutorul cu configurarea mediului Python.", "LEGITIM"),
]

texte = [item[0] for item in date]
etichete = [item[1] for item in date]

cuvinte_legatura = set(stopwords.words("romanian"))

def preproceseaza_text(text):
    text = text.lower()
    tokeni = re.findall(r"\b\w+\b", text)
    tokeni_curati = [
        t for t in tokeni if not t.isdigit() and t not in cuvinte_legatura
    ]
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

for nume, ngram_range, clf in experimente:
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=ngram_range)),
        ("clf", clf),
    ])

    model.fit(X_train, y_train)
    predictii = model.predict(X_test)
    acuratete = accuracy_score(y_test, predictii)

    print(f"=== {nume} ===")
    print(f"Acuratete pe test: {acuratete:.2f}")
    print("Matrice de confuzie (randuri = real, coloane = prezis):")
    print(confusion_matrix(y_test, predictii, labels=["SPAM", "LEGITIM"]))
    print("Raport clasificare:")
    print(classification_report(y_test, predictii, zero_division=0))