import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# 1. Set mic de exemple pentru antrenare (SPAM vs. LEGITIM)
texte_antrenare = [
    "Felicitari! Ai castigat un premiu gratuit, apasa acum pe link!",
    "Oferta speciala gratuita doar astazi, revendica premiul tau!",
    "Castiga bani rapid fara efort, intra pe linkul de mai jos!",
    "Contul tau va fi blocat, confirma parola urgent pe link!",
    "Salut, trimit atasat raportul pentru sedinta de maine.",
    "Te rog sa verifici documentul atasat si sa imi confirmi.",
    "Ne vedem la ora 10 in sala de seminar pentru laborator.",
    "Am actualizat notitele de curs si le-am urcat in folder."
]

etichete_antrenare = [
    "SPAM",
    "SPAM",
    "SPAM",
    "SPAM",
    "LEGITIM",
    "LEGITIM",
    "LEGITIM",
    "LEGITIM"
]

# 2. Functie de curatare in romana (aceeasi idee ca in Laboratorul 1)
cuvinte_legatura = set(stopwords.words("romanian"))

def preproceseaza_text(text):
    text = text.lower()
    tokeni = re.findall(r"\b\w+\b", text)
    tokeni_curati = [
        t for t in tokeni if not t.isdigit() and t not in cuvinte_legatura
    ]
    return " ".join(tokeni_curati)

texte_curate = [preproceseaza_text(t) for t in texte_antrenare]

# 3. Construim si antrenam pipeline-ul TF-IDF + Naive Bayes
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", MultinomialNB())
])

model.fit(texte_curate, etichete_antrenare)

# 4. Testam pe mesaje noi, nevazute la antrenare
mesaje_noi = [
    "Ai castigat un telefon gratuit, apasa pe link acum!",
    "Salut, poti sa imi trimiti raportul de la laborator?",
    "Premiu special disponibil astazi, revendica rapid!"
]

mesaje_noi_curate = [preproceseaza_text(m) for m in mesaje_noi]
predictii = model.predict(mesaje_noi_curate)
probabilitati = model.predict_proba(mesaje_noi_curate)
clase = model.classes_

print("=== REZULTATE PREDICTII PE MESAJE NOI ===")
for mesaj, pred, probs in zip(mesaje_noi, predictii, probabilitati):
    scoruri = {cls: round(float(p), 3) for cls, p in zip(clase, probs)}
    print(f"Mesaj: {mesaj}")
    print(f"  -> Predictie: {pred} | Probabilitati: {scoruri}\n")