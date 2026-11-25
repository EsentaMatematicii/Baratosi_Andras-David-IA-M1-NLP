import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

nltk.download("stopwords", quiet=True)

# 1. Set de exemple pentru antrenare (SPAM vs. LEGITIM)
texte_antrenare = [
    "Felicitări! Ai câștigat un premiu gratuit, apasă acum pe link!",
    "Ofertă specială gratuită doar astăzi, revendică premiul tău!",
    "Câștigă bani rapid fără efort, intră pe linkul de mai jos!",
    "Contul tău va fi blocat, confirmă parola urgent pe link!",
    "Salut, trimit atașat raportul pentru ședința de mâine.",
    "Te rog să verifici documentul atașat și să îmi confirmi.",
    "Ne vedem la ora 10 în sala de seminar pentru laborator.",
    "Am actualizat notițele de curs și le-am urcat în folder.",
]

etichete_antrenare = [
    "SPAM",
    "SPAM",
    "SPAM",
    "SPAM",
    "LEGITIM",
    "LEGITIM",
    "LEGITIM",
    "LEGITIM",
]

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


texte_curate = [preproceseaza_text(t) for t in texte_antrenare]

# 2. Construim si antrenam pipeline-ul TF-IDF + Naive Bayes
model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("clf", MultinomialNB(alpha=0.5)),
])

model.fit(texte_curate, etichete_antrenare)

# 3. Testam pe mesaje noi, nevazute la antrenare
mesaje_noi = [
    "Ai câștigat un telefon gratuit, apasă pe link acum!",
    "Salut, poți să îmi trimiți raportul de la laborator?",
    "Premiu special disponibil astăzi, revendică rapid!",
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