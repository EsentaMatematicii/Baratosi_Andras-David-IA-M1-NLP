import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# 1. Un mic set de texte (exemplu: mesaje / e-mailuri scurte)
documente = [
    "Felicitari! Ai castigat un premiu gratuit, apasa acum pe link!",
    "Salut, trimit atasat raportul pentru sedinta de maine.",
    "Oferta speciala gratuita doar astazi, revendica premiul tau!",
    "Te rog sa verifici documentul atasat si sa imi confirmi."
]

# 2. Lista de cuvinte de legatura (stopwords) in romana
cuvinte_legatura = set(stopwords.words("romanian"))

def curata_text(text):
    # Transformam in litere mici
    text = text.lower()
    # Extragem doar cuvintele (fara punctuatie)
    tokeni = re.findall(r"\b\w+\b", text)
    # Pastram doar cuvintele care nu sunt cifre si nu sunt cuvinte de legatura
    tokeni_curati = [
        t for t in tokeni if not t.isdigit() and t not in cuvinte_legatura
    ]
    return " ".join(tokeni_curati)

# Aplicam curatarea pe fiecare document
documente_curate = [curata_text(doc) for doc in documente]

print("=== TEXTE CURATATE ===")
for i, doc in enumerate(documente_curate, start=1):
    print(f"Doc {i}: {doc}")

# 3. Transformam textele in numere folosind TF-IDF
vectorizator = TfidfVectorizer()
matrice_tfidf = vectorizator.fit_transform(documente_curate)

print("\n=== VOCABULAR (CUVINTELE INVATATE) ===")
print(vectorizator.get_feature_names_out())

print("\n=== DIMENSIUNEA MATRICEI NUMERICE (documente x cuvinte) ===")
print(matrice_tfidf.shape)