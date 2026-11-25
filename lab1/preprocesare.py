import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

nltk.download("stopwords", quiet=True)

# 1. Set de texte demonstrative in limba romana (inclusiv diacritice)
documente = [
    "Felicitări! Ai câștigat un premiu gratuit, apasă acum pe link!",
    "Salut, trimit atașat raportul pentru ședința de mâine.",
    "Ofertă specială gratuită doar astăzi, revendică premiul tău!",
    "Te rog să verifici documentul atașat și să nu îl ignori.",
]

# 2. Resurse lingvistice pentru limba romana (pastram negatiile importante)
negatii = {"nu", "nici", "fara", "fără"}
cuvinte_legatura = set(stopwords.words("romanian")) - negatii
stemmer_ro = SnowballStemmer("romanian")


def normalizeaza_diacritice(text: str) -> str:
    return (
        text.replace("ş", "ș")
        .replace("ţ", "ț")
        .replace("Ş", "Ș")
        .replace("Ţ", "Ț")
    )


def curata_text(text: str, aplica_stemming: bool = True) -> str:
    text = normalizeaza_diacritice(text).lower()
    tokeni = re.findall(r"\b[a-zăâîșț]+\b", text)
    tokeni_curati = [t for t in tokeni if t not in cuvinte_legatura]
    if aplica_stemming:
        tokeni_curati = [stemmer_ro.stem(t) for t in tokeni_curati]
    return " ".join(tokeni_curati)


documente_curate = [curata_text(doc) for doc in documente]

print("=== TEXTE CURATATE SI NORMALIZATE (RO) ===")
for i, doc in enumerate(documente_curate, start=1):
    print(f"Doc {i}: {doc}")

# 3. Reprezentare Bag-of-Words si TF-IDF
bow = CountVectorizer()
matrice_bow = bow.fit_transform(documente_curate)

vectorizator = TfidfVectorizer()
matrice_tfidf = vectorizator.fit_transform(documente_curate)

print("\n=== VOCABULAR (CUVINTELE INVATATE) ===")
print(vectorizator.get_feature_names_out())

print("\n=== DIMENSIUNI MATRICE (documente x termeni) ===")
print(f"BoW: {matrice_bow.shape} | TF-IDF: {matrice_tfidf.shape}")