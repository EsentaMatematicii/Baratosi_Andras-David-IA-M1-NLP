import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# 1. Exemple didactice de mesaje/comenzi in context ADAS
date_antrenare = [
    ("Activeaza pilotul automat adaptiv.", "COMANDA_ADAS"),
    ("Dezactiveaza asistenta la pastrarea benzii.", "COMANDA_ADAS"),
    ("Seteaza distanta minima fata de masina din fata.", "COMANDA_ADAS"),
    ("Porneste frana automata de urgenta.", "COMANDA_ADAS"),
    ("Ce inseamna avertizarea pentru unghiul mort?", "INTREBARE_SISTEM"),
    ("Este activ acum sistemul de recunoastere a semnelor?", "INTREBARE_SISTEM"),
    ("Cum functioneaza asistenta la parcare?", "INTREBARE_SISTEM"),
    ("De ce s-a dezactivat pilotul automat pe ploaie?", "INTREBARE_SISTEM"),
    ("Avertizarea de banda apare prea des pe drumul asta.", "OBSERVATIE_CONFORT"),
    ("Masina franeaza cam brusc cand intra cineva in fata.", "OBSERVATIE_CONFORT"),
    ("Alerta sonora este prea tare in oras.", "OBSERVATIE_CONFORT"),
    ("Sistemul a reactionat foarte bine la franarea din fata.", "OBSERVATIE_CONFORT"),
]

texte = [item[0] for item in date_antrenare]
etichete = [item[1] for item in date_antrenare]

# 2. Preprocesare in romana (consistenta cu laboratoarele anterioare)
cuvinte_legatura = set(stopwords.words("romanian"))

def preproceseaza_text(text):
    text = text.lower()
    tokeni = re.findall(r"\b\w+\b", text)
    tokeni_curati = [
        t for t in tokeni if not t.isdigit() and t not in cuvinte_legatura
    ]
    return " ".join(tokeni_curati)

texte_curate = [preproceseaza_text(t) for t in texte]

# 3. Pipeline TF-IDF (uni + bigrame) + LogisticRegression
model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("clf", LogisticRegression(random_state=42)),
])

model.fit(texte_curate, etichete)

# 4. Testare pe formulari noi
mesaje_test = [
    "Opreste asistenta de mentinere a benzii acum.",
    "De ce nu mai merge recunoasterea indicatoarelor rutiere?",
    "Franeaza cam tarziu cand incetineste coloana.",
]

print("=== TEST CLASIFICARE INTENTII ADAS ===")
for mesaj in mesaje_test:
    pred = model.predict([preproceseaza_text(mesaj)])[0]
    print(f"Mesaj: {mesaj}")
    print(f"  -> Intentie prezisa: {pred}\n")