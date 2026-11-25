import os
import re
import time
import unicodedata
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import FeatureUnion, Pipeline

nltk.download("stopwords", quiet=True)

SEED = 42
np.random.seed(SEED)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# 1. Corpus ADAS in limba romana (3 clase: COMANDA_ADAS, INTREBARE_SISTEM, OBSERVATIE_CONFORT)
DATE_BRUTE = [
    # --- COMANDA_ADAS ---
    ("Activează pilotul automat adaptiv.", "COMANDA_ADAS"),
    ("Pornește ACC la 130 km/h pe autostradă.", "COMANDA_ADAS"),
    ("Dezactivează asistența la păstrarea benzii.", "COMANDA_ADAS"),
    ("Oprește sistemul LKA acum.", "COMANDA_ADAS"),
    ("Setează distanța minimă față de mașina din față.", "COMANDA_ADAS"),
    ("Reglează distanța ACC la 2 secunde.", "COMANDA_ADAS"),
    ("Pornește frâna automată de urgență AEB.", "COMANDA_ADAS"),
    ("Activează monitorizarea unghiului mort BSM.", "COMANDA_ADAS"),
    ("Crește viteza țintă la 110 km/h.", "COMANDA_ADAS"),
    ("Scade viteza pilotului automat la 90 km/h.", "COMANDA_ADAS"),
    ("Activează asistența la parcare.", "COMANDA_ADAS"),
    ("Oprește avertizarea sonoră pentru depășirea benzii.", "COMANDA_ADAS"),
    ("Setează limitatorul de viteză la 50 km/h în localitate.", "COMANDA_ADAS"),
    ("Activează recunoașterea indicatoarelor rutiere.", "COMANDA_ADAS"),
    ("Dezactivează pilotul automat imediat.", "COMANDA_ADAS"),
    ("Pornește menținerea activă pe centrul benzii.", "COMANDA_ADAS"),
    ("Mărește distanța de siguranță la 3 secunde.", "COMANDA_ADAS"),
    ("Oprește asistența la schimbarea benzii.", "COMANDA_ADAS"),
    ("Activează faza lungă automată.", "COMANDA_ADAS"),
    ("Dezactivează avertizarea de coliziune frontală.", "COMANDA_ADAS"),
    ("Setează viteza de croazieră la 100 km/h.", "COMANDA_ADAS"),
    ("Pornește senzorii de parcare față și spate.", "COMANDA_ADAS"),
    ("Oprește alerta pentru unghiul mort pe drumul ăsta.", "COMANDA_ADAS"),
    ("Activează frânarea automată la mersul înapoi.", "COMANDA_ADAS"),
    # --- INTREBARE_SISTEM ---
    ("Ce înseamnă avertizarea pentru unghiul mort?", "INTREBARE_SISTEM"),
    ("Este activ acum sistemul de recunoaștere a semnelor?", "INTREBARE_SISTEM"),
    ("Cum funcționează asistența la parcare?", "INTREBARE_SISTEM"),
    ("De ce s-a dezactivat pilotul automat pe ploaie?", "INTREBARE_SISTEM"),
    ("La ce viteză intră în funcțiune AEB?", "INTREBARE_SISTEM"),
    ("Cum pot regla distanța la ACC?", "INTREBARE_SISTEM"),
    ("De ce nu mai merge recunoașterea indicatoarelor rutiere?", "INTREBARE_SISTEM"),
    ("Care este viteza maximă pentru menținerea benzii LKA?", "INTREBARE_SISTEM"),
    ("Sistemul BSM detectează și motocicletele?", "INTREBARE_SISTEM"),
    ("De ce clipesc martorii pentru radarul frontal?", "INTREBARE_SISTEM"),
    ("Cum activez limitatorul de viteză adaptiv?", "INTREBARE_SISTEM"),
    ("Funcționează pilotul automat și în coloană la semafor?", "INTREBARE_SISTEM"),
    ("Ce semnifică simbolul portocaliu de pe bord pentru bandă?", "INTREBARE_SISTEM"),
    ("De ce s-a oprit brusc asistența la păstrarea benzii?", "INTREBARE_SISTEM"),
    ("Pot folosi ACC când ninge sau e ceață densă?", "INTREBARE_SISTEM"),
    ("Unde văd setarea curentă pentru distanța frontală?", "INTREBARE_SISTEM"),
    ("Cum dezactivez temporar frânarea de urgență?", "INTREBARE_SISTEM"),
    ("De ce nu detectează marcajele pe drumul acesta?", "INTREBARE_SISTEM"),
    ("Este pornită acum asistența la menținerea benzii?", "INTREBARE_SISTEM"),
    ("Ce face sistemul dacă nu țin mâinile pe volan?", "INTREBARE_SISTEM"),
    ("De ce apare mesajul senzor radar murdar?", "INTREBARE_SISTEM"),
    ("Cum schimb unitatea de măsură pentru viteză?", "INTREBARE_SISTEM"),
    ("La ce distanță avertizează senzorii laterali?", "INTREBARE_SISTEM"),
    ("De ce nu pornește pilotul automat sub 30 km/h?", "INTREBARE_SISTEM"),
    # --- OBSERVATIE_CONFORT ---
    ("Avertizarea de bandă apare prea des pe drumul ăsta.", "OBSERVATIE_CONFORT"),
    ("Mașina frânează cam brusc când intră cineva în față.", "OBSERVATIE_CONFORT"),
    ("Alerta sonoră este prea tare în oraș.", "OBSERVATIE_CONFORT"),
    ("Sistemul a reacționat foarte bine la frânarea din față.", "OBSERVATIE_CONFORT"),
    ("Frânează cam târziu când încetinește coloana.", "OBSERVATIE_CONFORT"),
    ("Volanul trage prea agresiv spre stânga la corectarea benzii.", "OBSERVATIE_CONFORT"),
    ("Pilotul automat accelerează prea lent după depășire.", "OBSERVATIE_CONFORT"),
    ("Este foarte confortabil ACC-ul pe autostradă la drum lung.", "OBSERVATIE_CONFORT"),
    ("Avertizarea de coliziune se declanșează fals în curbe strânse.", "OBSERVATIE_CONFORT"),
    ("Menținerea benzii vibrează cam enervant pe drumuri înguste.", "OBSERVATIE_CONFORT"),
    ("Distanța minimă lasă prea mult loc și se bagă alte mașini.", "OBSERVATIE_CONFORT"),
    ("Asistența la parcare a parcat impecabil între două mașini.", "OBSERVATIE_CONFORT"),
    ("Senzorii piuie continuu chiar dacă mai e destul spațiu.", "OBSERVATIE_CONFORT"),
    ("Frânarea automată m-a speriat puțin la o umbră pe asfalt.", "OBSERVATIE_CONFORT"),
    ("Tranziția la oprire în trafic bară la bară este foarte lină.", "OBSERVATIE_CONFORT"),
    ("Corecția de direcție este cam smucită pe denivelări.", "OBSERVATIE_CONFORT"),
    ("Îmi place cât de precis citește limitele de viteză.", "OBSERVATIE_CONFORT"),
    ("Semnalul acustic pentru unghiul mort este deranjant de des.", "OBSERVATIE_CONFORT"),
    ("Accelerația pilotului adaptiv e prea bruscă în modul sport.", "OBSERVATIE_CONFORT"),
    ("Pe ploaie sistemul devine cam ezitant la marcajele șterse.", "OBSERVATIE_CONFORT"),
    ("Avertizarea de oboseală a apărut prea devreme azi.", "OBSERVATIE_CONFORT"),
    ("Frâna de urgență a intervenit salvator la trecerea de pietoni.", "OBSERVATIE_CONFORT"),
    ("Zgomotul alertelor din bord este obositor la drum lung.", "OBSERVATIE_CONFORT"),
    ("Păstrarea benzii merge mult mai fluid după ultimul update.", "OBSERVATIE_CONFORT"),
]


# 2. Preprocesare avansata (6+ tehnici pentru limba romana + domeniul ADAS)
NEGATII = {"nu", "nici", "fara", "fără", "niciodata", "niciodată"}
STOPWORDS_RO = set(stopwords.words("romanian")) - NEGATII
STEMMER_RO = SnowballStemmer("romanian")

ABREVIERI_ADAS = {
    r"\bacc\b": "pilot automat adaptiv",
    r"\blka\b": "mentinere banda",
    r"\blkas\b": "mentinere banda",
    r"\baeb\b": "franare automata urgenta",
    r"\bbsm\b": "unghi mort",
    r"\bfcw\b": "avertizare coliziune frontala",
}


def elimina_diacritice(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def normalizeaza_text_adas(
    text: str,
    expand_abrevieri: bool = True,
    aplica_stemming: bool = True,
) -> str:
    # Tehnica 1: uniformizare diacritice cu sedila -> virgula + eliminare diacritice pentru robustete ASR
    t = (
        text.replace("ş", "ș")
        .replace("ţ", "ț")
        .replace("Ş", "Ș")
        .replace("Ţ", "Ț")
    )
    t = elimina_diacritice(t).lower().strip()

    # Tehnica 2: mascare valori numerice + unitati (viteza / timp / distanta)
    t = re.sub(r"\b\d+\s*(km/h|kmh|km)\b", " <VITEZA> ", t)
    t = re.sub(r"\b\d+\s*(secunde|sec|s)\b", " <TIMP> ", t)
    t = re.sub(r"\b\d+\b", " <NUM> ", t)

    # Tehnica 3: expandare abrevieri tehnice ADAS
    if expand_abrevieri:
        for pattern, inlocuire in ABREVIERI_ADAS.items():
            t = re.sub(pattern, f" {inlocuire} ", t)

    # Tehnica 4: curatare punctuatie si tokenizare
    tokeni = re.findall(r"<[A-Z]+>|\b[a-z]+\b", t)

    # Tehnica 5: eliminare stopwords pastrand negatiile
    tokeni = [tok for tok in tokeni if tok not in STOPWORDS_RO]

    # Tehnica 6: stemming in limba romana (exceptand tokenii speciali)
    if aplica_stemming:
        tokeni = [
            tok if tok.startswith("<") else STEMMER_RO.stem(tok)
            for tok in tokeni
        ]

    return " ".join(tokeni)


# 3. Extractor de trasaturi domeniu ADAS (contributie originala Model 1)
class AdasDomainFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        feats = []
        for raw_text in X:
            t = elimina_diacritice(raw_text).lower()
            has_qmark = 1.0 if "?" in raw_text else 0.0
            starts_interrog = 1.0 if re.match(
                r"^\s*(ce|cum|de ce|cand|unde|care|este|pot|functioneaza|la ce)\b", t
            ) else 0.0
            has_imperative = 1.0 if re.search(
                r"\b(activeaza|dezactiveaza|porneste|opreste|seteaza|regleaza|creste|scade|mareste|modifica)\b", t
            ) else 0.0
            has_comfort_adj = 1.0 if re.search(
                r"\b(prea|cam|brusc|enervant|deranjant|confortabil|lin|fluid|impecabil|fals|tarziu|devreme|obositor|speriat)\b", t
            ) else 0.0
            has_number = 1.0 if re.search(r"\d+", t) else 0.0
            word_count = len(t.split()) / 15.0
            feats.append([
                has_qmark,
                starts_interrog,
                has_imperative,
                has_comfort_adj,
                has_number,
                word_count,
            ])
        return csr_matrix(np.array(feats, dtype=np.float32))


# 4. Deduplicare si impartire stratificata Train / Val / Test (70% / 15% / 15%)
df = pd.DataFrame(DATE_BRUTE, columns=["text_brut", "eticheta"])
df["text_norm"] = df["text_brut"].apply(normalizeaza_text_adas)
df = df.drop_duplicates(subset=["text_norm"]).reset_index(drop=True)

X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
    df["text_brut"].tolist(),
    df["eticheta"].tolist(),
    test_size=0.30,
    random_state=SEED,
    stratify=df["eticheta"].tolist(),
)

X_val_raw, X_test_raw, y_val, y_test = train_test_split(
    X_temp_raw,
    y_temp,
    test_size=0.50,
    random_state=SEED,
    stratify=y_temp,
)

print(f"Total exemple deduplicate: {len(df)} | Train: {len(X_train_raw)} | Val: {len(X_val_raw)} | Test: {len(X_test_raw)}")
print("Distributie clase:\n", df["eticheta"].value_counts(), "\n")


def construieste_pipeline_m1(
    C: float = 1.0,
    use_domain_features: bool = True,
    use_char_ngrams: bool = True,
    expand_abrevieri: bool = True,
    aplica_stemming: bool = True,
) -> Pipeline:
    prep = lambda txt: normalizeaza_text_adas(
        txt, expand_abrevieri=expand_abrevieri, aplica_stemming=aplica_stemming
    )
    branches = [
        ("word_tfidf", TfidfVectorizer(preprocessor=prep, ngram_range=(1, 2), sublinear_tf=True)),
    ]
    if use_char_ngrams:
        branches.append(
            ("char_tfidf", TfidfVectorizer(preprocessor=prep, analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True))
        )
    if use_domain_features:
        branches.append(("adas_domain", AdasDomainFeatures()))

    return Pipeline([
        ("features", FeatureUnion(branches)),
        ("clf", LogisticRegression(C=C, class_weight="balanced", max_iter=1000, random_state=SEED)),
    ])


# 5. Studiu de ablatie Preprocesare & Contributie Model 1 (pe setul de VALIDARE)
ablatie_rows = []
configuratii_ablatie = [
    ("M1 Baseline (doar word TF-IDF, fara stemming/abrevieri)", dict(C=1.0, use_domain_features=False, use_char_ngrams=False, expand_abrevieri=False, aplica_stemming=False)),
    ("M1 + Preprocesare ADAS (abrevieri + stemming RO)", dict(C=1.0, use_domain_features=False, use_char_ngrams=False, expand_abrevieri=True, aplica_stemming=True)),
    ("M1 + Char n-grams (3,5) pentru flexiuni RO", dict(C=1.0, use_domain_features=False, use_char_ngrams=True, expand_abrevieri=True, aplica_stemming=True)),
    ("M1 Complet (+ Trasaturi domeniu ADAS)", dict(C=1.0, use_domain_features=True, use_char_ngrams=True, expand_abrevieri=True, aplica_stemming=True)),
]

for nume_cfg, params in configuratii_ablatie:
    pipe = construieste_pipeline_m1(**params)
    pipe.fit(X_train_raw, y_train)
    pred_v = pipe.predict(X_val_raw)
    f1_v = f1_score(y_val, pred_v, average="macro")
    acc_v = accuracy_score(y_val, pred_v)
    ablatie_rows.append({"Configuratie": nume_cfg, "Val_Accuracy": round(acc_v, 4), "Val_F1_macro": round(f1_v, 4)})

df_ablatie = pd.DataFrame(ablatie_rows)
df_ablatie.to_csv(os.path.join(RESULTS_DIR, "ablatie_model1.csv"), index=False)
print("=== STUDIU DE ABLATIE PE VALIDARE (MODEL 1) ===")
print(df_ablatie.to_string(index=False), "\n")


# 6. Optimizare hiperparametri (>= 3 configuratii pe model, pe setul de VALIDARE)
hp_rows = []

# Hiperparametri Model 1 (LogisticRegression + ADAS Features)
best_m1_score = -1.0
best_m1_C = 1.0
for C_val in [0.25, 1.0, 4.0, 10.0]:
    t0 = time.perf_counter()
    m1 = construieste_pipeline_m1(C=C_val)
    m1.fit(X_train_raw, y_train)
    dt = time.perf_counter() - t0
    f1_v = f1_score(y_val, m1.predict(X_val_raw), average="macro")
    hp_rows.append({"Model": "Model 1 (LogReg + ADAS)", "Config": f"C={C_val}", "Val_F1_macro": round(f1_v, 4), "Timp_s": round(dt, 4)})
    if f1_v > best_m1_score:
        best_m1_score = f1_v
        best_m1_C = C_val


def construieste_pipeline_m2(hidden_dim: int = 64, alpha: float = 1e-3, seed: int = 42, use_domain: bool = True) -> Pipeline:
    prep = lambda txt: normalizeaza_text_adas(txt, expand_abrevieri=True, aplica_stemming=True)
    branches = [
        ("word_tfidf", TfidfVectorizer(preprocessor=prep, ngram_range=(1, 2), sublinear_tf=True)),
        ("char_tfidf", TfidfVectorizer(preprocessor=prep, analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)),
    ]
    if use_domain:
        branches.append(("adas_domain", AdasDomainFeatures()))

    return Pipeline([
        ("features", FeatureUnion(branches)),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(hidden_dim,),
            activation="relu",
            alpha=alpha,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=15,
            random_state=seed,
        )),
    ])


# Hiperparametri Model 2 (Neural MLP cu Early Stopping, mediat pe 3 seed-uri)
best_m2_score = -1.0
best_m2_cfg = (64, 1e-3)
for hidden_dim, alpha_val in [(32, 1e-3), (64, 1e-3), (128, 1e-2)]:
    seed_f1s = []
    t0 = time.perf_counter()
    for s in [42, 123, 2026]:
        m2 = construieste_pipeline_m2(hidden_dim=hidden_dim, alpha=alpha_val, seed=s)
        m2.fit(X_train_raw, y_train)
        seed_f1s.append(f1_score(y_val, m2.predict(X_val_raw), average="macro"))
    dt = (time.perf_counter() - t0) / 3.0
    mean_f1 = float(np.mean(seed_f1s))
    std_f1 = float(np.std(seed_f1s))
    hp_rows.append({
        "Model": "Model 2 (Neural MLP + ES)",
        "Config": f"hidden={hidden_dim}, alpha={alpha_val}",
        "Val_F1_macro": f"{mean_f1:.4f} ± {std_f1:.4f}",
        "Timp_s": round(dt, 4),
    })
    if mean_f1 > best_m2_score:
        best_m2_score = mean_f1
        best_m2_cfg = (hidden_dim, alpha_val)

df_hp = pd.DataFrame(hp_rows)
df_hp.to_csv(os.path.join(RESULTS_DIR, "hiperparametri_validare.csv"), index=False)
print("=== OPTIMIZARE HIPERPARAMETRI PE VALIDARE ===")
print(df_hp.to_string(index=False), "\n")


# 7. Evaluare finala pe setul de TEST (o singura data!)
CLASE = ["COMANDA_ADAS", "INTREBARE_SISTEM", "OBSERVATIE_CONFORT"]

final_m1 = construieste_pipeline_m1(C=best_m1_C)
final_m1.fit(X_train_raw, y_train)
pred_test_m1 = final_m1.predict(X_test_raw)

m2_test_f1s = []
for s in [42, 123, 2026]:
    m2_s = construieste_pipeline_m2(hidden_dim=best_m2_cfg[0], alpha=best_m2_cfg[1], seed=s)
    m2_s.fit(X_train_raw, y_train)
    m2_test_f1s.append(f1_score(y_test, m2_s.predict(X_test_raw), average="macro"))

final_m2 = construieste_pipeline_m2(hidden_dim=best_m2_cfg[0], alpha=best_m2_cfg[1], seed=SEED)
final_m2.fit(X_train_raw, y_train)
pred_test_m2 = final_m2.predict(X_test_raw)

rezultate_test = pd.DataFrame([
    {
        "Model": f"Model 1: LogReg + ADAS Features (C={best_m1_C})",
        "Test_Accuracy": round(accuracy_score(y_test, pred_test_m1), 4),
        "Test_F1_macro": f"{f1_score(y_test, pred_test_m1, average='macro'):.4f}",
        "Recall_COMANDA_ADAS": round(recall_score(y_test, pred_test_m1, labels=CLASE, average=None)[0], 4),
    },
    {
        "Model": f"Model 2: Neural MLP + EarlyStopping (h={best_m2_cfg[0]})",
        "Test_Accuracy": round(accuracy_score(y_test, pred_test_m2), 4),
        "Test_F1_macro": f"{np.mean(m2_test_f1s):.4f} ± {np.std(m2_test_f1s):.4f}",
        "Recall_COMANDA_ADAS": round(recall_score(y_test, pred_test_m2, labels=CLASE, average=None)[0], 4),
    },
])
rezultate_test.to_csv(os.path.join(RESULTS_DIR, "rezultate_finale_test.csv"), index=False)

print("=== REZULTATE FINALE PE SETUL DE TEST ===")
print(rezultate_test.to_string(index=False), "\n")
print("Raport detaliat Model 1 (Test):")
print(classification_report(y_test, pred_test_m1, labels=CLASE, zero_division=0))

# 8. Generare grafice obligatorii in results/ (curbe invatare MLP + matrice confuzie)
cm = confusion_matrix(y_test, pred_test_m1, labels=CLASE)
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(CLASE)))
ax.set_yticks(range(len(CLASE)))
ax.set_xticklabels(CLASE, rotation=20, ha="right")
ax.set_yticklabels(CLASE)
ax.set_xlabel("Eticheta Prezisa")
ax.set_ylabel("Eticheta Reala")
ax.set_title("Matrice de Confuzie pe Test (Model 1)")
for i in range(len(CLASE)):
    for j in range(len(CLASE)):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "matrice_confuzie_test.png"), dpi=150)
plt.close()

mlp_est = final_m2.named_steps["mlp"]
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(mlp_est.loss_curve_, label="Train Loss")
if hasattr(mlp_est, "validation_scores_") and mlp_est.validation_scores_:
    ax2 = ax.twinx()
    ax2.plot(mlp_est.validation_scores_, color="green", linestyle="--", label="Val Score (Early Stopping)")
    ax2.set_ylabel("Validation Score")
ax.set_xlabel("Epoca")
ax.set_ylabel("Train Loss")
ax.set_title("Curba de Invatare Model 2 (MLP + Early Stopping)")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "curba_invatare_mlp.png"), dpi=150)
plt.close()

# 9. Analiza erorilor si test pe cazuri pragmatice dificile
mesaje_pragmatice = [
    "Oprește asistența de menținere a benzii acum.",
    "De ce nu mai merge recunoașterea indicatoarelor rutiere?",
    "Frânează cam târziu când încetinește coloana.",
    "Poți să pornești ACC-ul la 120 km/h?",
    "Nu mai ține banda cum trebuie pe ploaie.",
]

print("=== ANALIZA CAZURI PRAGMATICE / DIFICILE ===")
for msg in mesaje_pragmatice:
    p1 = final_m1.predict([msg])[0]
    p2 = final_m2.predict([msg])[0]
    print(f"Mesaj: {msg}")
    print(f"  -> Model 1: {p1} | Model 2: {p2}\n")

print(f"Fisierele rezultate au fost salvate in: {RESULTS_DIR}")