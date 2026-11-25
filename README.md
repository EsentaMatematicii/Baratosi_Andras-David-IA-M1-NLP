# Semantica și pragmatica limbajului natural (NLP)

Depozit cu lucrări practice de laborator și proiect aplicat în domeniul **ADAS** (*Advanced Driver-Assistance Systems*) realizate în cadrul programului de master **Inteligență Artificială** (Anul 1, Modulul 2).

## Structura depozitului

- **`lab1/preprocesare.py`** — normalizare diacritice românești, păstrarea negațiilor, *stemming* `SnowballStemmer("romanian")` și reprezentări `BoW` / `TF-IDF`.
- **`lab2/clasificare_spam.py`** — clasificare binară de mesaje (`SPAM` vs. `LEGITIM`) folosind un `Pipeline` cu `TfidfVectorizer` (uni+bigrame) și `MultinomialNB`.
- **`lab3/evaluare_ngrame.py`** — împărțire stratificată *train/test*, comparație între unigrame `(1, 1)` și unigrame + bigrame `(1, 2)`, precum și evaluarea clasificatorilor `MultinomialNB` și `LogisticRegression` (`Accuracy`, `F1-macro`, matrice de confuzie, raport de clasificare).
- **`proiect_adas/`** — proiect aplicat în domeniul **ADAS** în limba română:
  - `01_clasificare_intentii.py` — preprocesare ADAS în 6 etape, deduplicare și împărțire stratificată Train/Val/Test, două modele distincte (`LogisticRegression` cu `TF-IDF` cuvinte + caractere + trăsături de domeniu ADAS vs. `MLPClassifier` cu `early_stopping` evaluat pe mai multe *seed*-uri), studiu de ablație, optimizarea hiperparametrilor și exportul rezultatelor în `proiect_adas/results/`.
  - `02_extragere_parametri.py` — extragerea structurată a acțiunii, a funcției ADAS vizate (inclusiv forme flexionare și abrevieri) și a valorilor numerice din comenzi.
  - `results/` — tabele CSV (`ablatie_model1.csv`, `hiperparametri_validare.csv`, `rezultate_finale_test.csv`) și grafice PNG (`matrice_confuzie_test.png`, `curba_invatare_mlp.png`).
  - `README.md` — documentația dedicată modulului ADAS.

## Cerințe și rulare

Mediul local folosește **Python 3.12** și bibliotecile listate în `requirements.txt` (`numpy`, `pandas`, `scikit-learn`, `nltk`, `matplotlib`).

```powershell
py -3.12 -m pip install -r requirements.txt
py -3.12 lab1/preprocesare.py
py -3.12 lab2/clasificare_spam.py
py -3.12 lab3/evaluare_ngrame.py
py -3.12 proiect_adas/01_clasificare_intentii.py
py -3.12 proiect_adas/02_extragere_parametri.py

## Declarație privind folosirea asistenților AI

În realizarea acestui depozit au fost utilizați asistenți AI (Google Gemini) pentru structurarea codului didactic, rafinarea regulilor de preprocesare pentru limba română, verificarea conformității cu cerințele disciplinei și redactarea documentației. Întregul cod a fost verificat, rulat local și înțeles integral de autor.