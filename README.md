# Semantica și pragmatica limbajului natural (NLP)

Depozit cu lucrări practice de laborator și proiect didactic realizate în cadrul programului de master **Inteligență Artificială** (Anul 1, Modulul 2).

## Structura depozitului

- **`lab1/preprocesare.py`** — curățare și tokenizare de text în limba română (`re`, `nltk` stopwords) și reprezentare vectorială `TF-IDF`.
- **`lab2/clasificare_spam.py`** — clasificare binară de mesaje (`SPAM` vs. `LEGITIM`) folosind un `Pipeline` cu `TfidfVectorizer` și `MultinomialNB`.
- **`lab3/evaluare_ngrame.py`** — împărțire *train/test*, comparație între unigrame `(1, 1)` și unigrame + bigrame `(1, 2)`, precum și evaluarea clasificatorilor `MultinomialNB` și `LogisticRegression` (acuratețe, matrice de confuzie, raport de clasificare).
- **`proiect_adas/`** — mini-proiect aplicat în domeniul **ADAS** (*Advanced Driver-Assistance Systems*):
  - `01_clasificare_intentii.py` — identificarea intenției șoferului (`COMANDA_ADAS`, `INTREBARE_SISTEM`, `OBSERVATIE_CONFORT`).
  - `02_extragere_parametri.py` — extragerea acțiunii, a funcției ADAS vizate și a valorilor numerice din comenzi.
  - `README.md` — documentația dedicată proiectului ADAS.

## Cerințe și rulare

Mediul local folosește **Python 3.12** și bibliotecile listate în `requirements.txt` (`numpy`, `pandas`, `scikit-learn`, `nltk`).

```powershell
py -3.12 -m pip install -r requirements.txt
py -3.12 lab1/preprocesare.py
py -3.12 lab2/clasificare_spam.py
py -3.12 lab3/evaluare_ngrame.py
py -3.12 proiect_adas/01_clasificare_intentii.py
py -3.12 proiect_adas/02_extragere_parametri.py