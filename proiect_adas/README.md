# Proiect NLP în domeniul ADAS

Mini-proiect didactic realizat în cadrul disciplinei *Semantica și pragmatica limbajului natural*, axat pe înțelegerea mesajelor și comenzilor șoferului în limba română.

## Structura modulului
- `01_clasificare_intentii.py` — preprocesare text în română (`re.findall` + `stopwords` din `nltk`), vectorizare `TF-IDF` (`ngram_range=(1, 2)`) și clasificare cu `LogisticRegression` pe categoriile `COMANDA_ADAS`, `INTREBARE_SISTEM` și `OBSERVATIE_CONFORT`.
- `02_extragere_parametri.py` — extragerea structurată a acțiunii (`ACTIVARE`, `DEZACTIVARE`, `SETARE`), a funcției ADAS vizate și a eventualelor valori numerice (ex.: `120 km/h`, `2 secunde`).

## Rulare
Din rădăcina depozitului:

```powershell
py -3.12 proiect_adas/01_clasificare_intentii.py
py -3.12 proiect_adas/02_extragere_parametri.py