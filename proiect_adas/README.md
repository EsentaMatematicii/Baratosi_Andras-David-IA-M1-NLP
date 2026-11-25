# Proiect NLP în domeniul ADAS

Modul aplicat realizat în cadrul disciplinei *Semantica și pragmatica limbajului natural* (Master IA, Anul 1, Modulul 2), axat pe înțelegerea mesajelor și comenzilor șoferului în limba română (`COMANDA_ADAS`, `INTREBARE_SISTEM`, `OBSERVATIE_CONFORT`) și pe extragerea structurată a parametrilor[cite: 259, 260, 261, 262].

## Structura modulului

- `01_clasificare_intentii.py` — preprocesare în limba română pentru domeniul ADAS (normalizare diacritice, mascarea valorilor numerice, expandarea abrevierilor tehnice precum `ACC`, `LKA`, `AEB`, `BSM`, tokenizare, păstrarea negațiilor și *stemming*), deduplicare și împărțire Train/Val/Test, comparație între **Model 1** (`TF-IDF` cuvinte + caractere + trăsături domeniu ADAS + `LogisticRegression`) și **Model 2** (`MLPClassifier` cu `early_stopping` evaluat pe mai multe *seed*-uri), studiu de ablație, optimizarea hiperparametrilor și salvarea rezultatelor în `results/`[cite: 245, 260].
- `02_extragere_parametri.py` — extragerea structurată a acțiunii (`ACTIVARE`, `DEZACTIVARE`, `SETARE`, `NECUNOSCUTA`), a funcției ADAS vizate (`pilot_automat_adaptiv`, `mentinere_banda`, `franare_urgenta`, `distanta_fata`, `viteza_tinta`, `unghi_mort`) și a valorilor numerice însoțite de unitatea de măsură (ex.: `120 km/h`, `2 secunde`)[cite: 259, 261].
- `results/` — tabele CSV (`ablatie_model1.csv`, `hiperparametri_validare.csv`, `rezultate_finale_test.csv`) și grafice PNG (`matrice_confuzie_test.png`, `curba_invatare_mlp.png`) generate la rularea scriptului.

## Rulare

Din rădăcina depozitului[cite: 262]:

```powershell
py -3.12 proiect_adas/01_clasificare_intentii.py
py -3.12 proiect_adas/02_extragere_parametri.py

## Declarație privind folosirea asistenților AI

În realizarea acestui depozit au fost utilizați asistenți AI (Google Gemini) pentru structurarea codului didactic, rafinarea regulilor de preprocesare pentru limba română, verificarea conformității cu cerințele disciplinei și redactarea documentației. Întregul cod a fost verificat, rulat local și înțeles integral de autor.