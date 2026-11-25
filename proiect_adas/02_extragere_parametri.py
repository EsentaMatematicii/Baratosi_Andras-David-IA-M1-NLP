import re


def extrage_parametri_comanda(text):
    text_mic = text.lower()

    # 1. Determinam actiunea principala
    if re.search(r"\b(dezactiveaza|opreste|inchide)\b", text_mic):
        actiune = "DEZACTIVARE"
    elif re.search(r"\b(activeaza|porneste|deschide)\b", text_mic):
        actiune = "ACTIVARE"
    elif re.search(r"\b(seteaza|regleaza|mareste|scade|modifica)\b", text_mic):
        actiune = "SETARE"
    else:
        actiune = "NECUNOSCUTA"

    # 2. Identificam functia ADAS vizata
    if "pilot" in text_mic or "cruise" in text_mic:
        functie = "pilot_automat_adaptiv"
    elif "banda" in text_mic:
        functie = "mentinere_banda"
    elif "frana" in text_mic or "franare" in text_mic:
        functie = "franare_urgenta"
    elif "distanta" in text_mic:
        functie = "distanta_fata"
    elif "viteza" in text_mic:
        functie = "viteza_tinta"
    else:
        functie = "neidentificata"

    # 3. Extragem o eventuala valoare numerica + unitate (ex: 120 km/h, 2 secunde)
    valoare_match = re.search(r"\b(\d+)\s*(km/h|km|secunde|sec|s|%)\b", text_mic)
    valoare = (
        f"{valoare_match.group(1)} {valoare_match.group(2)}"
        if valoare_match
        else None
    )

    return {
        "actiune": actiune,
        "functie": functie,
        "valoare": valoare,
    }


comenzi_test = [
    "Activeaza pilotul automat adaptiv.",
    "Seteaza viteza la 120 km/h pe autostrada.",
    "Opreste asistenta de mentinere a benzii acum.",
    "Regleaza distanta la 2 secunde fata de masina din fata.",
]

print("=== EXTRAGERE PARAMETRI COMENZI ADAS ===")
for comanda in comenzi_test:
    rezultat = extrage_parametri_comanda(comanda)
    print(f"Comanda: {comanda}")
    print(f"  -> {rezultat}\n")