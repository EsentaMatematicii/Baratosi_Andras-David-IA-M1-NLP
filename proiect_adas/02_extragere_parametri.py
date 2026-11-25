import re
import unicodedata


def normalizeaza_ro(text: str) -> str:
    t = (
        text.replace("ş", "ș")
        .replace("ţ", "ț")
        .replace("Ş", "Ș")
        .replace("Ţ", "Ț")
    )
    nfkd = unicodedata.normalize("NFKD", t)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()


def extrage_parametri_comanda(text: str) -> dict:
    text_norm = normalizeaza_ro(text)

    # 1. Determinam actiunea principala
    if re.search(r"\b(dezactiveaza|opreste|inchide|anuleaza)\b", text_norm):
        actiune = "DEZACTIVARE"
    elif re.search(r"\b(activeaza|porneste|deschide|cupleaza)\b", text_norm):
        actiune = "ACTIVARE"
    elif re.search(r"\b(seteaza|regleaza|mareste|scade|modifica|creste)\b", text_norm):
        actiune = "SETARE"
    else:
        actiune = "NECUNOSCUTA"

    # 2. Identificam functia ADAS vizata (inclusiv forme flexionare si abrevieri)
    if re.search(r"\b(pilot\w*|cruise|acc)\b", text_norm):
        functie = "pilot_automat_adaptiv"
    elif re.search(r"\b(banda|benzii|benzi|lka|lkas)\b", text_norm):
        functie = "mentinere_banda"
    elif re.search(r"\b(frana\w*|aeb|coliziun\w*)\b", text_norm):
        functie = "franare_urgenta"
    elif re.search(r"\b(distanta\w*|interval\w*)\b", text_norm):
        functie = "distanta_fata"
    elif re.search(r"\b(viteza\w*|limitator\w*)\b", text_norm):
        functie = "viteza_tinta"
    elif re.search(r"\b(unghi\w*\s+mort|bsm)\b", text_norm):
        functie = "unghi_mort"
    else:
        functie = "neidentificata"

    # 3. Extragem o eventuala valoare numerica + unitate (ex: 120 km/h, 2 secunde)
    valoare_match = re.search(r"\b(\d+)\s*(km/h|kmh|km|secunde|sec|s|m|%)\b", text_norm)
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


if __name__ == "__main__":
    comenzi_test = [
        "Activează pilotul automat adaptiv.",
        "Setează viteza la 120 km/h pe autostradă.",
        "Oprește asistența de menținere a benzii acum.",
        "Reglează distanța la 2 secunde față de mașina din față.",
        "Pornește ACC la 130 km/h.",
        "Dezactivează monitorizarea unghiului mort BSM.",
    ]

    print("=== EXTRAGERE PARAMETRI COMENZI ADAS ===")
    for comanda in comenzi_test:
        rezultat = extrage_parametri_comanda(comanda)
        print(f"Comanda: {comanda}")
        print(f"  -> {rezultat}\n")