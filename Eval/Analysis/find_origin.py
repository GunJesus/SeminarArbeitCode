import json
import csv

# --- Dateipfade ---
FILE_CSV = "../../DataSets/dataset_random_120.csv"
FILE_REFUSALS = "../round1_refusals.jsonl"
FILE_NON_REFUSALS = "../round1_non_refusals.jsonl"

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def save_jsonl(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def run():
    # JSON-Dateien laden
    refusals = load_jsonl(FILE_REFUSALS)
    non_refusals = load_jsonl(FILE_NON_REFUSALS)

    updated_count = 0

    # CSV öffnen und jede Zeile prüfen
    with open(FILE_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            csv_prompt = row["prompt"]
            csv_origin = row["origin"]

            # Suche in beiden JSON-Dateien nach passendem PROMPT
            for entry in refusals:
                if entry.get("PROMPT") == csv_prompt:
                    entry["origin"] = csv_origin
                    updated_count += 1
                    break  # gehe zur nächsten CSV-Zeile

            for entry in non_refusals:
                if entry.get("PROMPT") == csv_prompt:
                    entry["origin"] = csv_origin
                    updated_count += 1
                    break

    # Änderungen zurückschreiben
    save_jsonl(FILE_REFUSALS, refusals)
    save_jsonl(FILE_NON_REFUSALS, non_refusals)

    print(f"✅ Fertig! {updated_count} Einträge wurden um 'origin' ergänzt.")

run()