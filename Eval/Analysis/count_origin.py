import json
from collections import Counter

# --- Dateipfade ---
FILE_REFUSALS = "../round1_refusals.jsonl"
FILE_NON_REFUSALS = "../round1_non_refusals.jsonl"

def load_jsonl(path):
    """Lädt eine JSONL-Datei als Liste von Dicts."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def count_origins(data):
    """Zählt, wie oft jeder 'origin'-Wert vorkommt."""
    origins = [entry.get("origin") for entry in data if "origin" in entry]
    return Counter(origins)

def main():
    refusals = load_jsonl(FILE_REFUSALS)
    non_refusals = load_jsonl(FILE_NON_REFUSALS)

    counter_refusals = count_origins(refusals)
    counter_non_refusals = count_origins(non_refusals)

    print("📊 Häufigkeiten in round1_refusals.jsonl:")
    for origin, count in counter_refusals.most_common():
        print(f"{origin}: {count}")

    print("\n📊 Häufigkeiten in round1_non_refusals.jsonl:")
    for origin, count in counter_non_refusals.most_common():
        print(f"{origin}: {count}")


main()
