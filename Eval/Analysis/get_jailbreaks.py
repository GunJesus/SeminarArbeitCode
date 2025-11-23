import json

# --- Dateipfade ---
INPUT_FILE = "../round1_non_refusals.jsonl"  # oder round1_non_refusals.jsonl
OUTPUT_FILE = "../jailbreaks.jsonl"

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def save_jsonl(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for obj in data:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def main():
    data = load_jsonl(INPUT_FILE)

    # Filter: nur origin == "generated_cases.json"
    filtered = [item for item in data if item.get("origin") == "generated_cases.json"]

    save_jsonl(OUTPUT_FILE, filtered)
    print(f"✅ Neue Datei '{OUTPUT_FILE}' mit {len(filtered)} Einträgen erstellt.")

if __name__ == "__main__":
    main()
