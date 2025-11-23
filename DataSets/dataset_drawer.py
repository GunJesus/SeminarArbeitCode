#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import csv
import json
import random

# -------------------------------
# Pfadangaben (lokal/IDE)
# -------------------------------
JSON_PATH = Path("Inputs/generated_cases.json")
CSV_PATH  = Path("Inputs/harmful_behaviors.csv")
OUT_PATH  = Path("dataset_random_120.csv")

SAMPLE_SIZE = 60  # Prompts je Quelle

# -------------------------------
# Hilfsfunktionen
# -------------------------------

def _normalize_text(s: str) -> str:
    """Entfernt Zeilenumbrüche und ersetzt Semikolons."""
    return s.replace("\n", " ").replace("\r", " ").replace(";", ",")

def load_json_prompts(path: Path, source: str, sample_size: int = 60, seed: int = 42):
    """Zieht bis zu sample_size zufällige Prompts aus der JSON-Datei."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # JSON kann Liste oder Dict enthalten
    if isinstance(data, dict):
        for key in ("data", "items", "cases", "entries"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break

    if not isinstance(data, list):
        print(f"[WARN] Ungültiges JSON-Format: {path.name}")
        return []

    prompts = []
    for entry in data:
        if isinstance(entry, dict):
            prompt = entry.get("Attack Prompt") or entry.get("prompt") or ""
        elif isinstance(entry, str):
            prompt = entry
        else:
            prompt = ""

        if prompt.strip():
            prompts.append({"prompt": _normalize_text(prompt.strip()), "origin": source})

    random.seed(seed)
    return random.sample(prompts, min(sample_size, len(prompts)))


def load_csv_prompts(path: Path, source: str, sample_size: int = 60, seed: int = 43):
    """Zieht bis zu sample_size zufällige Prompts aus einer CSV-Datei, Spalte 'goal'."""
    records = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "goal" not in reader.fieldnames:
            print(f"[WARN] Spalte 'goal' nicht gefunden in {path.name}. Verfügbare Spalten: {reader.fieldnames}")
            return []
        for row in reader:
            prompt = row.get("goal") or ""
            if prompt.strip():
                records.append({"prompt": _normalize_text(prompt.strip()), "origin": source})

    random.seed(seed)
    return random.sample(records, min(sample_size, len(records)))


def write_csv(out_path: Path, records):
    """Schreibt kombinierte CSV mit Header: prompt; origin"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["prompt", "origin"])
        for r in records:
            writer.writerow([r["prompt"], r["origin"]])
    print(f"[OK] CSV erstellt: {out_path.resolve()} ({len(records)} Einträge)")


# -------------------------------
# Verarbeitung
# -------------------------------
json_sample = load_json_prompts(JSON_PATH, "generated_cases.json", SAMPLE_SIZE)
csv_sample  = load_csv_prompts(CSV_PATH,  "harmful_behaviors.csv", SAMPLE_SIZE)

combined = json_sample + csv_sample
write_csv(OUT_PATH, combined)
