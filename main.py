#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from transformers import AutoModelForCausalLM, AutoTokenizer
import csv, json, time
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# ========= CONFIG =========
MODEL_NAME   = "Qwen/Qwen3-0.6B"
CSV_PATH     = "DataSets/dataset_random_120.csv"  # ; -getrennt
PROMPT_COL   = "prompt"
CHAT_ID_COL  = "chat_id"          # None => jede Zeile isoliert
SYSTEM_COL   = "system"           # None => keine Systemrolle
OUT_JSONL    = "batch_chat.jsonl" # maschinenlesbar
OUT_TXT      = "batch_chat.txt"   # menschenlesbar
ENCODING     = "utf-8-sig"
MAX_TOKENS   = 32768
CLEAR_LOGS   = True
THINK_END_ID = 151668             # </think>
# ==========================

def apply_template(tokenizer, messages):
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

def split_thinking_and_content(tokenizer, output_ids):
    try:
        idx = len(output_ids) - output_ids[::-1].index(THINK_END_ID)
    except ValueError:
        idx = 0
    thinking = tokenizer.decode(output_ids[:idx], skip_special_tokens=True).strip("\n")
    content  = tokenizer.decode(output_ids[idx:],  skip_special_tokens=True).strip("\n")
    return thinking, content

def log_json(entry, path):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def log_txt(text, path):
    with open(path, "a", encoding="utf-8") as f:
        f.write(text + "\n")

def main():
    csvp = Path(CSV_PATH)
    if not csvp.exists():
        raise FileNotFoundError(f"CSV nicht gefunden: {csvp.resolve()}")

    if CLEAR_LOGS:
        Path(OUT_JSONL).write_text("", encoding="utf-8")
        Path(OUT_TXT).write_text("",   encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype="auto", device_map="auto")

    histories = defaultdict(list)  # je chat_id: [(prompt, answer), ...]

    with open(csvp, "r", encoding=ENCODING, newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        if not reader.fieldnames or PROMPT_COL not in reader.fieldnames:
            raise ValueError(f"Spalte '{PROMPT_COL}' fehlt. Header: {reader.fieldnames}")

        for i, row in enumerate(reader, start=1):
            ts = datetime.now().isoformat(timespec="seconds")
            prompt = (row.get(PROMPT_COL) or "").strip()
            if not prompt:
                log_txt(f"[row {i}] ({ts}) Übersprungen: leeres Prompt", OUT_TXT)
                continue

            chat_id = (row.get(CHAT_ID_COL) or f"row-{i}") if CHAT_ID_COL else f"row-{i}"
            system_msg = (row.get(SYSTEM_COL) or "").strip() if SYSTEM_COL else ""

            msgs = []
            if system_msg:
                msgs.append({"role": "system", "content": system_msg})
            for p, a in histories[chat_id]:
                msgs += [{"role": "user", "content": p}, {"role": "assistant", "content": a}]
            msgs.append({"role": "user", "content": prompt})

            text = apply_template(tokenizer, msgs)
            inputs = tokenizer([text], return_tensors="pt").to(model.device)

            t0 = time.time()
            gen = model.generate(**inputs, max_new_tokens=MAX_TOKENS)
            dt = time.time() - t0

            new_ids = gen[0][len(inputs.input_ids[0]):].tolist()
            thinking, content = split_thinking_and_content(tokenizer, new_ids)

            histories[chat_id].append((prompt, content))

            log_json({"PROMPT": prompt, "ANTWORT": content, "chat_id": chat_id, "row": i, "ts": ts}, OUT_JSONL)
            block = [
                f"[row {i} | chat {chat_id} | {ts}]",
                f"SYSTEM : {system_msg}" if system_msg else None,
                f"PROMPT : {prompt}",
                f"thinking content: {thinking if thinking else '(leer)'}",
                f"content: {content}",
                f"[Antwortzeit: {dt:.2f} Sekunden]",
                "-" * 72
            ]
            log_txt("\n".join([b for b in block if b is not None]), OUT_TXT)

    log_txt("[Batch abgeschlossen]", OUT_TXT)
    print("Fertig. Logs geschrieben.")

if __name__ == "__main__":
    main()
