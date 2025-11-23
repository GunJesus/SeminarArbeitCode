from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import json

# ---------------- Konfiguration ----------------
model_name = "Qwen/Qwen3-0.6B"
LOG_PATH = "chat_log.jsonl"

# ---------------- Modell laden (unverändert) ----------------
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto"
)

# ---------------- Chat-State ----------------
history = []  # Liste von Tupeln (prompt, content) für Kontext
THINK_END_TOKEN_ID = 151668

def make_messages(history, user_prompt):
    """Baut die Nachrichtenliste für die Chat-Template aus Verlauf + aktuellem Prompt."""
    messages = []
    for p, a in history:
        messages.append({"role": "user", "content": p})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": user_prompt})
    return messages

def apply_template(messages):
    """Wendet die Chat-Template an; beachtet enable_thinking=True wie in deinem Skript."""
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True  # Switches between thinking and non-thinking modes. Default is True.
        )
    except TypeError:
        # Fallback für Umgebungen, deren Template den Parameter (noch) nicht kennt
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    return text

def split_thinking_and_content(output_ids):
    """
    Übernimmt deine Logik: versucht den letzten </think>-Marker (ID 151668) zu finden.
    Wenn nicht gefunden, geht alles in 'content'.
    """
    try:
        # rindex finding 151668 (</think>)  <-- beibehalten
        index = len(output_ids) - output_ids[::-1].index(THINK_END_TOKEN_ID)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    return thinking_content, content

def log_turn(prompt_text, answer_text, path=LOG_PATH):
    """
    Schreibt eine einzelne JSON-Zeile in die Logdatei:
    {"PROMPT": "...", "ANTWORT": "..."}
    """
    entry = {"PROMPT": prompt_text, "ANTWORT": answer_text}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# ---------------- Interaktive Schleife ----------------
print("Interaktiver Modus gestartet.")
print("Befehle: /exit (Beenden), /reset (Kontext löschen), /history (Verlauf anzeigen), /help\n")

turn = 1
while True:
    try:
        prompt = input(f"[{turn}] Du > ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nBeende…")
        break

    if not prompt:
        continue

    cmd = prompt.lower()
    if cmd in {"/exit", "exit", "quit", ":q"}:
        print("Auf Wiedersehen.")
        break
    if cmd in {"/reset", "/new"}:
        history.clear()
        print("Kontext zurückgesetzt.")
        continue
    if cmd in {"/history"}:
        if not history:
            print("(Noch kein Verlauf.)")
        else:
            for i, (p, a) in enumerate(history, 1):
                print(f"{i:02d}. PROMPT : {p}\n    ANTWORT: {a}\n")
        continue
    if cmd in {"/help", "/h", "?"}:
        print("Befehle:\n  /reset  Kontext/Verlauf löschen\n  /history Verlauf anzeigen\n  /exit   Beenden\n")
        continue

    # ---- Kontext + Template anwenden (deine Parameter bleiben erhalten) ----
    messages = make_messages(history, prompt)
    text = apply_template(messages)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # ---- Generierung (wie in deinem Skript) ----
    start_time = time.time()  # <-- Start Messung (pro Turn)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768  # <-- beibehalten aus deinem Skript
    )
    elapsed = time.time() - start_time

    # ---- Nur die neu generierten IDs extrahieren (wie bei dir) ----
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    # ---- Thinking/Content split (wie bei dir, inkl. Kommentar) ----
    thinking_content, content = split_thinking_and_content(output_ids)

    # ---- Ausgabe ----
    print("thinking content:", thinking_content if thinking_content else "(kein Marker gefunden oder leer)")
    print("content:", content)
    print(f"[Antwortzeit: {elapsed:.2f} Sekunden]\n")

    # ---- Verlauf für Chat/Nachfragen aktualisieren ----
    history.append((prompt, content))

    # ---- Logging in separate Textdatei als JSON-Zeile ----
    try:
        log_turn(prompt, content)
    except Exception as e:
        print(f"[Warnung] Konnte nicht in Logdatei schreiben: {e}")

    turn += 1
