import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

from training.farsi.normalize_fa import normalize_for_model

MODEL_DIR = r".\Models\Homo-GE2PE-Persian-HF"

# Persian text to test
text = "سلام، من مهدی هستم."

print("Loading local Persian G2P model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)

model = T5ForConditionalGeneration.from_pretrained(
    MODEL_DIR,
    local_files_only=True
).eval()

# Normalise FIRST (spells digits out, unifies Arabic/Persian letter variants).
text = normalize_for_model(text)

# The question mark is ambiguous in the model's notation.
text = text.replace("؟", "").replace("?", "")

inputs = tokenizer(
    [text],
    add_special_tokens=False,
    padding=True,
    return_attention_mask=True,
    return_tensors="pt"
)

with torch.no_grad():
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        num_beams=5,
        min_length=1,
        max_length=512,
        early_stopping=True
    )

raw = tokenizer.batch_decode(
    outputs,
    skip_special_tokens=True
)[0].strip()

# Convert Homo-GE2PE notation to Pocket TTS notation.
translation = str.maketrans({
    "/": "a",
    "a": "A",
    "@": "?",
    "$": "S",
    "c": "C",
})

# KEEP "1": it marks the ezafe ("?eqtesAde1 ?AmrikA" is eqtesad-E amrika,
# one bound phrase). synthesize.py strips it before the model sees the text.
phonemes = raw.translate(translation)

print("\nOriginal Persian:")
print(" ", text)

print("\nRaw G2P output:")
print(" ", raw)

print("\nPocket TTS phonemes:")
print(" ", phonemes)

with open("phonemes.txt", "w", encoding="utf-8") as f:
    f.write(phonemes)

print("\nSaved to phonemes.txt")
