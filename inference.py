# ============================================================
# inference.py
# OCR → Fine-tuned Qwen2.5-7B → Answer
#
# Usage:
#   python inference.py --test_dir <absolute_path_to_test_dir>
#
# Writes submission.csv in the SAME directory as this script.
#
# Citations:
#   - Qwen2.5: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
#   - pytesseract: https://github.com/madmaze/pytesseract
#   - transformers: https://github.com/huggingface/transformers
# ============================================================

import os
import re
import gc
import argparse
import torch
import pandas as pd
import cv2
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Args ──────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--test_dir", required=True, help="Absolute path to test directory")
args = parser.parse_args()

# ── Paths (all relative to THIS script's directory) ───────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.join(SCRIPT_DIR, "model_weights")        # downloaded by setup.bash
TEST_DIR    = args.test_dir
TEST_CSV    = os.path.join(TEST_DIR, "test.csv")
OUTPUT_CSV  = os.path.join(SCRIPT_DIR, "submission.csv")       # saved next to script

# Auto-detect images folder inside test_dir
IMAGE_DIR = None
for name in ["images", "image", "imgs"]:
    p = os.path.join(TEST_DIR, name)
    if os.path.isdir(p):
        IMAGE_DIR = p
        break

# ── Validations ───────────────────────────────────────────────
assert os.path.exists(TEST_CSV),   f"test.csv not found at: {TEST_CSV}"
assert IMAGE_DIR is not None,      f"No images/ folder found in: {TEST_DIR}"
assert os.path.exists(MODEL_DIR),  f"Model weights missing at: {MODEL_DIR} — was setup.bash run?"

# ── Offline mode (no internet during inference) ───────────────
os.environ["TRANSFORMERS_OFFLINE"]    = "1"
os.environ["HF_DATASETS_OFFLINE"]     = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device    : {DEVICE}")
print(f"TEST_DIR  : {TEST_DIR}")
print(f"IMAGE_DIR : {IMAGE_DIR}")
print(f"MODEL_DIR : {MODEL_DIR}")
print(f"OUTPUT    : {OUTPUT_CSV}")

# ── Load Model ────────────────────────────────────────────────
print(f"\nLoading model from {MODEL_DIR} ...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token

total_vram = (
    sum(torch.cuda.get_device_properties(i).total_memory
        for i in range(torch.cuda.device_count())) / 1e9
    if DEVICE == "cuda" else 0
)
print(f"Total VRAM: {total_vram:.1f} GB")

if total_vram >= 40:
    # L40s — load in bfloat16 full precision
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype       = torch.bfloat16,
        device_map        = "auto",
        local_files_only  = True,
        trust_remote_code = True,
    )
else:
    # T4/P100 — load in 4-bit
    from transformers import BitsAndBytesConfig
    bnb = BitsAndBytesConfig(
        load_in_4bit           = True,
        bnb_4bit_quant_type    = "nf4",
        bnb_4bit_compute_dtype = torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        quantization_config = bnb,
        device_map          = "auto",
        local_files_only    = True,
        trust_remote_code   = True,
    )

model.eval()
print("Model loaded.\n")

# ── OCR ───────────────────────────────────────────────────────
def extract_text_ocr(image_path: str) -> str:
    """
    Preprocess image and extract text using pytesseract.
    Upscales 2x for better accuracy on small text.
    """
    img = cv2.imread(image_path)
    if img is None:
        return ""

    # Upscale 2x (helps with small fonts)
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Enhance contrast
    pil_img = Image.fromarray(gray)
    pil_img = ImageEnhance.Contrast(pil_img).enhance(2.0)
    pil_img = pil_img.filter(ImageFilter.SHARPEN)

    # OCR with layout-aware config
    text = pytesseract.image_to_string(
        pil_img,
        config="--oem 3 --psm 6"
    )
    return text.strip()

# ── Build Prompt ──────────────────────────────────────────────
def build_prompt(ocr_text: str) -> str:
    return (
        "<|im_start|>user\n"
        "You are a deep learning expert. Below is OCR-extracted text from an MCQ image.\n"
        "Read the question and options carefully, then output ONLY the number (1, 2, 3, or 4) "
        "of the correct option. If you are completely unsure, output 5.\n"
        "Do not explain. Do not add any other text. Just the number.\n\n"
        f"{ocr_text}\n\n"
        "Answer:"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

# ── Parse Answer ──────────────────────────────────────────────
def parse_answer(text: str, image_name: str = "") -> int:
    text = text.strip()

    # Direct digit at start
    m = re.match(r"^([1-5])", text)
    if m:
        return int(m.group(1))

    # "answer is X" pattern
    m = re.search(r"(?:answer|correct)[^\d]*([1-5])", text, re.IGNORECASE)
    if m:
        return int(m.group(1))

    # Letter mapping A→1 B→2 C→3 D→4
    m = re.search(r"\b([A-Da-d])\b", text)
    if m:
        return {"A":1,"B":2,"C":3,"D":4}.get(m.group(1).upper(), 5)

    print(f"  WARNING [{image_name}]: parse failed → 5 | raw: {text[:100]}")
    return 5

# ── Inference ─────────────────────────────────────────────────
def predict(image_path: str, image_name: str = "") -> int:
    ocr_text = extract_text_ocr(image_path)
    print(f"  OCR ({len(ocr_text)} chars): {ocr_text[:200].replace(chr(10),' ')}")

    prompt = build_prompt(ocr_text)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)

    try:
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens = 5,       # only need a single digit
                do_sample      = False,   # greedy = deterministic
                pad_token_id   = tokenizer.eos_token_id,
            )
        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        raw_text  = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
        print(f"  Model output: '{raw_text}'")
        return parse_answer(raw_text, image_name)

    except torch.cuda.OutOfMemoryError:
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  OOM on {image_name} → 5")
        return 5
    except Exception as e:
        print(f"  ERROR on {image_name}: {e} → 5")
        return 5
    finally:
        del inputs
        gc.collect()
        torch.cuda.empty_cache()

# ── Main Loop ─────────────────────────────────────────────────
def find_image(image_dir: str, name: str) -> str | None:
    for ext in ["", ".png", ".jpg", ".jpeg", ".PNG", ".JPG"]:
        p = os.path.join(image_dir, f"{name}{ext}")
        if os.path.exists(p):
            return p
    return None

test_df   = pd.read_csv(TEST_CSV)
image_col = next(
    (c for c in ["image_name", "image_id"] if c in test_df.columns),
    test_df.columns[0]
)
print(f"Total questions : {len(test_df)}")
print(f"Image column    : '{image_col}'\n")

results = []

for idx, row in test_df.iterrows():
    image_name = str(row[image_col])
    image_path = find_image(IMAGE_DIR, image_name)

    print(f"[{idx+1}/{len(test_df)}] {image_name}")

    if image_path is None:
        print(f"  Image not found → 5")
        results.append({"id": image_name, "image_name": image_name, "option": 5})
        continue

    answer = predict(image_path, image_name)
    print(f"  → ANSWER: {answer}\n")
    results.append({"id": image_name, "image_name": image_name, "option": answer})

# ── Save submission.csv ───────────────────────────────────────
submission_df = pd.DataFrame(results)
submission_df["option"] = submission_df["option"].apply(
    lambda x: int(x) if int(x) in {1,2,3,4,5} else 5
)
submission_df = submission_df[["id", "image_name", "option"]]
submission_df.to_csv(OUTPUT_CSV, index=False)

print(f"\nSaved → {OUTPUT_CSV}")
print(submission_df["option"].value_counts().sort_index().to_string())
print(f"\nAttempted : {len(submission_df[submission_df['option'] != 5])}")
print(f"Skipped   : {len(submission_df[submission_df['option'] == 5])}")
