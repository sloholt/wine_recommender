"""
Step 3 of the wine recommender: turn a wine label into a clean JSON record using Claude.

For photos, Claude gets BOTH the image and the Cloud Vision OCR text. It reads the image
directly (catching curved seals and sideways text that OCR misses) and uses the OCR text
as a cross-check for spelling.

python extract_label.py label.jpg        # photo: OCR + image sent to Claude (needs read_label.py)
python extract_label.py label_text.txt   # text only, if you already have OCR output

Output: a JSON file next to the input, e.g. label.json

Setup:
    pip install anthropic pydantic pillow google-cloud-vision

API key (set in the terminal; never paste it into the code):
"""

import base64
import io
import json
import os
import sys
import re
import unicodedata
from pathlib import Path
from typing import Optional

import anthropic
from PIL import Image, ImageOps
from pydantic import BaseModel, Field, ValidationError

# ---- Settings -----------------------------------------------------------------
MODEL = os.environ.get("CLAUDE_MODEL", "claude-haiku-5-5")
IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
MAX_IMAGE_SIDE = 1568  # longest edge in pixels; larger photos are shrunk before sending


# ---- The Label Record Shape (aka JSON format) --------------------------------
class WineLabel(BaseModel):
    producer: Optional[str] = Field(
        None,
        description="Winery or company that makes the wine. Often in a seal, crest or small print, not the big name.",
    )
    wine_name: Optional[str] = Field(
        None,
        description="Name of this wine, e.g. 'Amelia'. Use the appellation if there is no other name.",
    )
    appellation: Optional[str] = Field(
        None,
        description="Official origin only, e.g. 'D.O. Valle del Limarí', 'Gavi DOCG'",
    )
    vineyard: Optional[str] = Field(
        None, description="Specific vineyard, estate or sub-area, e.g. 'Quebrada Seca'"
    )
    classification: Optional[str] = Field(
        None, description="e.g. DOCG, D.O., AOC, Grand Cru, Reserva"
    )
    country: Optional[str] = None
    region: Optional[str] = None
    variety: Optional[str] = Field(
        None, description="Grape variety or blend, e.g. 'Chardonnay'"
    )
    vintage: Optional[int] = Field(None, description="Harvest year, e.g. 2023")
    abv: Optional[float] = Field(
        None, description="Alcohol by volume in percent, e.g. 13.5"
    )
    volume_ml: Optional[int] = Field(None, description="Bottle size in ml, e.g. 750")
    inferred_fields: list[str] = Field(
        default_factory=list,
        description="Names of fields whose value is NOT printed anywhere on the label but was inferred",
    )
    ocr_corrections: list[str] = Field(
        default_factory=list,
        description="Only confident fixes of misread words, as 'wrong -> right'",
    )


INSTRUCTIONS = """You extract structured data from a wine label.
You may receive a photo of the label and/or OCR text from it.

How to use the inputs:
- Read the photo carefully, including small print, curved text in seals or crests, and text printed sideways along the label edges (bottle size and alcohol % are often there).
- Use the OCR text as a spelling cross-check. OCR often drops curved or rotated text and leaves fragments; trust the photo when they disagree.

Field rules:
- The large decorative name is usually the WINE NAME, not the producer. The producer is often in a seal, crest, logo or small print.
- Keep the official appellation separate from a vineyard or sub-area name (e.g. 'Quebrada Seca / D.O. Valle del Limarí' -> vineyard 'Quebrada Seca', appellation 'D.O. Valle del Limarí').
- Copy every printed name EXACTLY as it appears on the label, in the label's language. Do NOT translate
  (write 'Valle del Limarí', not 'Limarí Valley'; 'Italia', not 'Italy'). Only values that are not printed
  anywhere (inferred values) may use their common English name.
- Ignore marketing copy, slogans and legal boilerplate.
- You MAY infer country, region or variety when the appellation fixes them. List a field in inferred_fields ONLY if its value is not printed anywhere on the label.
- NEVER invent a vintage, alcohol %, or bottle size. If you cannot read them, use null. If a number is partly hidden, use null rather than guessing.
- ocr_corrections: only list confident fixes of a misread word. Leave it empty for fragments you cannot resolve; do not explain fragments.
- Use null for anything you cannot determine.
"""

SAVE_TOOL = {
    "name": "save_wine_label",
    "description": "Save the structured fields extracted from a wine label.",
    "input_schema": WineLabel.model_json_schema(),
}


# ---- Helpers ---------------------------------------------------------------------
def run_ocr(path: Path) -> str:
    try:
        from read_label import read_label  #  Cloud Vision script
    except ImportError:
        print("Note: read_label.py not found; sending the photo without OCR text.")
        return ""
    try:
        return read_label(str(path))
    except Exception as e:  # OCR failing leads to claude reading the photo
        print(f"Note: OCR failed ({e}); sending the photo without OCR text.")
        return ""


def encode_image(path: Path) -> dict:
    """Fix phone rotation, shrink large photos, and return a Claude image block."""
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=90)
    data = base64.standard_b64encode(buffer.getvalue()).decode("utf-8")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/jpeg", "data": data},
    }


# ---- Code-side enforcement: printed vs. inferred, and no translated region names -----
# Words that don't identify a place on their own, ignored when comparing region names.
GENERIC_WORDS = {
    "valle",
    "valley",
    "vallee",
    "tal",
    "del",
    "de",
    "des",
    "du",
    "la",
    "le",
    "les",
    "di",
    "della",
    "dei",
    "of",
    "the",
    "region",
    "county",
    "coast",
    "costa",
    "alto",
    "upper",
    "lower",
    "bajo",
}
# Classification labels stripped from an appellation to get the printed place name.
CLASSIFICATION_TOKENS = {
    "do",
    "doc",
    "docg",
    "doca",
    "dop",
    "aoc",
    "aop",
    "ava",
    "igt",
    "igp",
    "vdp",
    "go",
    "wo",
}
CHECKED_FIELDS = ("country", "region", "variety")


def norm(text: Optional[str]) -> str:
    """Lowercase, strip accents and punctuation: 'Valle del Limarí' -> 'valle del limari'."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def place_words(text: Optional[str]) -> set[str]:
    return {w for w in norm(text).split() if w not in GENERIC_WORDS and len(w) > 1}


def strip_classification(appellation: Optional[str]) -> str:
    """'D.O. Valle del Limarí' -> 'Valle del Limarí'; 'Gavi DOCG' -> 'Gavi'."""
    if not appellation:
        return ""
    words = appellation.split()
    is_class = lambda w: re.sub(r"[^a-z]", "", w.lower()) in CLASSIFICATION_TOKENS
    while words and is_class(words[0]):
        words.pop(0)
    while words and is_class(words[-1]):
        words.pop()
    return " ".join(words)


def enforce_printed_text(record: WineLabel, raw_text: str) -> list[str]:
    """Fix the record in place so it follows the rules, and return a list of changes made.

    1. Region: if Claude translated a printed region (e.g. 'Limarí Valley'), replace it with
       the printed form from the appellation ('Valle del Limarí').
    2. inferred_fields: country, region and variety are listed only if their value is NOT
       printed on the label (checked against the OCR text and the fields read from the photo).
    """
    changes = []
    printed = (
        " "
        + norm(
            " ".join(
                filter(
                    None,
                    [
                        raw_text,
                        record.appellation,
                        record.vineyard,
                        record.wine_name,
                        record.producer,
                        record.classification,
                    ],
                )
            )
        )
        + " "
    )
    is_printed = lambda value: bool(norm(value)) and f" {norm(value)} " in printed

    # 1. Undo translated region names
    if record.region and not is_printed(record.region):
        printed_region = strip_classification(record.appellation)
        if printed_region and place_words(record.region) & place_words(printed_region):
            changes.append(
                f"region '{record.region}' -> '{printed_region}' (printed form)"
            )
            record.region = printed_region

    # 2. Rebuild inferred_fields from the evidence, not from Claude's claim
    inferred = set(record.inferred_fields) & set(WineLabel.model_fields)
    for field in CHECKED_FIELDS:
        value = getattr(record, field)
        should_be_inferred = value is not None and not is_printed(value)
        if should_be_inferred and field not in inferred:
            changes.append(f"'{field}' is not printed on the label; marked as inferred")
        if not should_be_inferred and field in inferred:
            changes.append(
                f"'{field}' is printed on the label; removed from inferred_fields"
            )
        inferred.discard(field)
        if should_be_inferred:
            inferred.add(field)
    record.inferred_fields = [f for f in WineLabel.model_fields if f in inferred]
    return changes


def check_record(record: WineLabel) -> list[str]:
    """Code-side sanity checks: the LLM proposes, code verifies."""
    problems = []
    if record.vintage is not None and not (1900 <= record.vintage <= 2030):
        problems.append(f"vintage {record.vintage} looks wrong")
    if record.abv is not None and not (5 <= record.abv <= 25):
        problems.append(f"abv {record.abv} looks wrong")
    if record.volume_ml is not None and record.volume_ml not in {
        187,
        250,
        375,
        500,
        750,
        1000,
        1500,
        3000,
    }:
        problems.append(f"volume_ml {record.volume_ml} is an unusual bottle size")
    unknown = set(record.inferred_fields) - set(WineLabel.model_fields)
    if unknown:
        problems.append(f"inferred_fields lists unknown names: {sorted(unknown)}")
    if not record.producer:
        problems.append("no producer found; matching will rely on wine name only")
    if (
        record.producer
        and record.wine_name
        and record.producer.lower() == record.wine_name.lower()
    ):
        problems.append("producer and wine_name are identical; producer may be wrong")
    return problems


def extract(label_text: str, image_block: Optional[dict]) -> WineLabel:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit(
            "ANTHROPIC_API_KEY is not set. See the setup notes at the top of this file."
        )

    workspace_id = os.environ.get("ANTHROPIC_WORKSPACE_ID")
    headers = {"anthropic-workspace-id": workspace_id} if workspace_id else None
    client = anthropic.Anthropic(default_headers=headers)

    content = []
    if image_block:
        content.append(image_block)
    ocr_part = label_text.strip() or "(no OCR text available; read the photo)"
    content.append(
        {"type": "text", "text": f"OCR text from the wine label:\n\n{ocr_part}"}
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=INSTRUCTIONS,
        tools=[SAVE_TOOL],
        tool_choice={"type": "tool", "name": "save_wine_label"},
        messages=[{"role": "user", "content": content}],
    )

    tool_call = next(block for block in response.content if block.type == "tool_use")
    return WineLabel.model_validate(tool_call.input)


# ---- Main ------------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python extract_label.py <label image or text file>")

    path = Path(sys.argv[1])
    if not path.exists():
        sys.exit(f"File not found: {path}")

    if path.suffix.lower() in IMAGE_TYPES:
        label_text = run_ocr(path)
        image_block = encode_image(path)
    else:
        label_text = path.read_text(encoding="utf-8")
        image_block = None
        if not label_text.strip():
            sys.exit("The text file is empty.")

    try:
        record = extract(label_text, image_block)
    except ValidationError as e:
        sys.exit(f"Claude returned data that doesn't match the schema:\n{e}")
    except anthropic.APIError as e:
        sys.exit(f"Anthropic API error: {e}")

    corrections = enforce_printed_text(record, label_text)

    output = record.model_dump()
    output["code_corrections"] = (
        corrections  # what the code changed in Claude's answer, for auditing
    )
    output["warnings"] = check_record(record)
    output["raw_text"] = label_text

    out_path = path.with_suffix(".json")
    out_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
