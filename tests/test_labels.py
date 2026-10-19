"""Tests for the label pipeline in src/labels:
    preprocessing.py  (Step 1: clean the photo)
    read_label.py     (Step 2: Google Vision OCR)
    extract_label.py  (Step 3: Claude -> validated JSON record)

The tests use the first image found in wine_recommender/wine_labels.

Run from the wine_recommender folder:
    pip install pytest pillow pydantic anthropic
    pytest tests/test_labels.py -v

Offline tests run by default and make no API calls. To also run the live tests
(real Google Vision + Claude calls, which cost money), set RUN_LIVE_TESTS=1:
    PowerShell:  $env:RUN_LIVE_TESTS="1"; pytest tests/test_labels.py -v
    bash:        RUN_LIVE_TESTS=1 pytest tests/test_labels.py -v
Live tests also need ANTHROPIC_API_KEY and Google Cloud credentials.
"""

import importlib
import json
import os
import sys
import types
from pathlib import Path

import pytest
from PIL import Image


# ---- Locate the project ----------------------------------------------------------
def _find_project_root() -> Path:
    """Walk up from this file until a folder containing src/labels is found."""
    for folder in Path(__file__).resolve().parents:
        if (folder / "src" / "labels").is_dir():
            return folder
    raise RuntimeError("Could not find src/labels above this test file.")


ROOT = _find_project_root()
LABELS_SRC = ROOT / "src" / "labels"
WINE_LABELS_DIR = ROOT / "wine_labels"
IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}

# The three files import each other by plain name (e.g. "from read_label import ..."),
# so src/labels must be importable directly.
sys.path.insert(0, str(LABELS_SRC))

import preprocessing  # noqa: E402
import extract_label  # noqa: E402

LIVE = os.environ.get("RUN_LIVE_TESTS", "").lower() in {"1", "true", "yes"}
live_only = pytest.mark.skipif(
    not LIVE, reason="set RUN_LIVE_TESTS=1 to call the real APIs"
)


@pytest.fixture(scope="session")
def label_path() -> Path:
    """The first label photo in wine_labels (sorted by name)."""
    if not WINE_LABELS_DIR.is_dir():
        pytest.skip(f"Folder not found: {WINE_LABELS_DIR}")
    images = sorted(
        p for p in WINE_LABELS_DIR.iterdir() if p.suffix.lower() in IMAGE_TYPES
    )
    if not images:
        pytest.skip(f"No label images in {WINE_LABELS_DIR}")
    return images[0]


@pytest.fixture
def clean_label(label_path, tmp_path) -> Path:
    """Step 1 output for the test label, written to a temporary folder."""
    return preprocessing.preprocess_image(label_path, tmp_path / "clean_label.jpg")


# ---- Step 1: preprocessing.py --------------------------------------------------
class TestPreprocessing:
    def test_output_is_rgb_jpeg_within_max_size(self, clean_label):
        assert clean_label.exists()
        with Image.open(clean_label) as img:
            assert img.format == "JPEG"
            assert img.mode == "RGB"
            assert img.width <= preprocessing.MAX_SIZE[0]
            assert img.height <= preprocessing.MAX_SIZE[1]

    def test_never_enlarges(self, label_path, tmp_path):
        out = preprocessing.preprocess_image(
            label_path, tmp_path / "big.jpg", max_size=(100_000, 100_000)
        )
        with Image.open(label_path) as orig, Image.open(out) as clean:
            upright = preprocessing.ImageOps.exif_transpose(
                orig
            )  # size after phone rotation
            assert clean.size == upright.size

    def test_custom_max_size(self, label_path, tmp_path):
        out = preprocessing.preprocess_image(
            label_path, tmp_path / "small.jpg", max_size=(300, 300)
        )
        with Image.open(out) as img:
            assert max(img.size) <= 300

    def test_rotation_90_swaps_width_and_height(self, label_path, tmp_path):
        plain = preprocessing.preprocess_image(
            label_path, tmp_path / "a.jpg", max_size=(800, 800)
        )
        turned = preprocessing.preprocess_image(
            label_path, tmp_path / "b.jpg", rotation=90, max_size=(800, 800)
        )
        with Image.open(plain) as a, Image.open(turned) as b:
            assert (a.width, a.height) == (b.height, b.width)

    def test_crop(self, label_path, tmp_path):
        out = preprocessing.preprocess_image(
            label_path, tmp_path / "crop.jpg", crop_box=(0, 0, 200, 150)
        )
        with Image.open(out) as img:
            assert img.size == (200, 150)

    def test_main_writes_clean_image(self, label_path, tmp_path, capsys):
        code = preprocessing.main([str(label_path), "--out-dir", str(tmp_path)])
        assert code == 0
        assert (tmp_path / "clean_label.jpg").exists()
        assert "Step 1 complete" in capsys.readouterr().out

    def test_main_demo_skips_processing(self, label_path, tmp_path, capsys):
        code = preprocessing.main(
            [str(label_path), "--out-dir", str(tmp_path), "--demo"]
        )
        assert code == 0
        assert not (tmp_path / "clean_label.jpg").exists()
        assert "Preprocessing skipped" in capsys.readouterr().out

    def test_main_missing_file(self, tmp_path):
        assert preprocessing.main([str(tmp_path / "nope.jpg")]) == 1


# ---- Step 2: read_label.py -----------------------------------------------------
def _fake_vision(text="CONCHA Y TORO\nAMELIA\n2023", error=""):
    """A stand-in for google.cloud.vision that returns fixed OCR text."""
    vision = types.ModuleType("google.cloud.vision")
    calls = {}

    class Image:  # noqa: D401 - mimics vision.Image
        def __init__(self, content):
            self.content = content

    class ImageAnnotatorClient:
        def text_detection(self, image):
            calls["bytes"] = image.content
            annotations = [types.SimpleNamespace(description=text)] if text else []
            return types.SimpleNamespace(
                error=types.SimpleNamespace(message=error), text_annotations=annotations
            )

    vision.Image = Image
    vision.ImageAnnotatorClient = ImageAnnotatorClient
    return vision, calls


def _import_read_label_with(monkeypatch, vision):
    google = types.ModuleType("google")
    cloud = types.ModuleType("google.cloud")
    cloud.vision = vision
    google.cloud = cloud
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.cloud", cloud)
    monkeypatch.setitem(sys.modules, "google.cloud.vision", vision)
    monkeypatch.delitem(sys.modules, "read_label", raising=False)
    module = importlib.import_module("read_label")
    monkeypatch.delitem(
        sys.modules, "read_label"
    )  # don't leak the fake into other tests
    return module


class TestReadLabel:
    def test_imports_cleanly(self, monkeypatch):
        """Importing must not run OCR or import itself."""
        vision, _ = _fake_vision()
        module = _import_read_label_with(monkeypatch, vision)
        assert callable(module.read_label)

    def test_sends_image_bytes_and_returns_text(self, monkeypatch, clean_label):
        vision, calls = _fake_vision("SANCERRE\n2022")
        module = _import_read_label_with(monkeypatch, vision)
        assert module.read_label(str(clean_label)) == "SANCERRE\n2022"
        assert calls["bytes"] == clean_label.read_bytes()

    def test_no_text_returns_empty_string(self, monkeypatch, clean_label):
        vision, _ = _fake_vision(text="")
        module = _import_read_label_with(monkeypatch, vision)
        assert module.read_label(str(clean_label)) == ""

    def test_api_error_raises(self, monkeypatch, clean_label):
        vision, _ = _fake_vision(error="quota exceeded")
        module = _import_read_label_with(monkeypatch, vision)
        with pytest.raises(RuntimeError, match="quota exceeded"):
            module.read_label(str(clean_label))

    @live_only
    def test_live_ocr_finds_text(self, clean_label):
        pytest.importorskip("google.cloud.vision")
        sys.modules.pop("read_label", None)
        from read_label import read_label

        text = read_label(str(clean_label))
        print("\nOCR text:\n", text)
        assert text.strip(), "Vision returned no text for the label"


# ---- Step 3: extract_label.py --------------------------------------------------
class TestExtractHelpers:
    def test_norm(self):
        assert extract_label.norm("Valle del Limarí") == "valle del limari"
        assert extract_label.norm(None) == ""

    def test_strip_classification(self):
        assert (
            extract_label.strip_classification("D.O. Valle del Limarí")
            == "Valle del Limarí"
        )
        assert extract_label.strip_classification("Gavi DOCG") == "Gavi"

    def test_real_fix_drops_no_ops(self):
        assert extract_label._real_fix("CONTR0LEE -> CONTROLEE")
        assert not extract_label._real_fix("CONTROLEE -> CONTROLEE")
        assert not extract_label._real_fix("97 -> null")

    def test_enforce_undoes_translated_region(self):
        record = extract_label.WineLabel(
            region="Limarí Valley",
            appellation="D.O. Valle del Limarí",
            country="Chile",
            inferred_fields=["region"],
        )
        changes = extract_label.enforce_printed_text(record, "AMELIA VALLE DEL LIMARI")
        assert record.region == "Valle del Limarí"
        assert "region" not in record.inferred_fields
        assert "country" in record.inferred_fields  # 'Chile' is not printed
        assert changes

    def test_check_record_flags_bad_values(self):
        record = extract_label.WineLabel(vintage=1850, abv=40, volume_ml=700)
        problems = " ".join(extract_label.check_record(record))
        for word in ("vintage", "abv", "volume_ml", "producer"):
            assert word in problems

    def test_encode_image_with_real_label(self, label_path):
        block = extract_label.encode_image(label_path)
        assert block["type"] == "image"
        assert block["source"]["media_type"] == "image/jpeg"
        assert len(block["source"]["data"]) > 1000


class TestExtractPipeline:
    def test_demo_mode(self, monkeypatch, label_path):
        monkeypatch.setattr(extract_label, "DEMO_MODE", True)
        out = extract_label.extract_label_from_photo(label_path)
        assert out["producer"] == "Concha y Toro"
        assert out["raw_text"] == "(demo mode)"

    def test_pipeline_with_stubbed_apis(self, monkeypatch, clean_label):
        """Runs the real pipeline on the label photo with OCR and Claude replaced by stubs."""
        monkeypatch.setattr(extract_label, "DEMO_MODE", False)
        monkeypatch.setattr(
            extract_label, "run_ocr", lambda path: "AMELIA VALLE DEL LIMARI 2023"
        )
        seen = {}

        def fake_extract(text, image_block):
            seen["text"], seen["image"] = text, image_block
            return extract_label.WineLabel(
                producer="Concha y Toro",
                wine_name="Amelia",
                region="Limarí Valley",
                appellation="D.O. Valle del Limarí",
                country="Chile",
                vintage=2023,
            )

        monkeypatch.setattr(extract_label, "extract", fake_extract)
        out = extract_label.extract_label_from_photo(clean_label)

        assert seen["text"] == "AMELIA VALLE DEL LIMARI 2023"
        assert seen["image"]["type"] == "image"
        assert out["region"] == "Valle del Limarí"  # code-side fix applied
        assert out["code_corrections"]
        assert set(out) >= {"warnings", "raw_text", "inferred_fields"}

    def test_to_label_record(self):
        out = dict(extract_label.DEMO_LABEL)
        record = extract_label.to_label_record(out)
        assert set(record) == set(extract_label.HANDOFF_FIELDS) | {"country"}

    def test_run_ocr_falls_back_to_empty_on_failure(self, monkeypatch, clean_label):
        broken = types.ModuleType("read_label")

        def boom(path):
            raise RuntimeError("no credentials")

        broken.read_label = boom
        monkeypatch.setitem(sys.modules, "read_label", broken)
        assert extract_label.run_ocr(clean_label) == ""

    def test_main_demo_writes_json(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            extract_label, "DEMO_MODE", False
        )  # restored after the test
        assert extract_label.main(["--demo", "--out-dir", str(tmp_path)]) == 0
        saved = json.loads(
            (tmp_path / "label_extraction.json").read_text(encoding="utf-8")
        )
        assert saved["wine_name"] == "Amelia"

    @live_only
    def test_live_end_to_end(self, monkeypatch, clean_label):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY is not set")
        monkeypatch.setattr(extract_label, "DEMO_MODE", False)
        out = extract_label.extract_label_from_photo(clean_label)
        print("\n" + json.dumps(out, indent=2, ensure_ascii=False))
        assert (
            out["producer"] or out["wine_name"]
        ), "Claude found neither producer nor wine name"
