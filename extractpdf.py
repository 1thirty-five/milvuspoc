"""
Extract text from documents with pypdfium2 + Unlimited-OCR, and feed it into Milvus.

Replaces the old PyMuPDF front-end. Two pieces do the work now:

  pypdfium2       the PDF engine (Google's PDFium, the renderer inside Chrome).
                  It pulls the text layer out of born-digital PDFs and rasterizes
                  pages to images for OCR. One pip wheel, no system binaries,
                  Apache-2.0/BSD-3 — which also retires PyMuPDF's AGPL-3.0
                  obligation. (Baidu's own examples rasterize with PyMuPDF; we
                  use pypdfium2 instead so PyMuPDF is gone from the project.)

  Unlimited-OCR   the OCR engine: https://github.com/baidu/Unlimited-OCR, model
                  `baidu/Unlimited-OCR` (MIT, 3.34B params). It is not a classic
                  detect-then-recognize OCR library but a vision-language model
                  that reads a whole page in one shot and emits structured
                  markdown — so tables, reading order, and multi-column layout
                  survive instead of collapsing into a bag of text boxes. Built
                  to push DeepSeek-OCR further at long-horizon document parsing.
                  Fully local: no API key, no page cap, no per-page cost.

This is the front of the pipeline: files -> text -> chunks -> (input.md | Milvus).
The common case is to drop documents into the `fileinput/` folder and ingest the
whole folder in one run:

    python extractpdf.py --store        # extract every file in fileinput/ -> Milvus

Each function does one thing and returns plain objects, so a UI layer can call
them independently:

    from extractpdf import extract_text, chunk_text
    text   = extract_text("scan.pdf")   # OCRs the pages that need it
    chunks = chunk_text(text)           # each chunk fits the VARCHAR(2048) field

Milvus stores `text` as VARCHAR(2048) (see loadmilvus.ensure_collection), so
extracted text is chunked to stay under that limit before it's embedded/stored.
Every chunk also carries its source filename, stored alongside the vector in the
collection's dynamic `source` field so you can tell which document it came from.

Tables get their own path. The OCR model returns them as real markup, so rather
than chunking that HTML as if it were prose, `chunk_text` expands the table's
rowspan/colspan into a grid and emits one labelled sentence per row --
"Name of company: MONOLITH SOFTWARE INC.; Location: Meguro-ku, Tokyo; ..." --
packed on row boundaries so a row is never split across chunks. See chunk_text.

HARDWARE — read before running. Unlimited-OCR is a 3.34B-parameter VLM and its
modeling code is hard-wired to CUDA (`.cuda()`, `torch.autocast("cuda")`), so
there is no CPU path. The BF16 weights are **6.21 GiB**, which does not fit this
machine's RTX 3050 (**6.0 GiB**). `--ocr-quant auto` (the default) therefore
loads 4-bit NF4 via bitsandbytes (~1.9 GiB) so it fits, dropping to bf16 only on
a card with headroom. If bitsandbytes is unavailable, or quality at 4-bit is not
good enough, run the model on a bigger GPU and point this script at it with
`--ocr-backend unlimited-server` (vLLM or SGLang, both documented upstream).

OCR modes (`--ocr`):

    auto     default. Use the text layer where there is one, and OCR only the
             pages that come back empty or near-empty. A born-digital PDF costs
             nothing extra; a scanned one is OCR'd page by page; a PDF that is
             half typed and half scanned images gets the right treatment per
             page. `--ocr-min-chars` is the cutoff that decides "near-empty".
             This matters more than usual here: the text layer is exact and free,
             while a VLM page costs seconds of GPU time and can hallucinate.
    always   ignore the text layer and OCR every page. Use when a PDF *has* a
             text layer but it is garbage — bad embedded fonts, mojibake, or a
             broken producer — which extracts cleanly but embeds to nonsense.
             Also the mode to use when you want the model's markdown structure
             (tables, headings) rather than the text layer's flat prose.
    never    text layer only. Fastest, needs no GPU at all, and reproduces the
             old PyMuPDF behavior: scanned pages simply yield nothing.

OCR backends (`--ocr-backend`):

    unlimited         default. Loads `baidu/Unlimited-OCR` locally through
                      transformers. First run downloads ~6.2 GiB to the HF cache.
    unlimited-server  same model, but served by vLLM or SGLang behind an
                      OpenAI-compatible endpoint (`--ocr-url`). Use when the GPU
                      that can hold the model is not this one.

Supported input: `.pdf`, plain `.txt`, and image files (`.png`, `.jpg`, `.jpeg`,
`.tif`, `.tiff`, `.bmp`, `.webp`) which are OCR'd directly. Note this drops the
XPS/EPUB/MOBI/FB2/CBZ formats MuPDF also opened — PDFium is a PDF engine only.
Nothing in `fileinput/` used them; add a per-format reader if that changes.

Prerequisites:
    pip install -r requirements.txt
    pip install accelerate bitsandbytes      # for the 4-bit local path

Storing is incremental: a chunk already in Milvus (matched by a hash of its
model + text) keeps its stored vector instead of being embedded again, so adding
one file to fileinput/ only costs that file's chunks. See store_records.

Run:
    python extractpdf.py                          # preview chunks from fileinput/
    python extractpdf.py --store                  # embed + store the whole folder
    python extractpdf.py --store --model bge-m3
    python extractpdf.py --store --reset          # ignore the cache, rebuild from scratch
    python extractpdf.py scan.pdf --ocr always    # force OCR on every page
    python extractpdf.py scan.pdf --ocr-dpi 400   # rasterize finer for small type
    python extractpdf.py doc.pdf  --ocr-quant 8bit
    python extractpdf.py doc.pdf  --ocr-backend unlimited-server --ocr-url http://gpubox:10000
    python extractpdf.py file.pdf                 # a single file instead of the folder
    python extractpdf.py file.pdf --out input.md  # write chunks as documents
"""

import base64
import re
import sys
import tempfile
import time
from pathlib import Path

import pypdfium2 as pdfium

# Sentence boundary: end punctuation (. ! ?) followed by whitespace and the start
# of a new sentence (optional opening quote/bracket, then a capital or digit).
# Requiring a capital/digit after the space avoids splitting on decimals ("2.5")
# and most lowercase abbreviations ("e.g. the ..."); it's not perfect around
# "Fig. 3" / "et al." but keeps chunks from cutting mid-sentence, which matters
# far more for embedding/clustering quality than the odd false split.
_SENTENCE_BOUNDARY = re.compile(r'(?<=[.!?])\s+(?=["\'(\[]?[A-Z0-9])')

# Default folder ingested when no path is given. Drop documents here.
DEFAULT_INPUT_DIR = "fileinput"

# Images are OCR'd whole, with no text layer to consult, so they ignore the OCR
# mode (except `never`, which skips them — there is nothing else to read).
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

# What we can open. PDFium is PDF-only, so the XPS/EPUB/MOBI/FB2/CBZ formats
# MuPDF handled are gone; images arrive in exchange, via OCR.
SUPPORTED_EXTS = {".pdf", ".txt"} | IMAGE_EXTS

# OCR modes. See the module docstring for when each one is the right call.
OCR_AUTO, OCR_ALWAYS, OCR_NEVER = "auto", "always", "never"
OCR_MODES = (OCR_AUTO, OCR_ALWAYS, OCR_NEVER)
DEFAULT_OCR_MODE = OCR_AUTO
DEFAULT_OCR_BACKEND = "unlimited"

# The model, and the generation settings Baidu documents for it. These are not
# arbitrary: `no_repeat_ngram_size` + `ngram_window` are the upstream guard
# against the degenerate repetition loops VLM decoders fall into on dense pages,
# and 32768 is the context the model was trained to parse a full page within.
UNLIMITED_MODEL = "baidu/Unlimited-OCR"
UNLIMITED_MAX_LENGTH = 32768
UNLIMITED_NO_REPEAT_NGRAM = 35
UNLIMITED_NGRAM_WINDOW = 128

# Figure captioning (--ocr-figures). `document parsing.` tags picture regions as
# `<|det|>image [x1, y1, x2, y2]<|/det|>`, which the normal text path discards as
# layout metadata. With this on, each such region is cropped out of the page
# render and sent back through the model under a figure prompt, so a chart could
# become text that embeds and searches like anything else.
#
# MEASURED RESULT ON THIS CORPUS: it does not work, and is off for that reason
# rather than merely for cost. The annual report has figures on only 2 of its 103
# pages, both org charts (governance and sustainability structures). Detection and
# cropping are correct -- the boxes were rendered and checked by eye -- but the
# description step fails: page 11 returned nothing at all, and page 33, a diagram
# with roughly forty labels, returned "Compliance Hotline General Meeting of
# Shareholders". Three prompts were compared on the same crop and `Parse the
# figure.` and `Free OCR.` produced byte-identical 50-character output while
# `document parsing.` produced 31. Identical results from different prompts means
# the prompt is not the bottleneck: the model is perceiving a fraction of the
# image whatever it is asked. The likely cause is scale -- single-image inference
# runs at base_size=1024 / image_size=640 with crop_mode, and these diagrams are
# ~1950x1460, so the tiling appears to lose the labels.
#
# What is kept, and why: region detection, the 0-999 coordinate conversion, and
# cropping are all verified correct and are the reusable half. A model that can
# read diagrams -- or a data chart rather than an org chart, which is a genuinely
# easier target -- would drop straight into this path. Turning it on today would
# write two-word captions into the vector store, so it stays off.
DEFAULT_OCR_FIGURES = False
FIGURE_PROMPT = "Parse the figure."
FIGURE_LABEL = "image"

# Detection boxes are normalized to 0-999 and scaled by the rendered page size.
# Taken from the model's own draw_bounding_boxes: `x1 / 999 * image_width`.
DET_COORD_SCALE = 999

# Crops smaller than this on either side are rules, bullets, or logo fragments --
# never a chart worth a 50-second inference pass.
MIN_FIGURE_PX = 48

# `document parsing.` is the upstream prompt and returns structured markdown,
# which preserves tables and reading order. `Free OCR.` returns flatter plain
# text; switch with --ocr-prompt if the markdown scaffolding is polluting chunks.
DEFAULT_OCR_PROMPT = "document parsing."

# Single-image inference config ("gundam" upstream): base_size 1024 with a 640
# crop grid, which tiles a page instead of downsampling it, so small print stays
# legible. This is why per-page cost is seconds rather than milliseconds.
UNLIMITED_BASE_SIZE = 1024
UNLIMITED_IMAGE_SIZE = 640
UNLIMITED_CROP_MODE = True

# Rasterization DPI for pages handed to OCR. PDF user space is 72 dpi, so the
# render scale is dpi/72. 300 is what Baidu's own PDF examples use, and the
# model's crop grid assumes roughly that much detail; below ~200 small type
# starts dropping out, above ~400 you pay in render time and VRAM for little.
DEFAULT_OCR_DPI = 300

# In `auto` mode, a page whose text layer yields fewer than this many characters
# is treated as having no real text and is sent to OCR. Genuinely scanned pages
# return 0; the margin covers pages carrying only a stray header or page number.
# Kept low deliberately — a sparse-but-real page (a cover, a section divider)
# costs one wasted OCR call, whereas missing a scanned page loses it entirely.
DEFAULT_OCR_MIN_CHARS = 32

# Default endpoint for the `unlimited-server` backend (SGLang's documented port).
DEFAULT_OCR_URL = "http://127.0.0.1:10000"

# Chunk sizing. Smaller chunks make each one a tighter semantic unit (roughly a
# few sentences about one idea), so its embedding is topically pure and clusters
# come out intuitive / single-topic. Milvus stores `text` as VARCHAR(2048), so
# these stay well under that. OVERLAP repeats a sentence or two between
# consecutive chunks so context spanning a boundary still embeds together.
DEFAULT_MAX_CHARS = 600
DEFAULT_OVERLAP = 100

# Table chunks get a larger budget than prose. A serialized row repeats its
# headers ("Name of company: ...; Location: ...") so rows run long, and the whole
# point of the table path is to keep rows intact -- a 600-char cap would split
# them back apart. Rows are still packed whole, so this is a ceiling, not a
# target. Kept well under the byte budget below.
TABLE_MAX_CHARS = 1500

# Milvus' `text` field is VARCHAR(2048), measured in **bytes**. Yen signs and
# curly quotes cost 2-3 bytes each, so character length alone does not bound it.
# 2000 leaves headroom for the joining spaces.
VARCHAR_BYTE_BUDGET = 2000

# Shortest chunk worth embedding. Word-splitting an oversized row can shed stray
# fragments -- a bare "H:" or "." -- and those embed to noise that then competes
# with real content in search and drags on clustering. Deliberately tiny: the
# test is for degenerate leftovers, not short-but-real text, so a genuine
# heading like "Development" is kept.
MIN_CHUNK_CHARS = 3

# Loading a 3.34B VLM costs a minute and gigabytes, and a folder run hits many
# files, so the built engine is cached and reused for the whole process.
_OCR_ENGINES = {}

# Loaded (tokenizer, model) pairs, keyed by resolved quantization. Separate from
# _OCR_ENGINES because that is keyed by prompt too: figure captioning uses a
# second prompt against the *same* weights, and without this the model would be
# loaded twice -- 25s and ~2 GiB of VRAM for a duplicate that will not fit.
_LOADED_MODELS = {}

# Filled in by the OCR paths so callers can report how much OCR actually ran.
_OCR_STATS = {"pages": 0, "seconds": 0.0, "figures": 0}


def _shim_transformers():
    """Restore one symbol Unlimited-OCR's remote code imports but transformers 5 dropped.

    `modeling_deepseekv2.py` does `from transformers.utils.import_utils import
    is_torch_fx_available`. That helper was removed in transformers 5.x, so the
    import raises before any weights load. Baidu pins transformers 4.57.1, but
    this project's embedding stack (sentence-transformers / bge-m3) is on 5.10.2
    and downgrading it to satisfy the OCR model would put the pipeline's most
    important stage on unsupported versions.

    Re-adding the symbol is the smaller intervention. It is only consulted to
    decide whether torch.fx symbolic tracing is available — a graph-tracing
    convenience for export/quantization tooling, not something inference needs —
    so reporting False is both safe and accurate here.
    """
    from transformers.utils import import_utils

    if not hasattr(import_utils, "is_torch_fx_available"):
        def is_torch_fx_available():
            return False

        import_utils.is_torch_fx_available = is_torch_fx_available
        import transformers.utils as utils
        if not hasattr(utils, "is_torch_fx_available"):
            utils.is_torch_fx_available = is_torch_fx_available


# Attributes `PretrainedConfig` set unconditionally in transformers 4.x and no
# longer does in 5.x, with their 4.x defaults. Unlimited-OCR's modeling code
# reads them straight off the config -- `config.pad_token_id` is the first thing
# DeepseekV2Model.__init__ touches -- so on 5.x the model fails to construct with
# an AttributeError before a single weight is loaded. Backfilling reproduces
# exactly what the model saw on the version Baidu developed against; it is not a
# guess about intent. Kept as data rather than a chain of try/excepts so the next
# missing attribute is a one-line addition.
_LEGACY_CONFIG_DEFAULTS = {
    "pad_token_id": None,
    "sep_token_id": None,
    "decoder_start_token_id": None,
    "use_cache": True,
    "tie_word_embeddings": True,
    "pruned_heads": {},
    "is_decoder": False,
    "add_cross_attention": False,
    "torchscript": False,
    "tie_encoder_decoder": False,
}


def _backfill_config(config):
    """Restore the config attributes the remote code expects but 5.x drops.

    Two sources, in order of authority:

    1. The config class's own declared defaults, read off the `__init__`
       signatures along its MRO. `DeepseekV2Config.__init__` assigns
       `self.attention_dropout`, `self.rms_norm_eps` and ~20 others *before*
       calling `super().__init__()`, and transformers 5's `PretrainedConfig`
       discards anything set ahead of it — so every one of those defaults is
       declared and then silently lost. Reading them back off the signature
       recovers Baidu's own values rather than a guess at them; anything
       genuinely specified in config.json is already present and untouched.

    2. `_LEGACY_CONFIG_DEFAULTS` for the base `PretrainedConfig` attributes that
       4.x set unconditionally and have no declaration to recover.

    Applies to nested sub-configs too, since the modeling code reads whichever
    one it is handed.
    """
    import inspect
    from transformers.configuration_utils import PretrainedConfig

    for klass in type(config).__mro__:
        init = klass.__dict__.get("__init__")
        if init is None:
            continue
        try:
            params = inspect.signature(init).parameters
        except (TypeError, ValueError):        # C-level or unintrospectable
            continue
        for name, param in params.items():
            if name == "self" or param.default is inspect.Parameter.empty:
                continue
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue
            if not hasattr(config, name):
                setattr(config, name, param.default)

    for name, default in _LEGACY_CONFIG_DEFAULTS.items():
        if not hasattr(config, name):
            setattr(config, name, default)

    # transformers 5 replaced the `rope_theta` / `rope_scaling` pair with a single
    # `rope_parameters` dict, and LlamaRotaryEmbedding now subscripts it directly.
    # The remote code only sets the old fields, so the new one is left None and
    # rotary embedding construction fails. transformers ships this converter for
    # exactly that migration -- calling it is better than assembling the dict by
    # hand, since it also handles scaled rope types and per-layer variants.
    if getattr(config, "rope_parameters", None) is None:
        standardize = getattr(config, "standardize_rope_params", None)
        if callable(standardize) and getattr(config, "rope_theta", None) is not None:
            standardize()

    for value in list(vars(config).values()):
        if isinstance(value, PretrainedConfig):
            _backfill_config(value)
    return config


def _repair_position_ids(model):
    """Rebuild `position_ids` buffers that transformers 5 loaded as garbage.

    The vision encoder declares its positions as a derived constant:

        self.register_buffer("position_ids", torch.arange(num_positions).expand((1, -1)))

    Being derived, it is not in the checkpoint. transformers 4 kept whatever the
    constructor had put there; transformers 5 builds modules on the meta device
    and then materializes anything missing from the checkpoint with
    *uninitialized memory* -- it even says so, reporting
    `vision_model.embeddings.position_ids | MISSING | newly initialized`.

    Those junk values are then used as gather indices into the position
    embedding table, so the first forward pass dies in a CUDA kernel with
    "index out of bounds" rather than anywhere near the real cause. Recomputing
    the arange restores exactly what the constructor intended -- this is not a
    correction to the model, it is the value the model already defines.
    """
    import torch

    repaired = []
    for name, buffer in list(model.named_buffers()):
        if not name.split(".")[-1] == "position_ids":
            continue
        expected = (torch.arange(buffer.shape[-1], device=buffer.device)
                    .expand(buffer.shape).contiguous())
        if buffer.dtype == expected.dtype and torch.equal(buffer, expected):
            continue                              # already correct, leave it alone
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        parent.register_buffer(attr, expected, persistent=False)
        repaired.append(name)

    if repaired:
        print(f"   repaired {len(repaired)} position_ids buffer(s): "
              f"{', '.join(repaired)}")
    return model


def _force_vision_dtype(model, dtype):
    """Cast the CLIP tower's output back to `dtype` so the vision/text merge works.

    The model merges vision features into the token embeddings with
    `masked_scatter_`, which requires both sides to share a dtype. The embeddings
    are bf16, but the CLIP tower comes out float32, so the merge raises
    `expected self and source to have same dtypes but got BFloat16 and Float`.

    The cause is not the weights -- those load as bf16 -- but the LayerNorms.
    `infer()` wraps generation in `torch.autocast("cuda", bfloat16)`, and under
    autocast `nn.LayerNorm` computes in float32 and *returns* float32, so the
    tower drifts to fp32 at its first norm and never comes back. deepencoder.py
    defines a dtype-preserving LayerNorm for exactly this (it computes in float32
    and returns `orig_type`), but the CLIP tower uses plain `nn.LayerNorm`
    instead, so nothing restores the dtype.

    Casting the tower's output applies that same convention one level up:
    compute in the wider type, hand back the model's type. Precision is
    unaffected -- the extra range is used and then discarded exactly as the
    custom LayerNorm intends.
    """
    vision = getattr(getattr(model, "model", model), "vision_model", None)
    if vision is None:
        return model

    def cast_output(module, args, output):
        if hasattr(output, "dtype") and output.dtype != dtype:
            return output.to(dtype)
        return output

    vision.register_forward_hook(cast_output)
    return model


def _pick_quantization(requested):
    """Resolve `--ocr-quant auto` against the GPU actually present.

    The BF16 checkpoint is 6.21 GiB of weights before any activations or KV
    cache, so it needs a card with meaningfully more than that. `auto` measures
    the device instead of assuming: with real headroom it stays at bf16, which is
    the reference precision the model was released in; otherwise it drops to
    4-bit NF4 (~1.9 GiB), which is the difference between running and not running
    at all on a 6 GiB laptop card.
    """
    if requested != "auto":
        return requested

    import torch
    if not torch.cuda.is_available():
        return "4bit"                             # will fail later, but fail on the real reason
    total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    return "none" if total >= 9.0 else "4bit"


def _load_unlimited(quant):
    """Load `baidu/Unlimited-OCR` and its tokenizer, quantized as needed."""
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    if not torch.cuda.is_available():
        raise SystemExit(
            "Unlimited-OCR requires CUDA — its modeling code calls .cuda() and\n"
            "torch.autocast('cuda') directly, so there is no CPU path.\n"
            "Options: run on a CUDA machine, serve the model elsewhere and use\n"
            "  --ocr-backend unlimited-server --ocr-url http://<host>:10000\n"
            "or skip OCR entirely with --ocr never (text layer only).")

    _shim_transformers()
    quant = _pick_quantization(quant)
    if quant in _LOADED_MODELS:                   # already paid for; reuse the weights
        return _LOADED_MODELS[quant]

    tokenizer = AutoTokenizer.from_pretrained(UNLIMITED_MODEL, trust_remote_code=True)
    config = _backfill_config(
        AutoConfig.from_pretrained(UNLIMITED_MODEL, trust_remote_code=True))
    # bfloat16 for *everything*, quantized or not. Only the language model's
    # Linear layers get 4-bit; the SAM/CLIP vision towers stay dense, and without
    # an explicit dtype they would load as float32. The model then merges vision
    # output into the bf16 token embeddings with masked_scatter_, which requires
    # matching dtypes and fails outright on a mismatch. bf16 throughout is also
    # what upstream loads, so this matches the reference setup rather than
    # inventing one. (transformers 5 renamed `torch_dtype` to `dtype`.)
    kwargs = {"trust_remote_code": True, "use_safetensors": True, "config": config,
              "dtype": torch.bfloat16}

    if quant in ("4bit", "8bit"):
        try:
            from transformers import BitsAndBytesConfig
            import bitsandbytes  # noqa: F401  (imported to fail early with a clear message)
        except ImportError:
            raise SystemExit(
                f"{quant} loading needs bitsandbytes. Run:\n"
                "    pip install bitsandbytes accelerate\n"
                "The BF16 weights are 6.21 GiB and will not fit a 6 GiB card, so\n"
                "this is not optional here — or use --ocr-backend unlimited-server.")
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=(quant == "4bit"),
            load_in_8bit=(quant == "8bit"),
            bnb_4bit_quant_type="nf4",
            # Compute stays bf16 even though storage is 4-bit: only the weights
            # are compressed, matmuls dequantize back to bf16, which is what the
            # model's own autocast blocks expect.
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        # Pin to GPU 0. Letting accelerate spill layers to CPU would break the
        # hard-coded .cuda() calls inside infer() rather than merely slow it down.
        kwargs["device_map"] = {"": 0}

    print(f"Loading {UNLIMITED_MODEL} ({quant})... first run downloads ~6.2 GiB.")
    model = AutoModel.from_pretrained(UNLIMITED_MODEL, **kwargs).eval()
    if quant == "none":
        model = model.cuda()                      # bnb paths are already placed by device_map
    _repair_position_ids(model)
    _force_vision_dtype(model, torch.bfloat16)
    _LOADED_MODELS[quant] = (tokenizer, model)
    return tokenizer, model


def get_ocr(backend=DEFAULT_OCR_BACKEND, prompt=DEFAULT_OCR_PROMPT,
            quant="auto", url=DEFAULT_OCR_URL):
    """Return a callable(image_path) -> str for the named OCR backend.

    The engine is built once and cached: constructing one means loading several
    gigabytes of weights, which must not be paid per page or per file.
    """
    key = (backend, prompt, quant, url)
    if key in _OCR_ENGINES:
        return _OCR_ENGINES[key]

    if backend == "unlimited":
        tokenizer, model = _load_unlimited(quant)

        def run(image_path):
            # eval_mode=True is what makes infer() *return* the text; without it
            # the method streams tokens to stdout and writes result.md instead,
            # which is fine interactively and useless in a pipeline. It also
            # disables the token streamer, so a folder run stays quiet.
            with tempfile.TemporaryDirectory(prefix="uocr_") as out_dir:
                return model.infer(
                    tokenizer,
                    prompt=f"<image>{prompt}",
                    image_file=str(image_path),
                    output_path=out_dir,
                    base_size=UNLIMITED_BASE_SIZE,
                    image_size=UNLIMITED_IMAGE_SIZE,
                    crop_mode=UNLIMITED_CROP_MODE,
                    max_length=UNLIMITED_MAX_LENGTH,
                    no_repeat_ngram_size=UNLIMITED_NO_REPEAT_NGRAM,
                    ngram_window=UNLIMITED_NGRAM_WINDOW,
                    eval_mode=True,
                ) or ""

    elif backend == "unlimited-server":
        import json
        import urllib.request

        def run(image_path):
            # OpenAI-compatible chat/completions, which is what both vLLM and
            # SGLang expose for this model. The image rides as a data: URI so the
            # server needs no shared filesystem with us.
            suffix = Path(image_path).suffix.lower().lstrip(".")
            mime = "jpeg" if suffix in ("jpg", "jpeg") else suffix
            data = base64.b64encode(Path(image_path).read_bytes()).decode()
            payload = {
                "model": "Unlimited-OCR",
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": f"<image>{prompt}"},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/{mime};base64,{data}"}},
                ]}],
                "temperature": 0,
                "max_tokens": UNLIMITED_MAX_LENGTH,
            }
            req = urllib.request.Request(
                f"{url.rstrip('/')}/v1/chat/completions",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=1200) as resp:
                body = json.load(resp)
            return body["choices"][0]["message"]["content"] or ""

    else:
        raise SystemExit(
            f"Unknown OCR backend: {backend} "
            "(expected unlimited or unlimited-server)")

    _OCR_ENGINES[key] = run
    return run


# Grounding blocks, e.g. `<|det|>title [104, 80, 340, 96]<|/det|>3. Description...`
# The *contents* of a det block are layout metadata -- an element type and pixel
# coordinates -- so the whole block is dropped, delimiters included. Stripping
# only the delimiters (the obvious reading) would leave "title [104, 80, 340, 96]"
# sitting in the prose, and those coordinates would then be embedded as if they
# were text, polluting every chunk they land in.
_OCR_DET_BLOCK = re.compile(r"<\|det\|>.*?<\|/det\|>", re.S)

# Any other special token is unwrapped rather than deleted with its contents --
# a `<|ref|>...<|/ref|>` span holds real page text, unlike a det block.
_OCR_TOKEN = re.compile(r"<\|[^|>]*\|>")


def clean_ocr_text(text):
    """Strip Unlimited-OCR's layout metadata, keeping the page's actual text.

    Markdown structure the model emits (headings, tables) is deliberately kept:
    that is the reason for using a document-parsing VLM over classic OCR in the
    first place, and it survives chunking as ordinary text.
    """
    text = _OCR_DET_BLOCK.sub(" ", text or "")
    return _OCR_TOKEN.sub(" ", text)


def _ocr_image(image_path, backend, prompt, quant, url, count_page=True):
    """OCR one image file and return the **raw** model output.

    Raw, not cleaned, because the `<|det|>` blocks the cleaner strips are exactly
    where figure bounding boxes live. Callers that only want prose call
    `clean_ocr_text` themselves; the figure path needs the metadata first.
    """
    started = time.time()
    raw = get_ocr(backend, prompt, quant, url)(image_path) or ""
    if count_page:
        _OCR_STATS["pages"] += 1
    _OCR_STATS["seconds"] += time.time() - started
    return raw


# Both shapes the model emits for a labelled region:
#   <|det|>image [x1, y1, x2, y2]<|/det|>
#   <|ref|>image<|/ref|><|det|>[[x1, y1, x2, y2]]<|/det|>
_DET_LABELLED = re.compile(
    r"<\|det\|>\s*([A-Za-z_][\w-]*)\s*(\[[^<]*?\])\s*<\|/det\|>", re.S)
_REF_LABELLED = re.compile(
    r"<\|ref\|>(.*?)<\|/ref\|>\s*<\|det\|>(.*?)<\|/det\|>", re.S)
_INTEGER = re.compile(r"-?\d+")


def _parse_boxes(text):
    """Pull 4-number boxes out of a coordinate string.

    The model's own helper does `eval()` on this substring. We do not: it is
    generated text, and a decoder that has just been told to emit brackets is
    exactly the wrong thing to hand to a Python evaluator. Reading the integers
    out is equally effective and cannot execute anything.
    """
    numbers = [int(n) for n in _INTEGER.findall(text or "")]
    return [tuple(numbers[i:i + 4]) for i in range(0, len(numbers) - 3, 4)]


def figure_regions(raw_text, label=FIGURE_LABEL):
    """Return the [(x1, y1, x2, y2), ...] figure boxes in 0-999 space."""
    boxes = []
    for found_label, coords in _DET_LABELLED.findall(raw_text or ""):
        if found_label.strip().lower() == label:
            boxes.extend(_parse_boxes(coords))
    for found_label, coords in _REF_LABELLED.findall(raw_text or ""):
        if found_label.strip().lower() == label:
            boxes.extend(_parse_boxes(coords))
    return boxes


def caption_figures(page_image, raw_text, page_number, backend, quant, url,
                    progress=False):
    """Crop each tagged figure out of `page_image` and describe it.

    Returns a list of caption strings, each prefixed with its page and index.
    That prefix is not decoration: a caption is **model-generated description**,
    not text transcribed from the page, so it must stay distinguishable from OCR
    output once it is embedded and later retrieved. Marking it in the text
    itself — rather than only in a metadata field — means it stays visible in
    search results and cannot be quoted back as if the report had said it.
    """
    boxes = figure_regions(raw_text)
    if not boxes:
        return []

    width, height = page_image.size
    captions = []
    for index, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        left = int(min(x1, x2) / DET_COORD_SCALE * width)
        right = int(max(x1, x2) / DET_COORD_SCALE * width)
        top = int(min(y1, y2) / DET_COORD_SCALE * height)
        bottom = int(max(y1, y2) / DET_COORD_SCALE * height)
        left, top = max(0, left), max(0, top)
        right, bottom = min(width, right), min(height, bottom)
        if right - left < MIN_FIGURE_PX or bottom - top < MIN_FIGURE_PX:
            continue

        if progress:
            print(f"\r    figure {index} on page {page_number}...",
                  end="", flush=True)
        crop = page_image.crop((left, top, right, bottom))
        with tempfile.TemporaryDirectory(prefix="uocr_fig_") as tmp:
            path = Path(tmp) / f"figure_{page_number:04d}_{index}.png"
            crop.save(path)
            # count_page=False: this is an extra pass over part of a page already
            # counted, so folding it into the page tally would misreport both the
            # page count and the per-page average.
            described = clean_ocr_text(
                _ocr_image(path, backend, FIGURE_PROMPT, quant, url,
                           count_page=False))
        described = " ".join(described.split())
        if described:
            captions.append(f"[Figure {index}, page {page_number}] {described}")
    return captions


def _pdf_pages(path, ocr, backend, dpi, min_chars, prompt, quant, url, progress,
               figures=DEFAULT_OCR_FIGURES, on_page=None):
    """Yield (page_number, text) for a PDF, OCR'ing pages according to `ocr`.

    The text layer is consulted first unless the mode is `always`, because
    reading it is effectively free and exact, while a VLM page costs seconds of
    GPU time and can only approximate what the file already states. OCR is the
    fallback for pages that have nothing to read, which is what makes `auto` safe
    to leave on: a born-digital PDF never loads the model at all.
    """
    pdf = pdfium.PdfDocument(path)
    total = len(pdf)
    scale = dpi / 72.0                            # PDF user space is 72 dpi
    shown = False                                 # did we print a progress line?
    try:
        for i in range(total):
            # `progress` prints an in-place counter, which is right for a
            # terminal and unreadable anywhere else. `on_page` is the same
            # news as a call, for callers that render rather than print.
            if on_page:
                on_page(i + 1, total)
            page = pdf[i]
            text = ""
            if ocr != OCR_ALWAYS:
                textpage = page.get_textpage()
                try:
                    text = textpage.get_text_bounded() or ""
                finally:
                    textpage.close()

            needs_ocr = (ocr == OCR_ALWAYS
                         or (ocr == OCR_AUTO and len(text.strip()) < min_chars))
            if needs_ocr:
                if progress:
                    print(f"\r    OCR page {i + 1}/{total}...", end="", flush=True)
                    shown = True
                # The model takes a file path, so the rendered page goes to a
                # temp PNG that is deleted as soon as it has been read.
                image = page.render(scale=scale).to_pil()
                with tempfile.TemporaryDirectory(prefix="uocr_page_") as tmp:
                    png = Path(tmp) / f"page_{i + 1:04d}.png"
                    image.save(png)
                    raw = _ocr_image(png, backend, prompt, quant, url)
                text = clean_ocr_text(raw) or text
                if figures:
                    # Captions are appended to the page's text so they flow
                    # through chunking like anything else. They are separated by
                    # a blank line, which reads as a paragraph break and keeps a
                    # caption from being packed into the same chunk as the prose
                    # sentence that happened to precede it.
                    captions = caption_figures(image, raw, i + 1, backend, quant,
                                               url, progress=progress)
                    if captions:
                        shown = True
                        text = "\n\n".join([text, *captions])
                        _OCR_STATS["figures"] += len(captions)

            yield i + 1, text
    finally:
        if shown:
            print("\r" + " " * 40 + "\r", end="")     # clear the progress line
        pdf.close()


def extract_pages(path, ocr=DEFAULT_OCR_MODE, backend=DEFAULT_OCR_BACKEND,
                  dpi=DEFAULT_OCR_DPI, min_chars=DEFAULT_OCR_MIN_CHARS,
                  prompt=DEFAULT_OCR_PROMPT, quant="auto", url=DEFAULT_OCR_URL,
                  figures=DEFAULT_OCR_FIGURES, progress=False, on_page=None):
    """Return [(page_number, text), ...] for a document (page numbers 1-based).

    PDFs are paginated; `.txt` files and images are single-"page" documents, so
    they come back as one entry, which keeps every caller on one shape.

    `on_page(page_number, total)` is called as each page is reached, for a caller
    that wants to show progress rather than print it. Single-page documents call
    it once, so a caller never has to special-case them.
    """
    suffix = Path(path).suffix.lower()

    if suffix == ".txt":
        if on_page:
            on_page(1, 1)
        return [(1, Path(path).read_text(encoding="utf-8", errors="replace"))]

    if suffix in IMAGE_EXTS:
        if ocr == OCR_NEVER:                      # an image has no text layer to fall back on
            return []
        if on_page:
            on_page(1, 1)
        return [(1, clean_ocr_text(
            _ocr_image(path, backend, prompt, quant, url)))]

    return list(_pdf_pages(path, ocr, backend, dpi, min_chars, prompt, quant,
                           url, progress, figures, on_page=on_page))


def extract_text(path, **kwargs):
    """Return the whole document's plain text, pages joined by blank lines.

    Accepts the same OCR keywords as extract_pages (`ocr`, `backend`, `dpi`,
    `min_chars`, `prompt`, `quant`, `url`).
    """
    return "\n\n".join(text for _, text in extract_pages(path, **kwargs))


def split_sentences(text):
    """Collapse line-wraps and split text into a list of sentence strings.

    Both sources arrive pre-broken: the PDF text layer carries a hard line break
    wherever a line wrapped on the page, and the OCR model emits markdown with
    its own line structure. Either way a single sentence is sprinkled with
    newlines. Collapsing all whitespace to single spaces rejoins those, then we
    split on sentence-ending punctuation (`_SENTENCE_BOUNDARY`). The result is
    the unit we pack into chunks, so chunk edges land at sentence ends instead of
    cutting mid-sentence.
    """
    text = " ".join(text.split())                 # rejoin wrapped lines -> one stream
    if not text:
        return []
    return _SENTENCE_BOUNDARY.split(text)


def _overlap_sentences(sentences, overlap):
    """Return the trailing whole sentences whose length is ~<= overlap chars.

    Carries context from the end of one chunk into the start of the next without
    ever beginning on a mid-word fragment. Always keeps at least the last
    sentence so consecutive chunks share a real boundary sentence.
    """
    if overlap <= 0:
        return []
    tail, total = [], 0
    for s in reversed(sentences):
        if tail and total + len(s) > overlap:
            break
        tail.insert(0, s)
        total += len(s) + 1

    # The loop above accepts the final sentence at *any* length, because the size
    # guard only applies once `tail` is non-empty -- the intent being that
    # consecutive chunks always share a real boundary sentence. That intent
    # breaks down on text with no sentence punctuation: a flattened table is one
    # enormous pseudo-sentence, `_split_oversized` cuts it into max_chars word
    # pieces, and carrying such a piece whole duplicates an entire chunk. Measured
    # on the annual report before this clip: 26% of all stored text was duplicated,
    # with 129 chunk pairs sharing over 400 characters and some sharing all 600.
    # That is wasted embedding cost, near-identical vectors crowding out distinct
    # results in search, and phantom density that skews clustering.
    #
    # So an over-budget lone carry is clipped to its trailing `overlap` characters,
    # on a word boundary. The bridge between chunks survives; the duplication does not.
    if len(tail) == 1 and len(tail[0]) > overlap:
        kept, used = [], 0
        for word in reversed(tail[0].split()):
            if kept and used + len(word) + 1 > overlap:
                break
            kept.insert(0, word)
            used += len(word) + 1
        return [" ".join(kept)] if kept else []
    return tail


def _split_oversized(piece, max_chars):
    """Break a single sentence that exceeds max_chars into <=max_chars pieces.

    Splits on whitespace so words stay intact; a lone word longer than max_chars
    (rare — URLs, hashes) is hard-cut as a last resort. Only hit by the
    occasional monster "sentence" (a table row or equation with no punctuation).
    """
    out, cur = [], ""
    for word in piece.split():
        if len(word) > max_chars:                 # word alone too big: hard-cut it
            if cur:
                out.append(cur)
                cur = ""
            for j in range(0, len(word), max_chars):
                out.append(word[j:j + max_chars])
            continue
        candidate = f"{cur} {word}".strip()
        if len(candidate) > max_chars:
            out.append(cur)
            cur = word
        else:
            cur = candidate
    if cur:
        out.append(cur)
    return out


_TABLE_BLOCK = re.compile(r"<table\b[^>]*>.*?</table\s*>", re.S | re.I)
_TABLE_ROW = re.compile(r"<tr\b[^>]*>(.*?)</tr\s*>", re.S | re.I)
_TABLE_CELL = re.compile(r"<(t[dh])\b([^>]*)>(.*?)</\1\s*>", re.S | re.I)
_SPAN_ATTR = re.compile(r"\b(rowspan|colspan)\s*=\s*[\"']?(\d+)", re.I)

# A malformed span (rowspan="9999") would otherwise allocate a grid that large.
# Real document tables never approach this.
MAX_SPAN = 64
_HTML_TAG = re.compile(r"<[^>]+>")

# Table tags that reach the prose path, which means they were never closed --
# generation hit `max_length` partway through a table, so `</table>` never
# arrived and `_TABLE_BLOCK` could not match. The cell *text* is still real page
# content and worth keeping; only the orphaned markup is noise. Deliberately
# narrow: a general `<[^>]+>` strip would also eat legitimate prose like "a < b >
# c", and truncated tables are rare enough not to justify that risk.
_ORPHAN_TABLE_TAG = re.compile(r"</?(?:table|tr|td|th)\b[^>]*>", re.I)


def _cell_text(html):
    """Strip markup out of one cell and collapse its whitespace."""
    return " ".join(_HTML_TAG.sub(" ", html).split())


def _spans(attrs):
    """Return (colspan, rowspan) from a cell's attribute string."""
    found = {k.lower(): int(v) for k, v in _SPAN_ATTR.findall(attrs or "")}
    return (min(max(found.get("colspan", 1), 1), MAX_SPAN),
            min(max(found.get("rowspan", 1), 1), MAX_SPAN))


def _expand_grid(html):
    """Expand a table's rowspan/colspan into a rectangular grid of cell texts.

    Annual-report tables stack their headers, so the model emits things like
    `<td rowspan="3">Name of company</td>` — one cell standing over three rows
    while other columns carry sub-headers beneath. Read naively, row 0 has far
    fewer cells than the data rows and every column lines up against the wrong
    label. Expanding spans into a real grid puts each value back under the header
    that actually governs it.

    A spanned cell's text is repeated into every slot it covers, which is what
    makes rows the same width and lets header rows be merged column-wise.
    Returns [] for a table with no readable cells.
    """
    grid = []
    spill = {}                                    # (row, col) -> text held by a rowspan
    for r, row_html in enumerate(_TABLE_ROW.findall(html)):
        row, col = [], 0
        for _tag, attrs, body in _TABLE_CELL.findall(row_html):
            while (r, col) in spill:              # slot already claimed from above
                row.append(spill.pop((r, col)))
                col += 1
            text = _cell_text(body)
            colspan, rowspan = _spans(attrs)
            for _ in range(colspan):
                row.append(text)
                for dr in range(1, rowspan):
                    spill[(r + dr, col)] = text
                col += 1
        while (r, col) in spill:                  # trailing rowspan columns
            row.append(spill.pop((r, col)))
            col += 1
        grid.append(row)

    width = max((len(row) for row in grid), default=0)
    return [row + [""] * (width - len(row)) for row in grid if any(row)]


def _header_depth(html, grid):
    """How many leading rows of `grid` form the header block.

    Two signals, whichever reaches deeper. A `<th>` row is an explicit header.
    Failing that, the rowspan on the first row's cells says it directly: a cell
    declared `rowspan="3"` in row 0 is stating that the header is three rows
    tall. Always leaves at least one data row, so a table that is nothing but
    headers is treated as data rather than yielding nothing.
    """
    rows_html = _TABLE_ROW.findall(html)
    if not rows_html or len(grid) < 2:
        return 0

    th_rows = 0
    for row_html in rows_html:
        tags = [tag.lower() for tag, _, _ in _TABLE_CELL.findall(row_html)]
        if tags and all(tag == "th" for tag in tags):
            th_rows += 1
        else:
            break

    span_depth = max((_spans(attrs)[1]
                      for _tag, attrs, _body in _TABLE_CELL.findall(rows_html[0])),
                     default=1)

    return max(1, min(max(th_rows, span_depth), len(grid) - 1))


def _merge_headers(header_rows):
    """Collapse stacked header rows into one composite label per column.

    Column-wise top to bottom, skipping blanks and values already seen in that
    column — a rowspan repeats its text down every row it covers, so without the
    dedupe every label would stutter ("Name of company - Name of company").
    """
    headers = []
    for column in zip(*header_rows):
        parts = []
        for value in column:
            value = value.strip()
            if value and value not in parts:
                parts.append(value)
        headers.append(" - ".join(parts))
    return headers


def parse_table(html):
    """Return (headers, rows) for an HTML table. `headers` is [] when unclear.

    Spans are expanded first so every row is the same width, then the leading
    header rows are merged into composite labels. A single-row table is all data
    — there is nothing for a header to label.
    """
    grid = _expand_grid(html)
    if len(grid) < 2:
        return [], grid

    depth = _header_depth(html, grid)
    headers = _merge_headers(grid[:depth])
    if not any(headers):
        return [], grid
    return headers, grid[depth:]


def serialize_table(html, caption=""):
    """Turn a table into one self-contained sentence per data row.

    `<td>Nintendo of America Inc.</td><td>Washington, USA</td>` under headers
    `Name of company | Location` becomes

        "Consolidated subsidiaries — Name of company: Nintendo of America Inc.;
         Location: Washington, USA."

    Pairing each value with its header is what makes a row retrievable on its
    own. A bare row of cells embeds as a list of proper nouns and numbers that
    matches almost nothing; "Location: Washington, USA" actually answers "where
    is the subsidiary based". Repeating the caption on every row means a chunk
    retrieved in isolation still says which table it came from.

    Rows whose cell count does not match the header count fall back to the plain
    cell values. That happens with the `rowspan`/`colspan` headers this model
    emits, and a wrong header pairing would be worse than none.
    """
    headers, rows = parse_table(html)
    lines = []
    for cells in rows:
        if headers and len(cells) == len(headers):
            pairs = [f"{h}: {c}" for h, c in zip(headers, cells) if c and h]
        else:
            pairs = [c for c in cells if c]
        # A colspan repeats one value across several columns, so the same
        # header:value pair can appear more than once in a row. Keep first order.
        parts = list(dict.fromkeys(pairs))
        if not parts:
            continue
        line = "; ".join(parts)
        # ASCII separator on purpose: an em dash costs 3 UTF-8 bytes against the
        # VARCHAR budget on every row and reads no better once embedded.
        lines.append(f"{caption} - {line}." if caption else f"{line}.")
    return lines


def _trailing_caption(prose, limit=100):
    """Use the last sentence before a table as that table's caption.

    Document-parsing output puts the heading ("(1) Consolidated subsidiaries")
    immediately before the markup, so the trailing sentence is normally exactly
    the label a reader would give the table.
    """
    sentences = split_sentences(prose)
    return sentences[-1].strip()[:limit].rstrip() if sentences else ""


def _table_fits(candidate, max_chars):
    """True when `candidate` is within both the char budget and Milvus' field.

    Milvus stores `text` as VARCHAR(2048) and that limit counts **bytes**, not
    characters. This corpus is full of multi-byte characters (¥, curly quotes),
    so a chunk that looks short enough by `len()` can still overflow the field.
    Table chunks get a bigger character budget than prose because splitting rows
    apart is what we are specifically trying to avoid, which makes the byte check
    load-bearing rather than theoretical.
    """
    return (len(candidate) <= max(max_chars, TABLE_MAX_CHARS)
            and len(candidate.encode("utf-8")) <= VARCHAR_BYTE_BUDGET)


def _chunk_table(html, caption, max_chars):
    """Pack serialized rows into chunks, never splitting a row across chunks."""
    lines = serialize_table(html, caption)
    if not lines:
        return []

    chunks, current = [], []
    for line in lines:
        candidate = " ".join(current + [line])
        if current and not _table_fits(candidate, max_chars):
            chunks.append(" ".join(current))
            current = []
        # A single row over budget is the one case we must break up; keeping it
        # whole would exceed the VARCHAR and the insert would fail outright.
        if not current and not _table_fits(line, max_chars):
            chunks.extend(_split_oversized(line, max_chars))
            continue
        current.append(line)
    if current:
        chunks.append(" ".join(current))
    return chunks


def _chunk_prose(text, max_chars=DEFAULT_MAX_CHARS, overlap=DEFAULT_OVERLAP):
    """Split prose into <=~max_chars chunks that never cut a sentence in half.

    Any orphaned table markup is dropped first — see `_ORPHAN_TABLE_TAG`.

    Whole sentences are packed together until the next one would overflow
    max_chars; the chunk is then flushed and the next one seeded with the
    previous chunk's trailing sentence(s) for `overlap` chars of context. A lone
    sentence longer than max_chars is word-split as a fallback. Keeping edges at
    sentence boundaries means each chunk is a coherent semantic unit, so its
    embedding (and the clusters built from it) reflect one topic, not a fragment.
    """
    text = _ORPHAN_TABLE_TAG.sub(" ", text)

    # Split into sentence units, word-splitting any single one that's too long.
    sentences = []
    for s in split_sentences(text):
        if len(s) > max_chars:
            sentences.extend(_split_oversized(s, max_chars))
        else:
            sentences.append(s)

    chunks, cur = [], []
    for s in sentences:
        if cur and len(" ".join(cur + [s])) > max_chars:
            chunks.append(" ".join(cur))
            cur = _overlap_sentences(cur, overlap)   # carry whole sentences over
        cur.append(s)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def chunk_text(text, max_chars=DEFAULT_MAX_CHARS, overlap=DEFAULT_OVERLAP):
    """Split text into chunks, routing tables and prose to different packers.

    Unlimited-OCR returns tables as real markup, which is the main thing it buys
    over classic OCR. Feeding that markup through the prose packer would waste it
    twice over: the sentence splitter finds no boundaries inside `<td>` cells, so
    a wide table is cut at an arbitrary 600th character -- and half a table is
    worse than none, because the surviving cells lose the header that says what
    they are. The tags themselves also dilute the embedding, since `rowspan` and
    `td` carry no meaning about subsidiaries or share capital.

    So tables are pulled out, turned into one self-contained sentence per row,
    and packed on row boundaries; everything between them goes through
    `_chunk_prose` exactly as before. Text with no tables in it -- every
    `--ocr never` run, and the whole corpus before OCR existed -- takes the same
    path it always did and chunks identically.
    """
    text = text or ""
    chunks, position, caption = [], 0, ""

    for match in _TABLE_BLOCK.finditer(text):
        prose = text[position:match.start()]
        if prose.strip():
            chunks.extend(_chunk_prose(prose, max_chars, overlap))
            caption = _trailing_caption(prose)
        chunks.extend(_chunk_table(match.group(0), caption, max_chars))
        position = match.end()

    tail = text[position:]
    if tail.strip():
        chunks.extend(_chunk_prose(tail, max_chars, overlap))
    return [c for c in chunks if len(c.strip()) >= MIN_CHUNK_CHARS]


def extract_chunks(path, max_chars=DEFAULT_MAX_CHARS, overlap=DEFAULT_OVERLAP, **kwargs):
    """File path -> list of chunk strings ready to embed/store.

    Extra keywords are passed through to extract_pages (`ocr`, `backend`, ...).
    """
    return chunk_text(extract_text(path, **kwargs),
                      max_chars=max_chars, overlap=overlap)


def find_documents(folder):
    """Return the supported document files directly in `folder`, sorted by name."""
    base = Path(folder)
    if not base.is_dir():
        raise SystemExit(f"Input folder not found: {folder}")
    return sorted(
        p for p in base.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    )


def extract_records(path, max_chars=DEFAULT_MAX_CHARS, overlap=DEFAULT_OVERLAP,
                    on_file=None, **kwargs):
    """Extract chunks from a file OR every supported file in a folder.

    Returns a list of {"text": chunk, "source": filename} records. A folder is
    walked one level deep over SUPPORTED_EXTS. `source` is the file name so each
    chunk stays traceable to its origin once stored. Extra keywords go to
    extract_pages (`ocr`, `backend`, `dpi`, `min_chars`, `prompt`, ...) --
    including `on_page`, which is how a caller follows a long file.

    `on_file(index, total, path)` is called before each file is opened, so a
    caller knows the size of the job from the first callback rather than having
    to walk the folder itself first.

    A file yielding no text now means something is genuinely wrong with it —
    encrypted, corrupt, or blank — rather than "it was scanned", which OCR
    handles. The message says so.
    """
    target = Path(path)
    files = [target] if target.is_file() else find_documents(target)
    if not files:
        raise SystemExit(
            f"No supported documents in {path} "
            f"(looked for: {', '.join(sorted(SUPPORTED_EXTS))}).")

    records = []
    for index, f in enumerate(files, start=1):
        if on_file:
            on_file(index, len(files), f)
        before = _OCR_STATS["pages"]
        try:
            chunks = extract_chunks(f, max_chars=max_chars, overlap=overlap,
                                    progress=True, **kwargs)
        except Exception as e:                    # one bad file shouldn't sink the batch
            print(f"  ! skipped {f.name}: {e}")
            continue
        if not chunks:
            print(f"  ! {f.name}: no text extracted "
                  f"(encrypted, corrupt, or genuinely blank?)")
            continue
        ocr_pages = _OCR_STATS["pages"] - before
        note = f", {ocr_pages} page(s) OCR'd" if ocr_pages else ""
        print(f"  + {f.name}: {len(chunks)} chunks{note}")
        # `seq` is the chunk's position in reading order. Milvus has no implicit
        # row order -- queries come back in segment order, and the incremental
        # store reuses cached rows while appending new ones, so auto-ids do not
        # track the document either. Without an explicit index, "the third table
        # row" is unrecoverable once stored, which is what makes a plot's hover
        # text look shuffled even though extraction was perfectly in order.
        records.extend({"text": c, "source": f.name, "seq": i}
                       for i, c in enumerate(chunks))
    return records


def write_input_md(records, path="input.md"):
    """Write chunk records as bullet documents under a '# Documents' heading.

    Produces a file loadmilvus.py can read directly. Newlines inside a chunk are
    flattened to spaces so each chunk stays one bullet line (load_inputs treats
    every non-empty line as a separate document). The `source` is dropped here —
    input.md is a plain text-only format — but is kept when storing to Milvus.
    """
    lines = ["# Documents", ""]
    lines += [f"- {' '.join(r['text'].split())}" for r in records]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(records)} document chunks to {path}.")


def show_chunks(records, preview=160):
    """Print a short preview of each chunk (index, source, length, leading text)."""
    print(f"\nExtracted {len(records)} chunks total:")
    for i, r in enumerate(records):
        head = " ".join(r["text"].split())[:preview]
        ellipsis = "..." if len(r["text"]) > preview else ""
        print(f"\n[{i}] {r['source']} ({len(r['text'])} chars)")
        print(f"    {head}{ellipsis}")


def parse_value_arg(argv, flag):
    """Read the value following `flag` in argv, or None if the flag is absent."""
    if flag in argv:
        i = argv.index(flag)
        if i + 1 >= len(argv):
            raise SystemExit(f"{flag} needs a value")
        return argv[i + 1]
    return None


FLAGS_WITH_VALUES = {"--out", "--max-chars", "--overlap", "--model", "--ocr",
                     "--ocr-backend", "--ocr-dpi", "--ocr-min-chars",
                     "--ocr-prompt", "--ocr-quant", "--ocr-url"}


def parse_path_arg(argv, default):
    """Return the first positional (non-flag) argument, or `default` if none."""
    skip = False
    for arg in argv[1:]:
        if skip:
            skip = False
            continue
        if arg in FLAGS_WITH_VALUES:
            skip = True
            continue
        if arg.startswith("--"):
            continue
        return arg
    return default


def parse_ocr_args(argv):
    """Read the OCR flags off argv into the keywords extract_records expects."""
    mode = parse_value_arg(argv, "--ocr") or DEFAULT_OCR_MODE
    if mode not in OCR_MODES:
        raise SystemExit(f"--ocr must be one of: {', '.join(OCR_MODES)}")
    quant = parse_value_arg(argv, "--ocr-quant") or "auto"
    if quant not in ("auto", "none", "4bit", "8bit"):
        raise SystemExit("--ocr-quant must be one of: auto, none, 4bit, 8bit")
    return {
        "ocr": mode,
        "backend": parse_value_arg(argv, "--ocr-backend") or DEFAULT_OCR_BACKEND,
        "dpi": int(parse_value_arg(argv, "--ocr-dpi") or DEFAULT_OCR_DPI),
        "min_chars": int(parse_value_arg(argv, "--ocr-min-chars")
                         or DEFAULT_OCR_MIN_CHARS),
        "prompt": parse_value_arg(argv, "--ocr-prompt") or DEFAULT_OCR_PROMPT,
        "quant": quant,
        "url": parse_value_arg(argv, "--ocr-url") or DEFAULT_OCR_URL,
        "figures": "--ocr-figures" in argv,
    }


def store_records(records, model_name, reset=False):
    """Embed the records Milvus doesn't already have, and store them.

    Embedding is the expensive stage (a full fileinput/ run is ~1000 chunks), so
    it is skipped wherever possible. Each chunk's (model, text) hash is its cache
    key; any hash already in the collection was embedded on an earlier run and
    its vector is reused as-is. Adding one PDF to fileinput/ therefore only costs
    the embedding of that PDF's chunks, not a re-embed of the whole folder.

    Note OCR output is part of that hash, so re-running with a different `--ocr`
    mode, prompt, or DPI produces different text and therefore re-embeds the
    affected chunks. That is correct — it is different text — but it means
    changing OCR settings is not free. VLM decoding is not bit-deterministic
    across driver or quantization changes either, so expect some churn.

    The collection still ends up mirroring the input exactly, as the old
    reset=True behavior did: chunks that vanished from the input (their file was
    edited or removed) are deleted. Pass reset=True to force a full rebuild.
    """
    # Imported lazily so plain extraction/preview doesn't pull in pymilvus.
    from loadmilvus import (DEFAULT_COLLECTION, connect, content_hash, embed,
                            ensure_collection, fetch_cached, get_dim, get_model,
                            insert_batched, resolve_model)

    # The cache key is the model's *id*, which resolve_model gives us without
    # loading any weights -- so the whole probe below (hash, look up, diff) runs
    # before the model is touched. On a full cache hit we return without ever
    # paying for it. Using the resolved id, not `model_name`, means the `bge-m3`
    # alias and its full HF id hash alike instead of missing each other.
    hf_id = resolve_model(model_name)[0]
    for r in records:
        r["chunk_hash"] = content_hash(hf_id, r["text"])
    wanted = {r["chunk_hash"] for r in records}

    client = connect()
    cached = {} if reset else fetch_cached(client, DEFAULT_COLLECTION)

    # Rows whose chunk is no longer in the input folder, and chunks the cache
    # doesn't have. `fresh` is deduped by hash, so a chunk repeated across (or
    # within) files is embedded and stored once.
    stale = [row_id for h, row_id in cached.items() if h not in wanted]
    fresh = {}
    for r in records:
        if r["chunk_hash"] not in cached:
            fresh.setdefault(r["chunk_hash"], r)

    hits = len(wanted) - len(fresh)
    print(f"{hits}/{len(wanted)} chunks already embedded (cache hit); "
          f"{len(fresh)} to embed, {len(stale)} to drop.")
    if not reset and not fresh and not stale:
        print("Collection already matches the input — nothing to do.")
        return 0

    # Only now, with real work to do, do we pay to load the model.
    model = get_model(model_name)
    ensure_collection(client, dim=get_dim(model), reset=reset)

    if stale:
        client.delete(collection_name=DEFAULT_COLLECTION, ids=stale)
        print(f"Dropped {len(stale)} chunks no longer present in the input.")
    if not fresh:
        client.flush(DEFAULT_COLLECTION)
        return 0

    new_records = list(fresh.values())
    embeddings = embed(model, [r["text"] for r in new_records])
    rows = [
        {"text": r["text"], "embedding": emb.tolist(),
         "source": r["source"], "seq": r.get("seq", -1),
         "chunk_hash": r["chunk_hash"]}
        for r, emb in zip(new_records, embeddings)
    ]
    count = insert_batched(client, rows, DEFAULT_COLLECTION)   # also flushes
    print(f"Inserted {count} new chunks "
          f"from {len({r['source'] for r in records})} files.")
    return count


def main():
    """Extract files into chunks, then preview / write to input.md / store."""
    # Extracted text can hold characters outside the terminal's legacy code page
    # (e.g. math symbols like pi from a PDF). Don't let a preview print crash the
    # run on Windows consoles (cp1252); replace anything unencodable.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    path = parse_path_arg(sys.argv, default=DEFAULT_INPUT_DIR)
    if not Path(path).exists():
        raise SystemExit(f"Path not found: {path}")

    max_chars = int(parse_value_arg(sys.argv, "--max-chars") or DEFAULT_MAX_CHARS)
    overlap = int(parse_value_arg(sys.argv, "--overlap") or DEFAULT_OVERLAP)
    ocr_kwargs = parse_ocr_args(sys.argv)

    where = f"folder {path}/" if Path(path).is_dir() else path
    engine = ("pypdfium2 (text layer only)" if ocr_kwargs["ocr"] == OCR_NEVER
              else f"pypdfium2 + {ocr_kwargs['backend']} ({ocr_kwargs['ocr']}, "
                   f"{ocr_kwargs['dpi']} dpi"
                   + (", figures" if ocr_kwargs["figures"] else "") + ")")
    print(f"Extracting text from {where} with {engine}...")

    records = extract_records(path, max_chars=max_chars, overlap=overlap,
                              **ocr_kwargs)
    if not records:
        raise SystemExit("No text extracted from any file.")

    if _OCR_STATS["pages"]:
        per_page = _OCR_STATS["seconds"] / _OCR_STATS["pages"]
        figures = (f", {_OCR_STATS['figures']} figure(s) described"
                   if _OCR_STATS["figures"] else "")
        print(f"OCR ran on {_OCR_STATS['pages']} page(s) in "
              f"{_OCR_STATS['seconds']:.1f}s ({per_page:.1f}s/page){figures}.")
    show_chunks(records)

    out = parse_value_arg(sys.argv, "--out")
    if out:
        write_input_md(records, out)

    if "--store" in sys.argv:
        from loadmilvus import parse_model_arg
        store_records(records, parse_model_arg(sys.argv),
                      reset="--reset" in sys.argv)
        print("\nDone. Chunks embedded and stored in Milvus.")
    elif not out:
        print("\nPreview only. Re-run with --out input.md to save, or "
              "--store to embed into Milvus.")


if __name__ == "__main__":
    main()
