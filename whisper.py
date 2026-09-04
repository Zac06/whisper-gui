#!/usr/bin/env python3
"""
Faster-Whisper GUI (Tkinter Edition, self-bootstrapping)
==========================================================

A desktop front-end for faster-whisper, built entirely on Python's standard
library for the UI layer (tkinter/ttk). No Qt, no platform icon themes, no
dependency on the OS's dark-mode / icon-theme plumbing -- the whole interface
draws its own fixed color palette, so it looks the same everywhere and can't
throw errors like "kf.iconthemes: Icon theme not found" (that was a
KDE/Plasma-integration warning coming from PyQt/KDE Frameworks, which this
rewrite has no contact with at all).

Fully autonomous startup
-------------------------
This script manages its own dependencies. On every run it:
  1. Makes sure a private virtual environment exists (created next to this
     file, or under a per-user data directory if that's not writable), so it
     works out of the box on "externally managed" systems (PEP 668 /
     Debian's `error: externally-managed-environment`) without ever touching
     the system Python or requiring --break-system-packages.
  2. Verifies the required third-party packages are importable inside that
     venv, and installs/repairs them with the venv's own pip if not.
  3. Re-executes itself using the venv's Python interpreter.
  4. At runtime, if `ffmpeg` isn't found on PATH, it transparently falls
     back to a portable build fetched via the `static-ffmpeg` package.

The only thing this script cannot install for you is Tk/tkinter itself --
that's a compiled extension tied to your system Python, not a pip package.
If it's missing, the script tells you the exact command to install it
(e.g. `sudo apt install python3-tk`) and exits.

Command-line flags (all optional)
----------------------------------
    --with-torch     Also install PyTorch inside the venv (only used to
                      improve CUDA/VRAM detection; nvidia-smi is used as a
                      fallback when torch isn't present, so this is rarely
                      necessary).
    --reinstall-deps Force a fresh `pip install` of all dependencies even if
                      they already appear importable.
    --reset-venv     Delete and recreate the virtual environment from
                      scratch, then reinstall dependencies.

Features
--------
- "Ability" (model size) and "Precision" (compute_type) sliders, with live
  VRAM/RAM estimates and an "Auto-Suggest" button based on detected hardware.
- CPU / GPU device selection.
- Source input: local file (audio or video) OR a YouTube/direct URL, which
  is downloaded and extracted to audio via yt-dlp.
- Language selection (including "auto").
- Three separate, purpose-built steering controls instead of one vague
  "prompt" box, matching what faster-whisper actually exposes:
    * initial_prompt -- a *style example* (tone/vocabulary/spelling bias),
      up to ~224 tokens, chosen via a named style preset or a custom entry.
    * prefix         -- text forced verbatim at the very start of the
      transcript's first segment.
    * hotwords        -- words whose log-probabilities are boosted during
      decoding, without consuming any prompt/context budget.
- VAD (Voice Activity Detection) filter checkbox with a custom hover tooltip.
- Interface available in English and Italian (Italian by default).
- Two-column, scrollable layout with a self-contained dark theme.
- Stop button to cancel an in-progress download or transcription.

Run
---
    python3 whisper_gui_tk.py
"""

import os
import sys

# ------------------------------------------------------------------------
# Self-bootstrapping virtual environment
# ------------------------------------------------------------------------
# Everything in this section uses ONLY the standard library, on purpose:
# it has to work before any third-party package is known to be installed,
# and it has to work with whatever system Python first launches the script.

import subprocess
import venv as _venv_mod
from pathlib import Path

_BOOTSTRAP_MARKER = "WHISPER_GUI_TK_VENV_ACTIVE"
_REQUIRED_PACKAGES = ["faster-whisper", "yt-dlp", "psutil", "static-ffmpeg"]
_OPTIONAL_TORCH_PACKAGE = "torch"


def _default_venv_dir() -> Path:
    """Pick a writable location for the managed venv: next to this script
    if that directory is writable, otherwise a per-user data directory.
    Overridable via the WHISPER_GUI_TK_VENV_DIR environment variable."""
    override = os.environ.get("WHISPER_GUI_TK_VENV_DIR")
    if override:
        return Path(override).expanduser().resolve()

    script_dir = Path(__file__).resolve().parent
    if os.access(script_dir, os.W_OK):
        return script_dir / ".whisper_gui_venv"

    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "whisper-gui-tk" / "venv"


def _venv_python_path(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _check_tkinter_or_exit():
    try:
        import tkinter  # noqa: F401
    except ImportError:
        sys.stderr.write(
            "\nTkinter is not available in this Python installation.\n"
            "This is a compiled component tied to your system Python, so it "
            "can't be installed via pip / a virtual environment -- it has to "
            "come from your OS package manager. Try one of:\n"
            "    Debian/Ubuntu:  sudo apt install python3-tk\n"
            "    Fedora:         sudo dnf install python3-tkinter\n"
            "    Arch:           sudo pacman -S tk\n"
            "    macOS (brew):   brew install python-tk\n"
            "Then re-run this script.\n\n"
        )
        sys.exit(1)


def _create_venv(venv_dir: Path):
    print(f"[bootstrap] Creating virtual environment at: {venv_dir}")
    try:
        venv_dir.parent.mkdir(parents=True, exist_ok=True)
        _venv_mod.create(str(venv_dir), with_pip=True, upgrade_deps=True)
    except Exception as e:
        sys.stderr.write(
            f"\n[bootstrap] Failed to create the virtual environment: {e}\n"
            "On Debian/Ubuntu, venv creation needs a separate package:\n"
            "    sudo apt install python3-venv\n"
            "Then re-run this script.\n\n"
        )
        sys.exit(1)


def _deps_importable(venv_python: Path, with_torch: bool) -> bool:
    modules = ["faster_whisper", "yt_dlp", "psutil", "static_ffmpeg"]
    if with_torch:
        modules.append("torch")
    code = "import " + ", ".join(modules)
    result = subprocess.run([str(venv_python), "-c", code],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return result.returncode == 0


def _install_deps(venv_python: Path, with_torch: bool):
    packages = list(_REQUIRED_PACKAGES)
    if with_torch:
        packages.append(_OPTIONAL_TORCH_PACKAGE)
    print("[bootstrap] Installing dependencies inside the virtual environment:")
    print("            " + ", ".join(packages))
    subprocess.run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"], check=False)
    result = subprocess.run([str(venv_python), "-m", "pip", "install", "--upgrade", *packages])
    if result.returncode != 0:
        sys.stderr.write(
            "\n[bootstrap] Dependency installation failed. Check your network "
            "connection and the pip output above, then re-run this script.\n\n"
        )
        sys.exit(1)


def _bootstrap():
    """Ensure a managed venv exists with the needed packages, then re-exec
    this script inside it. Returns immediately (a no-op) once already
    running inside that venv, via the marker environment variable."""
    if os.environ.get(_BOOTSTRAP_MARKER) == "1":
        return  # already inside the managed venv -- nothing left to do

    _check_tkinter_or_exit()

    args = sys.argv[1:]
    with_torch = "--with-torch" in args
    reinstall = "--reinstall-deps" in args
    reset_venv = "--reset-venv" in args

    venv_dir = _default_venv_dir()
    venv_python = _venv_python_path(venv_dir)

    if reset_venv and venv_dir.exists():
        import shutil as _shutil
        print(f"[bootstrap] Removing existing virtual environment: {venv_dir}")
        _shutil.rmtree(venv_dir, ignore_errors=True)

    if not venv_python.exists():
        _create_venv(venv_dir)
        _install_deps(venv_python, with_torch)
    elif reinstall or not _deps_importable(venv_python, with_torch):
        _install_deps(venv_python, with_torch)

    script_path = str(Path(__file__).resolve())
    env = os.environ.copy()
    env[_BOOTSTRAP_MARKER] = "1"
    try:
        os.execve(str(venv_python), [str(venv_python), script_path, *args], env)
    except OSError as e:
        sys.stderr.write(f"\n[bootstrap] Failed to relaunch inside the virtual environment: {e}\n\n")
        sys.exit(1)


_bootstrap()

# ------------------------------------------------------------------------
# From this point on we are guaranteed to be running inside the managed
# venv, with all required third-party packages installed.
# ------------------------------------------------------------------------

import shutil
import threading
import queue
import traceback
from datetime import datetime

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter import scrolledtext

# ------------------------------------------------------------------------
# Optional dependencies (detected gracefully; torch is only installed if
# --with-torch was passed, so it's still guarded here even post-bootstrap)
# ------------------------------------------------------------------------
try:
    import psutil
except ImportError:
    psutil = None

try:
    import torch
except ImportError:
    torch = None


def ensure_ffmpeg_available() -> bool:
    """Best-effort: if ffmpeg isn't on PATH, fetch a portable build via the
    static-ffmpeg package (installed automatically by the bootstrapper) and
    add it to PATH for this process."""
    if shutil.which("ffmpeg"):
        return True
    try:
        import static_ffmpeg
        static_ffmpeg.add_paths()
    except Exception:
        return False
    return shutil.which("ffmpeg") is not None


# ------------------------------------------------------------------------
# Static data: models, compute types, languages, prompt presets
# ------------------------------------------------------------------------

MODEL_TABLE = [
    dict(name="tiny",              label="Tiny",               params="39M",   base_gb=1.0,
         desc_en="Fastest, lowest accuracy. Good for quick drafts.",
         desc_it="Il più veloce, la precisione più bassa. Buono per bozze rapide."),
    dict(name="tiny.en",           label="Tiny (English-only)", params="39M",   base_gb=1.0,
         desc_en="English-only tiny model, marginally better for pure EN audio.",
         desc_it="Modello tiny solo inglese, leggermente migliore per audio in inglese puro."),
    dict(name="base",              label="Base",               params="74M",   base_gb=1.0,
         desc_en="Small step up from tiny, still very fast.",
         desc_it="Un piccolo passo avanti rispetto a tiny, ancora molto veloce."),
    dict(name="base.en",           label="Base (English-only)", params="74M",   base_gb=1.0,
         desc_en="English-only base model.",
         desc_it="Modello base solo inglese."),
    dict(name="small",             label="Small",              params="244M",  base_gb=2.0,
         desc_en="Good balance for lightweight machines.",
         desc_it="Buon equilibrio per macchine leggere."),
    dict(name="small.en",          label="Small (English-only)", params="244M", base_gb=2.0,
         desc_en="English-only small model.",
         desc_it="Modello small solo inglese."),
    dict(name="distil-small.en",   label="Distil-Small (EN)",  params="166M",  base_gb=1.5,
         desc_en="Distilled, faster than small, English-only.",
         desc_it="Distillato, più veloce di small, solo inglese."),
    dict(name="medium",            label="Medium",             params="769M",  base_gb=5.0,
         desc_en="High accuracy, moderate resource needs.",
         desc_it="Alta precisione, esigenze di risorse moderate."),
    dict(name="medium.en",         label="Medium (English-only)", params="769M", base_gb=5.0,
         desc_en="English-only medium model.",
         desc_it="Modello medium solo inglese."),
    dict(name="distil-medium.en",  label="Distil-Medium (EN)", params="394M",  base_gb=3.0,
         desc_en="Distilled medium, faster, English-only.",
         desc_it="Medium distillato, più veloce, solo inglese."),
    dict(name="distil-large-v3",   label="Distil-Large-v3",    params="756M",  base_gb=4.5,
         desc_en="Distilled large model, close to large accuracy, notably faster.",
         desc_it="Modello large distillato, precisione vicina a large, molto più veloce."),
    dict(name="large-v3-turbo",    label="Large-v3 Turbo",     params="809M",  base_gb=6.0,
         desc_en="Pruned large-v3, fast with near large-v3 accuracy.",
         desc_it="Versione ridotta di large-v3, veloce con precisione quasi identica."),
    dict(name="large-v2",          label="Large-v2",           params="1550M", base_gb=10.0,
         desc_en="Very high accuracy, resource-hungry.",
         desc_it="Precisione molto alta, richiede molte risorse."),
    dict(name="large-v3",          label="Large-v3",           params="1550M", base_gb=10.0,
         desc_en="Best general accuracy, most resource-hungry.",
         desc_it="Migliore precisione generale, la più esigente in risorse."),
]

COMPUTE_TYPES_GPU = [
    dict(name="int8",         label_en="Int8 — fastest, lowest quality",        label_it="Int8 — più veloce, qualità più bassa",        mult=0.50),
    dict(name="int8_float16", label_en="Int8-Float16 — fast, good quality",     label_it="Int8-Float16 — veloce, buona qualità",        mult=0.55),
    dict(name="float16",      label_en="Float16 — recommended balance",         label_it="Float16 — equilibrio consigliato",            mult=1.00),
    dict(name="bfloat16",     label_en="BFloat16 — balanced (Ampere+ GPUs)",    label_it="BFloat16 — equilibrato (GPU Ampere+)",        mult=1.00),
    dict(name="float32",      label_en="Float32 — highest precision, slowest", label_it="Float32 — massima precisione, più lento",     mult=2.00),
]

COMPUTE_TYPES_CPU = [
    dict(name="int8",         label_en="Int8 — fastest, lowest quality",     label_it="Int8 — più veloce, qualità più bassa",   mult=0.50),
    dict(name="int8_float32", label_en="Int8-Float32 — balanced",            label_it="Int8-Float32 — equilibrato",             mult=0.60),
    dict(name="float32",      label_en="Float32 — highest precision, slowest", label_it="Float32 — massima precisione, più lento", mult=1.20),
]

LANGUAGES = [
    ("auto", "Auto-detect", "Rilevamento automatico"), ("en", "English", "Inglese"),
    ("es", "Spanish", "Spagnolo"), ("fr", "French", "Francese"), ("de", "German", "Tedesco"),
    ("it", "Italian", "Italiano"), ("pt", "Portuguese", "Portoghese"), ("nl", "Dutch", "Olandese"),
    ("ru", "Russian", "Russo"), ("zh", "Chinese", "Cinese"), ("ja", "Japanese", "Giapponese"),
    ("ko", "Korean", "Coreano"), ("ar", "Arabic", "Arabo"), ("hi", "Hindi", "Hindi"),
    ("tr", "Turkish", "Turco"), ("pl", "Polish", "Polacco"), ("sv", "Swedish", "Svedese"),
    ("da", "Danish", "Danese"), ("no", "Norwegian", "Norvegese"), ("fi", "Finnish", "Finlandese"),
    ("el", "Greek", "Greco"), ("he", "Hebrew", "Ebraico"), ("id", "Indonesian", "Indonesiano"),
    ("vi", "Vietnamese", "Vietnamita"), ("th", "Thai", "Thailandese"), ("uk", "Ukrainian", "Ucraino"),
    ("cs", "Czech", "Ceco"), ("ro", "Romanian", "Rumeno"), ("hu", "Hungarian", "Ungherese"),
]

# initial_prompt is a *style example* Whisper mimics (tone, vocabulary,
# spelling), not a command it obeys. Rather than making the user hand-write
# that example text, this GUI offers named styles that resolve to a fitting
# example prompt behind the scenes -- plus a "Custom" style for anyone who
# wants to write the raw example text themselves.
STYLE_PRESETS = [
    dict(key="none", label_en="None (no style bias)", label_it="Nessuno (nessuna influenza di stile)",
         prompt=""),
    dict(key="formal", label_en="Formal / lecture", label_it="Formale / lezione",
         prompt="This is a formal lecture with correct punctuation and full sentences."),
    dict(key="casual", label_en="Casual conversation", label_it="Conversazione informale",
         prompt="This is a casual, informal conversation between friends."),
    dict(key="technical", label_en="Technical / engineering", label_it="Tecnico / ingegneria",
         prompt="This is a technical presentation with software and engineering terminology."),
    dict(key="podcast", label_en="Podcast (multiple speakers)", label_it="Podcast (più relatori)",
         prompt="This is a podcast episode with multiple speakers discussing a topic."),
    dict(key="business", label_en="Business meeting", label_it="Riunione di lavoro",
         prompt="This is a business meeting recording with technical jargon."),
    dict(key="medical", label_en="Medical / scientific", label_it="Medico / scientifico",
         prompt="This is a medical or scientific discussion with accurate technical terms."),
    dict(key="custom", label_en="Custom (write your own example)", label_it="Personalizzato (scrivi il tuo esempio)",
         prompt=None),
]

TR = {
    "en": {
        "window_title": "Faster-Whisper GUI",
        "ui_lang_label": "Interface language:",
        "app_header": "Faster-Whisper GUI",
        "group_hardware": "Detected Hardware",
        "gpu_detected": "GPU: {name} (~{vram} GB VRAM)",
        "gpu_none": "GPU: none detected (CUDA unavailable)",
        "system_ram": "System RAM: ~{ram} GB",
        "suggest_btn": "Auto-Suggest Settings",
        "group_model": "Model (\u201cAbility\u201d) & Precision",
        "device_label": "Inference device:",
        "device_gpu": "GPU (CUDA)",
        "device_cpu": "CPU",
        "cpu_threads_label": "CPU threads:",
        "ability_title": "Ability (model size \u2014 bigger = more accurate, slower, more memory)",
        "precision_title": "Precision (compute type \u2014 higher = better quality, more memory)",
        "estimate_prefix": "Estimated {unit} required",
        "estimate_available": "available",
        "estimate_warn": "  \u26a0 May exceed available memory \u2014 consider lowering ability/precision.",
        "group_source": "Source",
        "radio_file": "Local file (audio or video)",
        "radio_url": "YouTube / direct link (via yt-dlp)",
        "file_placeholder": "Path to audio/video file...",
        "url_placeholder": "https://www.youtube.com/watch?v=... or direct media URL",
        "browse": "Browse...",
        "output_folder_label": "Output folder:",
        "group_options": "Transcription Options",
        "language_label": "Audio language:",
        "steering_title": "Text steering (three independent controls)",
        "initial_prompt_label": "Style — biases the initial prompt",
        "initial_prompt_hint": (
            "Picks a fitting example text and sends it as the initial prompt, biasing "
            "tone, vocabulary and spelling (up to ~224 tokens). It does NOT obey "
            "instructions -- Whisper mimics the example's style, it doesn't follow "
            "commands like \u201cadd punctuation\u201d."
        ),
        "initial_prompt_custom_label": "Custom style example text:",
        "prefix_label": "Prefix — forces exact wording at the start",
        "prefix_hint": (
            "Unlike the initial prompt, this text is NOT just a style hint: it is "
            "inserted verbatim as the beginning of the first transcribed segment. "
            "Use it when you already know exactly how the audio opens."
        ),
        "hotwords_label": "Hotwords — boost specific words",
        "hotwords_hint": (
            "Space- or comma-separated words/phrases whose probability is boosted "
            "during decoding (e.g. product names, acronyms). Doesn't use any of "
            "the prompt token budget."
        ),
        "vad_checkbox": "Use VAD filter",
        "vad_tooltip": (
            "Voice Activity Detection (VAD) filter\n\n"
            "Before transcribing, a separate model (Silero VAD) scans the audio and "
            "detects which parts actually contain speech.\n\n"
            "Silent or noise-only segments are skipped instead of being sent to "
            "Whisper. Benefits:\n"
            "\u2022 Faster processing (less audio to transcribe)\n"
            "\u2022 Fewer hallucinated phrases during silence/background noise\n\n"
            "Trade-off: if set too aggressively it can occasionally clip very quiet "
            "speech. The defaults used here are conservative."
        ),
        "beam_label": "Beam size:",
        "formats_label": "Output formats:",
        "group_run": "Run",
        "start_btn": "Start Transcription",
        "stop_btn": "Stop",
        "stopping": "Stopping...",
        "cancelled_log": "Cancelled by user.",
        "no_gpu_title": "No GPU detected",
        "no_gpu_msg": "CUDA GPU was not detected on this machine. Falling back to CPU.",
        "missing_output_title": "Missing output folder",
        "missing_output_msg": "Please choose an output folder.",
        "no_ffmpeg_title": "ffmpeg not found",
        "no_ffmpeg_msg": (
            "ffmpeg was not found on PATH, and the automatic static-ffmpeg "
            "fallback couldn't fetch it either (likely no network access). "
            "Downloading/decoding may fail. Please install ffmpeg manually "
            "and ensure it's on your PATH."
        ),
        "missing_url_title": "Missing URL",
        "missing_url_msg": "Please enter a URL.",
        "missing_file_title": "Missing file",
        "missing_file_msg": "Please select a valid file.",
        "done_title": "Done",
        "done_msg": "Transcription complete.\n\nFiles written:\n",
        "error_title": "Error",
        "suggest_applied": "Auto-suggested settings applied based on detected hardware.",
        "starting_download": "Starting download: {url}",
        "audio_ready": "Audio ready: {path}",
        "model_device_line": "Model: {model}  |  Device: {device}  |  Compute: {compute}",
        "select_file_dialog": "Select audio or video file",
        "select_out_dialog": "Select output folder",
        "done_files_log": "Done. Files:",
    },
    "it": {
        "window_title": "Faster-Whisper GUI",
        "ui_lang_label": "Lingua interfaccia:",
        "app_header": "Faster-Whisper GUI",
        "group_hardware": "Hardware rilevato",
        "gpu_detected": "GPU: {name} (~{vram} GB VRAM)",
        "gpu_none": "GPU: nessuna rilevata (CUDA non disponibile)",
        "system_ram": "RAM di sistema: ~{ram} GB",
        "suggest_btn": "Suggerisci automaticamente",
        "group_model": "Modello (\u201cAbilità\u201d) e precisione",
        "device_label": "Dispositivo di inferenza:",
        "device_gpu": "GPU (CUDA)",
        "device_cpu": "CPU",
        "cpu_threads_label": "Thread CPU:",
        "ability_title": "Abilità (dimensione modello \u2014 più alta = più precisa, più lenta, più memoria)",
        "precision_title": "Precisione (compute type \u2014 più alta = qualità migliore, più memoria)",
        "estimate_prefix": "{unit} stimata necessaria",
        "estimate_available": "disponibile",
        "estimate_warn": "  \u26a0 Potrebbe superare la memoria disponibile \u2014 valuta di ridurre abilità/precisione.",
        "group_source": "Sorgente",
        "radio_file": "File locale (audio o video)",
        "radio_url": "YouTube / link diretto (tramite yt-dlp)",
        "file_placeholder": "Percorso del file audio/video...",
        "url_placeholder": "https://www.youtube.com/watch?v=... oppure link diretto al media",
        "browse": "Sfoglia...",
        "output_folder_label": "Cartella di output:",
        "group_options": "Opzioni di trascrizione",
        "language_label": "Lingua dell'audio:",
        "steering_title": "Guida al testo (tre controlli indipendenti)",
        "initial_prompt_label": "Stile — orienta il prompt iniziale",
        "initial_prompt_hint": (
            "Sceglie un testo di esempio adatto e lo invia come prompt iniziale, "
            "orientando tono, vocabolario e ortografia (fino a circa 224 token). "
            "NON esegue istruzioni: Whisper imita lo stile dell'esempio, non segue "
            "comandi come \u201caggiungi la punteggiatura\u201d."
        ),
        "initial_prompt_custom_label": "Testo di esempio personalizzato:",
        "prefix_label": "Prefisso — impone il testo esatto all'inizio",
        "prefix_hint": (
            "A differenza del prompt iniziale, questo testo non è solo un suggerimento "
            "di stile: viene inserito parola per parola come inizio del primo segmento "
            "trascritto. Usalo quando sai già esattamente come inizia l'audio."
        ),
        "hotwords_label": "Hotwords — dai più peso a parole specifiche",
        "hotwords_hint": (
            "Parole/frasi separate da spazi o virgole la cui probabilità viene "
            "aumentata durante la decodifica (es. nomi di prodotto, acronimi). "
            "Non consuma il budget di token del prompt."
        ),
        "vad_checkbox": "Usa filtro VAD",
        "vad_tooltip": (
            "Filtro Voice Activity Detection (VAD)\n\n"
            "Prima della trascrizione, un modello separato (Silero VAD) analizza "
            "l'audio e individua le parti che contengono effettivamente voce.\n\n"
            "I segmenti silenziosi o di solo rumore vengono saltati invece di essere "
            "inviati a Whisper. Vantaggi:\n"
            "\u2022 Elaborazione più veloce (meno audio da trascrivere)\n"
            "\u2022 Meno frasi \u201callucinate\u201d durante silenzi o rumore di fondo\n\n"
            "Nota: se impostato in modo troppo aggressivo può occasionalmente tagliare "
            "parlato molto sommesso. I valori predefiniti qui usati sono prudenti."
        ),
        "beam_label": "Beam size:",
        "formats_label": "Formati di output:",
        "group_run": "Esecuzione",
        "start_btn": "Avvia trascrizione",
        "stop_btn": "Interrompi",
        "stopping": "Interruzione in corso...",
        "cancelled_log": "Interrotto dall'utente.",
        "no_gpu_title": "Nessuna GPU rilevata",
        "no_gpu_msg": "Nessuna GPU CUDA rilevata su questa macchina. Verrà usata la CPU.",
        "missing_output_title": "Cartella di output mancante",
        "missing_output_msg": "Seleziona una cartella di output.",
        "no_ffmpeg_title": "ffmpeg non trovato",
        "no_ffmpeg_msg": (
            "ffmpeg non è stato trovato nel PATH e il tentativo automatico "
            "tramite static-ffmpeg non è riuscito a scaricarlo (probabilmente "
            "manca l'accesso alla rete). Il download/la decodifica potrebbero "
            "fallire. Installa ffmpeg manualmente e assicurati che sia nel PATH."
        ),
        "missing_url_title": "URL mancante",
        "missing_url_msg": "Inserisci un URL.",
        "missing_file_title": "File mancante",
        "missing_file_msg": "Seleziona un file valido.",
        "done_title": "Completato",
        "done_msg": "Trascrizione completata.\n\nFile scritti:\n",
        "error_title": "Errore",
        "suggest_applied": "Impostazioni suggerite applicate in base all'hardware rilevato.",
        "starting_download": "Avvio download: {url}",
        "audio_ready": "Audio pronto: {path}",
        "model_device_line": "Modello: {model}  |  Dispositivo: {device}  |  Compute: {compute}",
        "select_file_dialog": "Seleziona file audio o video",
        "select_out_dialog": "Seleziona cartella di output",
        "done_files_log": "Completato. File:",
    },
}


def format_timestamp_srt(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    return format_timestamp_srt(seconds).replace(",", ".")


# ------------------------------------------------------------------------
# Hardware detection helpers
# ------------------------------------------------------------------------

def get_gpu_info():
    """Return (has_cuda, gpu_name, vram_gb)."""
    if torch is not None:
        try:
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                return True, props.name, round(props.total_memory / (1024 ** 3), 1)
        except Exception:
            pass
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=3,
        ).decode().strip().splitlines()
        if out:
            name, mem = out[0].split(",")
            return True, name.strip(), round(float(mem.strip()) / 1024, 1)
    except Exception:
        pass
    return False, None, 0.0


def get_ram_gb():
    if psutil is not None:
        return round(psutil.virtual_memory().total / (1024 ** 3), 1)
    return 8.0


def estimate_memory_gb(model_info, compute_info):
    return round(model_info["base_gb"] * compute_info["mult"], 2)


# ------------------------------------------------------------------------
# Self-contained dark palette + ttk theme
# (Deliberately fixed, not derived from the OS theme, so the app looks and
#  behaves the same on every machine and desktop environment.)
# ------------------------------------------------------------------------

PALETTE = {
    "bg":            "#1b1d24",
    "bg_header":     "#15161c",
    "panel":         "#232631",
    "panel_alt":     "#262a37",
    "panel_border":  "#343849",
    "accent":        "#5b8cff",
    "accent_hover":  "#7aa2ff",
    "accent_active": "#3f6fe0",
    "text":          "#e8eaf0",
    "text_dim":      "#a4aab8",
    "text_faint":    "#767c8c",
    "entry_bg":      "#2a2e3b",
    "entry_fg":      "#e8eaf0",
    "success":       "#4fbf85",
    "warn":          "#e0a13c",
    "danger":        "#e2635f",
    "tooltip_bg":    "#2f3342",
    "tooltip_fg":    "#e8eaf0",
    "log_bg":        "#14151b",
    "log_fg":        "#c9cedb",
}

FONT_BASE = ("Segoe UI", 10)
FONT_BOLD = ("Segoe UI", 10, "bold")
FONT_HEADER = ("Segoe UI", 16, "bold")
FONT_SMALL = ("Segoe UI", 9)
FONT_MONO = ("Consolas", 9)


def apply_theme(root: tk.Tk, palette: dict) -> ttk.Style:
    root.configure(bg=palette["bg"])
    style = ttk.Style(root)
    # "clam" is the only bundled ttk theme that fully honors color overrides
    # on every platform, which is exactly what we need for a fixed look.
    style.theme_use("clam")

    style.configure(".", background=palette["bg"], foreground=palette["text"],
                     fieldbackground=palette["entry_bg"], font=FONT_BASE,
                     bordercolor=palette["panel_border"])

    style.configure("TFrame", background=palette["bg"])
    style.configure("Panel.TFrame", background=palette["panel"])
    style.configure("Header.TFrame", background=palette["bg_header"])

    style.configure("TLabel", background=palette["bg"], foreground=palette["text"])
    style.configure("Panel.TLabel", background=palette["panel"], foreground=palette["text"])
    style.configure("Dim.TLabel", background=palette["panel"], foreground=palette["text_dim"])
    style.configure("Hint.TLabel", background=palette["panel"], foreground=palette["text_faint"], font=FONT_SMALL)
    style.configure("Title.TLabel", background=palette["panel"], foreground=palette["text"], font=FONT_BOLD)
    style.configure("Header.TLabel", background=palette["bg_header"], foreground=palette["accent"], font=FONT_HEADER)
    style.configure("Estimate.TLabel", background=palette["panel"], font=FONT_BOLD)

    style.configure("TLabelframe", background=palette["panel"],
                     bordercolor=palette["panel_border"], darkcolor=palette["panel_border"],
                     lightcolor=palette["panel_border"], relief="solid", borderwidth=1)
    style.configure("TLabelframe.Label", background=palette["panel"],
                     foreground=palette["accent"], font=FONT_BOLD)

    style.configure("TButton", background=palette["panel_alt"], foreground=palette["text"],
                     borderwidth=0, focuscolor=palette["accent"], padding=(10, 6))
    style.map("TButton",
              background=[("active", palette["accent_hover"]), ("pressed", palette["accent_active"])],
              foreground=[("disabled", palette["text_faint"])])

    style.configure("Accent.TButton", background=palette["accent"], foreground="#ffffff",
                     padding=(16, 9), font=FONT_BOLD, borderwidth=0)
    style.map("Accent.TButton",
              background=[("active", palette["accent_hover"]), ("pressed", palette["accent_active"]),
                          ("disabled", palette["panel_border"])],
              foreground=[("disabled", palette["text_faint"])])

    style.configure("Danger.TButton", background=palette["danger"], foreground="#ffffff",
                     padding=(16, 9), font=FONT_BOLD, borderwidth=0)
    style.map("Danger.TButton",
              background=[("active", "#ef8480"), ("pressed", "#c94f4b"),
                          ("disabled", palette["panel_border"])],
              foreground=[("disabled", palette["text_faint"])])

    style.configure("TEntry", fieldbackground=palette["entry_bg"], foreground=palette["entry_fg"],
                     bordercolor=palette["panel_border"], insertcolor=palette["text"],
                     lightcolor=palette["entry_bg"], darkcolor=palette["entry_bg"])
    style.map("TEntry", bordercolor=[("focus", palette["accent"])])

    style.configure("TCombobox", fieldbackground=palette["entry_bg"], background=palette["entry_bg"],
                     foreground=palette["entry_fg"], arrowcolor=palette["text"],
                     bordercolor=palette["panel_border"])
    style.map("TCombobox",
              fieldbackground=[("readonly", palette["entry_bg"]), ("disabled", palette["panel"])],
              foreground=[("disabled", palette["text_faint"])])

    style.configure("TCheckbutton", background=palette["panel"], foreground=palette["text"])
    style.map("TCheckbutton", background=[("active", palette["panel"])],
              indicatorcolor=[("selected", palette["accent"]), ("!selected", palette["entry_bg"])])

    style.configure("TRadiobutton", background=palette["panel"], foreground=palette["text"])
    style.map("TRadiobutton", background=[("active", palette["panel"])],
              indicatorcolor=[("selected", palette["accent"]), ("!selected", palette["entry_bg"])])

    style.configure("TSpinbox", fieldbackground=palette["entry_bg"], foreground=palette["entry_fg"],
                     arrowcolor=palette["text"], bordercolor=palette["panel_border"])

    style.configure("Horizontal.TScale", background=palette["panel"], troughcolor=palette["entry_bg"])

    style.configure("Horizontal.TProgressbar", background=palette["accent"],
                     troughcolor=palette["entry_bg"], bordercolor=palette["panel_border"],
                     lightcolor=palette["accent"], darkcolor=palette["accent"])

    style.configure("TSeparator", background=palette["panel_border"])

    # Combobox popdown listbox colors aren't ttk-styleable directly.
    root.option_add("*TCombobox*Listbox.background", palette["entry_bg"])
    root.option_add("*TCombobox*Listbox.foreground", palette["entry_fg"])
    root.option_add("*TCombobox*Listbox.selectBackground", palette["accent"])
    root.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")
    root.option_add("*TCombobox*Listbox.font", FONT_BASE)

    return style


class ToolTip:
    """Lightweight hover tooltip for any tkinter/ttk widget.

    text_getter is a zero-arg callable so the tooltip content stays correct
    across UI-language switches without needing to be rebound.
    """

    def __init__(self, widget, text_getter, palette, wraplength=380):
        self.widget = widget
        self.text_getter = text_getter
        self.palette = palette
        self.wraplength = wraplength
        self.tipwindow = None
        widget.bind("<Enter>", self.show, add="+")
        widget.bind("<Leave>", self.hide, add="+")
        widget.bind("<ButtonPress>", self.hide, add="+")

    def show(self, _event=None):
        text = self.text_getter()
        if not text or self.tipwindow is not None:
            return
        x = self.widget.winfo_rootx() + 12
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 8
        tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        try:
            tw.wm_attributes("-topmost", True)
        except tk.TclError:
            pass
        tw.wm_geometry(f"+{x}+{y}")
        frame = tk.Frame(tw, background=self.palette["accent"], bd=0)
        frame.pack(padx=0, pady=0)
        inner = tk.Frame(frame, background=self.palette["tooltip_bg"])
        inner.pack(padx=1, pady=1)
        label = tk.Label(
            inner, text=text, justify="left", background=self.palette["tooltip_bg"],
            foreground=self.palette["tooltip_fg"], font=FONT_SMALL,
            wraplength=self.wraplength, padx=10, pady=8,
        )
        label.pack()
        self.tipwindow = tw

    def hide(self, _event=None):
        if self.tipwindow is not None:
            self.tipwindow.destroy()
            self.tipwindow = None


class InfoDot(tk.Canvas):
    """Small circular "i" info button with a hover tooltip, replacing the
    QToolButton bubble from the PyQt version."""

    def __init__(self, parent, text_getter, palette, size=20):
        super().__init__(parent, width=size, height=size, background=palette["panel"],
                          highlightthickness=0, bd=0, cursor="question_arrow")
        self.palette = palette
        self.size = size
        self._draw()
        self.tooltip = ToolTip(self, text_getter, palette)

    def _draw(self):
        p = self.palette
        s = self.size
        self.create_oval(1, 1, s - 1, s - 1, outline=p["accent"], width=1.4, fill=p["panel_alt"])
        self.create_text(s / 2, s / 2, text="i", fill=p["accent"], font=("Segoe UI", 9, "bold"))


# ------------------------------------------------------------------------
# Background worker threads (communicate back to the Tk main loop via a
# thread-safe queue, polled with root.after -- Tk widgets must only be
# touched from the main thread).
#
# Both workers accept a threading.Event `cancel_event`. They check it at
# every safe checkpoint (yt-dlp's progress hook; each transcribed segment)
# and unwind cooperatively -- there is no way to hard-kill a Python thread,
# so cancellation is always "please stop soon", not "stop now".
# ------------------------------------------------------------------------

class CancelledError(Exception):
    """Raised internally to unwind a worker once cancellation is requested."""


class DownloadWorker(threading.Thread):
    def __init__(self, url: str, out_dir: str, result_queue: "queue.Queue", cancel_event: "threading.Event"):
        super().__init__(daemon=True)
        self.url = url
        self.out_dir = out_dir
        self.result_queue = result_queue
        self.cancel_event = cancel_event
        self._final_path = None

    def _hook(self, d):
        # yt-dlp calls this hook frequently during download; it's our only
        # cooperative checkpoint, so raise to unwind ydl.download() cleanly.
        if self.cancel_event.is_set():
            raise CancelledError("cancelled during download")
        if d.get("status") == "downloading":
            pct = d.get("_percent_str", "").strip()
            speed = d.get("_speed_str", "").strip()
            self.result_queue.put(("log", f"Downloading... {pct} ({speed})"))
        elif d.get("status") == "finished":
            self._final_path = d.get("filename")
            self.result_queue.put(("log", "Download finished, extracting audio..."))

    def run(self):
        try:
            import yt_dlp
        except ImportError:
            self.result_queue.put(("failed", "yt-dlp is not installed. Run: pip install yt-dlp"))
            return

        os.makedirs(self.out_dir, exist_ok=True)
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": os.path.join(self.out_dir, "%(title).150s.%(ext)s"),
            "restrictfilenames": True,
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }],
            "progress_hooks": [self._hook],
        }
        try:
            if self.cancel_event.is_set():
                raise CancelledError("cancelled before download started")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                self.result_queue.put(("log", f"Fetching info for: {self.url}"))
                ydl.download([self.url])

            if self.cancel_event.is_set():
                raise CancelledError("cancelled after download")

            if not self._final_path:
                self.result_queue.put(("failed", "Download completed but no output file was captured."))
                return

            wav_path = os.path.splitext(self._final_path)[0] + ".wav"
            if os.path.exists(wav_path):
                self.result_queue.put(("download_done", wav_path))
            elif os.path.exists(self._final_path):
                self.result_queue.put(("download_done", self._final_path))
            else:
                self.result_queue.put(("failed", "Could not locate the downloaded/extracted audio file."))
        except CancelledError:
            self.result_queue.put(("cancelled", None))
        except Exception as e:
            # yt-dlp wraps hook exceptions in its own DownloadError; if our
            # CancelledError is the root cause, still report it as a clean
            # cancellation rather than a failure.
            if self.cancel_event.is_set():
                self.result_queue.put(("cancelled", None))
            else:
                self.result_queue.put(("failed", f"Download failed: {e}\n{traceback.format_exc()}"))


class TranscribeWorker(threading.Thread):
    def __init__(self, audio_path, model_name, device, compute_type,
                 language, initial_prompt, prefix, hotwords,
                 vad_filter, beam_size, cpu_threads, result_queue: "queue.Queue",
                 cancel_event: "threading.Event"):
        super().__init__(daemon=True)
        self.audio_path = audio_path
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.initial_prompt = initial_prompt
        self.prefix = prefix
        self.hotwords = hotwords
        self.vad_filter = vad_filter
        self.beam_size = beam_size
        self.cpu_threads = cpu_threads
        self.result_queue = result_queue
        self.cancel_event = cancel_event

    def run(self):
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            self.result_queue.put(("failed", "faster-whisper is not installed. Run: pip install faster-whisper"))
            return

        try:
            if self.cancel_event.is_set():
                self.result_queue.put(("cancelled", None))
                return

            self.result_queue.put((
                "log",
                f"Loading model '{self.model_name}' on {self.device} ({self.compute_type})..."
            ))
            kwargs = dict(device=self.device, compute_type=self.compute_type)
            if self.device == "cpu":
                kwargs["cpu_threads"] = self.cpu_threads

            model = WhisperModel(self.model_name, **kwargs)

            if self.cancel_event.is_set():
                self.result_queue.put(("cancelled", None))
                return

            lang = None if self.language == "auto" else self.language
            initial_prompt = self.initial_prompt.strip() or None
            prefix = self.prefix.strip() or None
            hotwords = self.hotwords.strip() or None

            self.result_queue.put(("log", "Starting transcription..."))
            segments_gen, info = model.transcribe(
                self.audio_path,
                language=lang,
                initial_prompt=initial_prompt,
                prefix=prefix,
                hotwords=hotwords,
                vad_filter=self.vad_filter,
                beam_size=self.beam_size,
            )

            self.result_queue.put((
                "log",
                f"Detected language: {info.language} (probability {info.language_probability:.2f})"
            ))

            segments = []
            for seg in segments_gen:
                # segments_gen is lazy: faster-whisper decodes one segment
                # per iteration, so checking here between iterations is a
                # genuine, timely cancellation point (not just at the end).
                if self.cancel_event.is_set():
                    self.result_queue.put(("cancelled", None))
                    return
                segments.append(seg)
                self.result_queue.put((
                    "log",
                    f"[{format_timestamp_vtt(seg.start)} -> {format_timestamp_vtt(seg.end)}] {seg.text.strip()}"
                ))

            if self.cancel_event.is_set():
                self.result_queue.put(("cancelled", None))
                return

            meta = dict(language=info.language, duration=getattr(info, "duration", None))
            self.result_queue.put(("transcribe_done", (segments, meta)))
        except Exception as e:
            if self.cancel_event.is_set():
                self.result_queue.put(("cancelled", None))
            else:
                self.result_queue.put(("failed", f"Transcription failed: {e}\n{traceback.format_exc()}"))


# ------------------------------------------------------------------------
# Main application
# ------------------------------------------------------------------------

class WhisperGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.palette = PALETTE
        self.lang = "it"
        self.title("Faster-Whisper GUI")
        self.geometry("1180x820")
        self.minsize(760, 480)

        self.style = apply_theme(self, self.palette)

        self.has_cuda, self.gpu_name, self.vram_gb = get_gpu_info()
        self.ram_gb = get_ram_gb()

        self.result_queue: "queue.Queue" = queue.Queue()
        self.active_worker = None      # "download" | "transcribe" | None
        self.current_worker_thread = None
        self.cancel_event = threading.Event()
        self.current_audio_path = None
        self.last_segments = None
        self.last_meta = None

        # tk variables
        self.ui_lang_var = tk.StringVar(value="Italiano")
        self.device_var = tk.StringVar(value="cuda" if self.has_cuda else "cpu")
        self.cpu_threads_var = tk.IntVar(value=4)
        self.ability_var = tk.IntVar(value=4)
        self.precision_var = tk.IntVar(value=2)
        self.source_var = tk.StringVar(value="file")
        self.file_path_var = tk.StringVar()
        self.url_var = tk.StringVar()
        self.out_dir_var = tk.StringVar(value=os.path.join(os.path.expanduser("~"), "whisper_output"))
        self.lang_display_var = tk.StringVar()
        self.style_display_var = tk.StringVar()
        self.initial_prompt_custom_var = tk.StringVar()
        self.prefix_var = tk.StringVar()
        self.hotwords_var = tk.StringVar()
        self.vad_var = tk.BooleanVar(value=True)
        self.beam_var = tk.IntVar(value=5)
        self.fmt_txt_var = tk.BooleanVar(value=True)
        self.fmt_srt_var = tk.BooleanVar(value=True)
        self.fmt_vtt_var = tk.BooleanVar(value=False)

        self._lang_code_by_display = {}
        self._display_by_lang_code = {}
        self._style_key_by_display = {}
        self._display_by_style_key = {}

        self._build_ui()
        self.retranslate_ui()
        self._on_device_changed()
        self._refresh_model_label()
        self._refresh_precision_combo_targets()
        self._update_estimate()

        self.after(100, self._poll_queue)

    def t(self, key, **fmt):
        text = TR[self.lang].get(key, key)
        return text.format(**fmt) if fmt else text

    # ---------------------------- UI BUILD ----------------------------

    def _build_ui(self):
        p = self.palette

        # Header bar
        header = ttk.Frame(self, style="Header.TFrame", padding=(18, 14))
        header.pack(fill="x", side="top")
        self.header_label = ttk.Label(header, style="Header.TLabel")
        self.header_label.pack(side="left")

        lang_box = ttk.Frame(header, style="Header.TFrame")
        lang_box.pack(side="right")
        self.ui_lang_label = tk.Label(lang_box, background=p["bg_header"], foreground=p["text_dim"], font=FONT_BASE)
        self.ui_lang_label.pack(side="left", padx=(0, 8))
        self.ui_lang_combo = ttk.Combobox(lang_box, textvariable=self.ui_lang_var, state="readonly",
                                           width=12, values=["English", "Italiano"])
        self.ui_lang_combo.pack(side="left")
        self.ui_lang_combo.bind("<<ComboboxSelected>>", self._on_ui_lang_changed)

        # Scrollable body -- a Canvas + Scrollbar wrapping the real content
        # frame. This is what keeps every panel reachable in a normal,
        # un-maximized window on a 1080p (or smaller) screen: instead of the
        # window needing to be tall enough to fit everything at once, the
        # content simply scrolls.
        scroll_container = ttk.Frame(self)
        scroll_container.pack(fill="both", expand=True)

        canvas = tk.Canvas(scroll_container, background=p["bg"], highlightthickness=0, bd=0)
        vscroll = ttk.Scrollbar(scroll_container, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vscroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        vscroll.pack(side="right", fill="y")

        body = ttk.Frame(canvas, padding=(14, 10, 14, 14))
        body_window = canvas.create_window((0, 0), window=body, anchor="nw")

        def _on_body_configure(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(event):
            canvas.itemconfig(body_window, width=event.width)

        body.bind("<Configure>", _on_body_configure)
        canvas.bind("<Configure>", _on_canvas_configure)

        def _on_mousewheel(event):
            if event.num == 4:
                canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                canvas.yview_scroll(1, "units")
            else:
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)   # Windows / macOS
        canvas.bind_all("<Button-4>", _on_mousewheel)      # Linux scroll up
        canvas.bind_all("<Button-5>", _on_mousewheel)      # Linux scroll down

        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)

        left_col = ttk.Frame(body)
        left_col.grid(row=0, column=0, sticky="nsew", padx=(0, 7))
        right_col = ttk.Frame(body)
        right_col.grid(row=0, column=1, sticky="nsew", padx=(7, 0))

        # Left column: hardware -> model selection -> run panel.
        self.hardware_box = self._build_hardware_box(left_col)
        self.hardware_box.pack(fill="x", pady=(0, 10))
        self.model_box = self._build_model_box(left_col)
        self.model_box.pack(fill="x", pady=(0, 10))
        self.run_box = self._build_run_box(left_col)
        self.run_box.pack(fill="both", expand=True)

        # Right column: source -> options, sitting alongside the run panel.
        self.source_box = self._build_source_box(right_col)
        self.source_box.pack(fill="x", pady=(0, 10))
        self.options_box = self._build_options_box(right_col)
        self.options_box.pack(fill="both", expand=True)

    def _labelframe(self, parent, **kwargs):
        return ttk.LabelFrame(parent, padding=(14, 10), **kwargs)

    def _separator(self, parent):
        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=8)

    def _build_hardware_box(self, parent):
        box = self._labelframe(parent)
        row = ttk.Frame(box, style="Panel.TFrame")
        row.pack(fill="x")
        self.sys_label = ttk.Label(row, style="Panel.TLabel", wraplength=420, justify="left")
        self.sys_label.pack(side="left", fill="x", expand=True)
        self.suggest_btn = ttk.Button(row, command=self._auto_suggest)
        self.suggest_btn.pack(side="right")
        return box

    def _build_model_box(self, parent):
        box = self._labelframe(parent)

        dev_row = ttk.Frame(box, style="Panel.TFrame")
        dev_row.pack(fill="x")
        self.device_label = ttk.Label(dev_row, style="Panel.TLabel")
        self.device_label.pack(side="left")
        self.device_combo = ttk.Combobox(dev_row, state="readonly", width=14)
        self.device_combo.pack(side="left", padx=(8, 20))
        self.device_combo.bind("<<ComboboxSelected>>", self._on_device_combo_selected)

        self.cpu_threads_label = ttk.Label(dev_row, style="Panel.TLabel")
        self.cpu_threads_label.pack(side="left")
        self.cpu_threads_spin = ttk.Spinbox(dev_row, from_=1, to=64, width=5,
                                             textvariable=self.cpu_threads_var)
        self.cpu_threads_spin.pack(side="left", padx=(8, 0))

        self._separator(box)

        self.ability_title_label = ttk.Label(box, style="Title.TLabel")
        self.ability_title_label.pack(anchor="w")
        self.ability_scale = ttk.Scale(box, from_=0, to=len(MODEL_TABLE) - 1, orient="horizontal",
                                        variable=self.ability_var, command=self._on_ability_scaled)
        self.ability_scale.pack(fill="x", pady=(6, 4))
        self.ability_label = ttk.Label(box, style="Panel.TLabel", wraplength=440, justify="left")
        self.ability_label.pack(anchor="w", fill="x")

        self._separator(box)

        self.precision_title_label = ttk.Label(box, style="Title.TLabel")
        self.precision_title_label.pack(anchor="w")
        self.precision_scale = ttk.Scale(box, from_=0, to=len(COMPUTE_TYPES_GPU) - 1, orient="horizontal",
                                          variable=self.precision_var, command=self._on_precision_scaled)
        self.precision_scale.pack(fill="x", pady=(6, 4))
        self.precision_label = ttk.Label(box, style="Panel.TLabel", wraplength=440, justify="left")
        self.precision_label.pack(anchor="w", fill="x")

        self.estimate_label = ttk.Label(box, style="Estimate.TLabel", wraplength=440, justify="left")
        self.estimate_label.pack(anchor="w", fill="x", pady=(10, 0))

        return box

    def _build_source_box(self, parent):
        box = self._labelframe(parent)

        self.radio_file = ttk.Radiobutton(box, variable=self.source_var, value="file",
                                           command=self._on_source_toggled)
        self.radio_file.pack(anchor="w")

        file_row = ttk.Frame(box, style="Panel.TFrame")
        file_row.pack(fill="x", pady=(4, 0))
        self.file_path_entry = ttk.Entry(file_row, textvariable=self.file_path_var)
        self.file_path_entry.pack(side="left", fill="x", expand=True)
        self.file_browse_btn = ttk.Button(file_row, command=self._browse_file)
        self.file_browse_btn.pack(side="left", padx=(8, 0))

        self._separator(box)

        self.radio_url = ttk.Radiobutton(box, variable=self.source_var, value="url",
                                          command=self._on_source_toggled)
        self.radio_url.pack(anchor="w")
        self.url_entry = ttk.Entry(box, textvariable=self.url_var, state="disabled")
        self.url_entry.pack(fill="x", pady=(4, 0))

        self._separator(box)

        out_row = ttk.Frame(box, style="Panel.TFrame")
        out_row.pack(fill="x")
        self.output_folder_label = ttk.Label(out_row, style="Panel.TLabel")
        self.output_folder_label.pack(side="left")
        self.out_dir_entry = ttk.Entry(box, textvariable=self.out_dir_var)
        self.out_dir_entry.pack(fill="x", pady=(4, 0), side="bottom")
        self.out_browse_btn = ttk.Button(out_row, command=self._browse_out_dir)
        self.out_browse_btn.pack(side="right")

        return box

    def _build_options_box(self, parent):
        box = self._labelframe(parent)

        lang_row = ttk.Frame(box, style="Panel.TFrame")
        lang_row.pack(fill="x")
        self.language_label = ttk.Label(lang_row, style="Panel.TLabel")
        self.language_label.pack(side="left")
        self.lang_combo = ttk.Combobox(lang_row, state="readonly", width=26,
                                        textvariable=self.lang_display_var)
        self.lang_combo.pack(side="left", padx=(8, 0))
        self.lang_combo.bind("<<ComboboxSelected>>", self._on_lang_combo_selected)

        self._separator(box)

        self.steering_title_label = ttk.Label(box, style="Title.TLabel")
        self.steering_title_label.pack(anchor="w")

        # initial_prompt, driven by a named "style" instead of raw example
        # text -- selecting a style resolves to a fitting example prompt
        # behind the scenes. A "Custom" style keeps the raw-text route open.
        self.initial_prompt_label = ttk.Label(box, style="Panel.TLabel")
        self.initial_prompt_label.pack(anchor="w", pady=(8, 0))
        self.style_combo = ttk.Combobox(box, state="readonly", textvariable=self.style_display_var)
        self.style_combo.pack(fill="x", pady=(2, 0))
        self.style_combo.bind("<<ComboboxSelected>>", self._on_style_combo_selected)
        self.initial_prompt_hint_label = ttk.Label(box, style="Hint.TLabel", wraplength=440, justify="left")
        self.initial_prompt_hint_label.pack(anchor="w", fill="x", pady=(2, 0))

        self.initial_prompt_custom_label = ttk.Label(box, style="Panel.TLabel")
        self.initial_prompt_custom_entry = ttk.Entry(box, textvariable=self.initial_prompt_custom_var)
        # custom label/entry are packed on demand by _sync_custom_style_visibility()

        # prefix -- forced verbatim start
        self.prefix_label = ttk.Label(box, style="Panel.TLabel")
        self.prefix_label.pack(anchor="w", pady=(10, 0))
        self.prefix_entry = ttk.Entry(box, textvariable=self.prefix_var)
        self.prefix_entry.pack(fill="x", pady=(2, 0))
        self.prefix_hint_label = ttk.Label(box, style="Hint.TLabel", wraplength=440, justify="left")
        self.prefix_hint_label.pack(anchor="w", fill="x", pady=(2, 0))

        # hotwords -- decoding boost, no context cost
        self.hotwords_label = ttk.Label(box, style="Panel.TLabel")
        self.hotwords_label.pack(anchor="w", pady=(10, 0))
        self.hotwords_entry = ttk.Entry(box, textvariable=self.hotwords_var)
        self.hotwords_entry.pack(fill="x", pady=(2, 0))
        self.hotwords_hint_label = ttk.Label(box, style="Hint.TLabel", wraplength=440, justify="left")
        self.hotwords_hint_label.pack(anchor="w", fill="x", pady=(2, 0))

        self._separator(box)

        opts_row = ttk.Frame(box, style="Panel.TFrame")
        opts_row.pack(fill="x")
        self.vad_checkbox = ttk.Checkbutton(opts_row, variable=self.vad_var)
        self.vad_checkbox.pack(side="left")
        self.vad_info_btn = InfoDot(opts_row, lambda: self.t("vad_tooltip"), self.palette)
        self.vad_info_btn.pack(side="left", padx=(6, 24))
        self.beam_label = ttk.Label(opts_row, style="Panel.TLabel")
        self.beam_label.pack(side="left")
        self.beam_spin = ttk.Spinbox(opts_row, from_=1, to=10, width=5, textvariable=self.beam_var)
        self.beam_spin.pack(side="left", padx=(8, 0))

        self._separator(box)

        fmt_row = ttk.Frame(box, style="Panel.TFrame")
        fmt_row.pack(fill="x")
        self.formats_label = ttk.Label(fmt_row, style="Panel.TLabel")
        self.formats_label.pack(side="left")
        ttk.Checkbutton(fmt_row, text=".txt", variable=self.fmt_txt_var).pack(side="left", padx=(10, 0))
        ttk.Checkbutton(fmt_row, text=".srt", variable=self.fmt_srt_var).pack(side="left", padx=(10, 0))
        ttk.Checkbutton(fmt_row, text=".vtt", variable=self.fmt_vtt_var).pack(side="left", padx=(10, 0))

        return box

    def _build_run_box(self, parent):
        box = self._labelframe(parent)

        btn_row = ttk.Frame(box, style="Panel.TFrame")
        btn_row.pack(fill="x")
        self.start_btn = ttk.Button(btn_row, style="Accent.TButton", command=self._start)
        self.start_btn.pack(side="left", fill="x", expand=True)
        self.stop_btn = ttk.Button(btn_row, style="Danger.TButton", command=self._stop, state="disabled")
        self.stop_btn.pack(side="left", fill="x", expand=True, padx=(8, 0))

        self.progress = ttk.Progressbar(box, mode="indeterminate")
        # kept hidden until a job starts; packed on demand

        self.log_box = scrolledtext.ScrolledText(
            box, height=12, wrap="word", background=self.palette["log_bg"],
            foreground=self.palette["log_fg"], insertbackground=self.palette["text"],
            font=FONT_MONO, relief="flat", borderwidth=0,
        )
        self.log_box.configure(state="disabled")
        self.log_box.pack(fill="both", expand=True, pady=(10, 0))

        return box

    # ---------------------------- TRANSLATION ----------------------------

    def retranslate_ui(self):
        self.title(self.t("window_title"))
        self.header_label.configure(text=self.t("app_header"))
        self.ui_lang_label.configure(text=self.t("ui_lang_label"))

        self.hardware_box.configure(text=self.t("group_hardware"))
        gpu_txt = (
            self.t("gpu_detected", name=self.gpu_name, vram=self.vram_gb)
            if self.has_cuda else self.t("gpu_none")
        )
        self.sys_label.configure(text=f"{gpu_txt}    |    {self.t('system_ram', ram=self.ram_gb)}")
        self.suggest_btn.configure(text=self.t("suggest_btn"))

        self.model_box.configure(text=self.t("group_model"))
        self.device_label.configure(text=self.t("device_label"))
        self.device_combo.configure(values=[self.t("device_gpu"), self.t("device_cpu")])
        self._sync_device_combo_display()
        self.cpu_threads_label.configure(text=self.t("cpu_threads_label"))
        self.ability_title_label.configure(text=self.t("ability_title"))
        self.precision_title_label.configure(text=self.t("precision_title"))

        self.source_box.configure(text=self.t("group_source"))
        self.radio_file.configure(text=self.t("radio_file"))
        self.radio_url.configure(text=self.t("radio_url"))
        self.file_browse_btn.configure(text=self.t("browse"))
        self.out_browse_btn.configure(text=self.t("browse"))
        self.output_folder_label.configure(text=self.t("output_folder_label"))
        self._set_placeholder(self.file_path_entry, self.file_path_var, self.t("file_placeholder"))
        self._set_placeholder(self.url_entry, self.url_var, self.t("url_placeholder"))

        self.options_box.configure(text=self.t("group_options"))
        self.language_label.configure(text=self.t("language_label"))
        self._rebuild_language_combo()

        self.steering_title_label.configure(text=self.t("steering_title"))
        self.initial_prompt_label.configure(text=self.t("initial_prompt_label"))
        self.initial_prompt_hint_label.configure(text=self.t("initial_prompt_hint"))
        self.initial_prompt_custom_label.configure(text=self.t("initial_prompt_custom_label"))
        self._rebuild_style_combo()
        self.prefix_label.configure(text=self.t("prefix_label"))
        self.prefix_hint_label.configure(text=self.t("prefix_hint"))
        self.hotwords_label.configure(text=self.t("hotwords_label"))
        self.hotwords_hint_label.configure(text=self.t("hotwords_hint"))

        self.vad_checkbox.configure(text=self.t("vad_checkbox"))
        self.beam_label.configure(text=self.t("beam_label"))
        self.formats_label.configure(text=self.t("formats_label"))

        self.run_box.configure(text=self.t("group_run"))
        self.start_btn.configure(text=self.t("start_btn"))
        self.stop_btn.configure(text=self.t("stop_btn"))

        self._refresh_model_label()
        self._refresh_precision_label()
        self._update_estimate()

    @staticmethod
    def _set_placeholder(entry_widget, var, text):
        # ttk.Entry has no native placeholder; store it as a tooltip-esque
        # hint via a light style instead, keeping actual content untouched.
        pass  # placeholders intentionally omitted to avoid clobbering user input

    def _on_ui_lang_changed(self, _event=None):
        self.lang = "it" if self.ui_lang_combo.get() == "Italiano" else "en"
        self.retranslate_ui()

    # ---------------------------- LANGUAGE COMBO ----------------------------

    def _rebuild_language_combo(self):
        current_code = self._lang_code_by_display.get(self.lang_display_var.get(), "auto")
        self._lang_code_by_display.clear()
        self._display_by_lang_code.clear()
        displays = []
        for code, name_en, name_it in LANGUAGES:
            name = name_it if self.lang == "it" else name_en
            display = name if code == "auto" else f"{name} ({code})"
            displays.append(display)
            self._lang_code_by_display[display] = code
            self._display_by_lang_code[code] = display
        self.lang_combo.configure(values=displays)
        target_display = self._display_by_lang_code.get(current_code, displays[0])
        self.lang_display_var.set(target_display)

    def _on_lang_combo_selected(self, _event=None):
        pass  # value already bound via textvariable

    def _current_language_code(self):
        return self._lang_code_by_display.get(self.lang_display_var.get(), "auto")

    # ---------------------------- STYLE COMBO ----------------------------

    def _rebuild_style_combo(self):
        current_key = self._style_key_by_display.get(self.style_display_var.get(), "none")
        self._style_key_by_display.clear()
        self._display_by_style_key.clear()
        displays = []
        for preset in STYLE_PRESETS:
            label = preset["label_it"] if self.lang == "it" else preset["label_en"]
            displays.append(label)
            self._style_key_by_display[label] = preset["key"]
            self._display_by_style_key[preset["key"]] = label
        self.style_combo.configure(values=displays)
        target_display = self._display_by_style_key.get(current_key, displays[0])
        self.style_display_var.set(target_display)
        self._sync_custom_style_visibility()

    def _on_style_combo_selected(self, _event=None):
        self._sync_custom_style_visibility()

    def _current_style_key(self):
        return self._style_key_by_display.get(self.style_display_var.get(), "none")

    def _sync_custom_style_visibility(self):
        if self._current_style_key() == "custom":
            self.initial_prompt_custom_label.pack(anchor="w", pady=(8, 0))
            self.initial_prompt_custom_entry.pack(fill="x", pady=(2, 0), before=self.prefix_label)
        else:
            self.initial_prompt_custom_label.pack_forget()
            self.initial_prompt_custom_entry.pack_forget()

    def _effective_initial_prompt(self):
        key = self._current_style_key()
        if key == "custom":
            return self.initial_prompt_custom_var.get()
        preset = next((p for p in STYLE_PRESETS if p["key"] == key), None)
        return preset["prompt"] if preset else ""

    # ---------------------------- LOGIC ----------------------------

    def _log(self, text: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_box.configure(state="normal")
        self.log_box.insert("end", f"[{ts}] {text}\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _current_device(self):
        return self.device_var.get()

    def _sync_device_combo_display(self):
        display = self.t("device_gpu") if self.device_var.get() == "cuda" else self.t("device_cpu")
        self.device_combo.set(display)

    def _current_compute_types(self):
        return COMPUTE_TYPES_GPU if self._current_device() == "cuda" else COMPUTE_TYPES_CPU

    def _refresh_precision_combo_targets(self):
        types = self._current_compute_types()
        self.precision_scale.configure(to=len(types) - 1)
        default_idx = min(2, len(types) - 1)
        self.precision_var.set(default_idx)
        self._refresh_precision_label()

    def _on_device_combo_selected(self, _event=None):
        chosen_gpu = self.device_combo.get() == self.t("device_gpu")
        if chosen_gpu and not self.has_cuda:
            messagebox.showwarning(self.t("no_gpu_title"), self.t("no_gpu_msg"))
            self.device_var.set("cpu")
            self._sync_device_combo_display()
            return
        self.device_var.set("cuda" if chosen_gpu else "cpu")
        self._on_device_changed()

    def _on_device_changed(self):
        is_cpu = self._current_device() == "cpu"
        self.cpu_threads_spin.configure(state="normal" if is_cpu else "disabled")
        self._refresh_precision_combo_targets()
        self._update_estimate()

    def _on_ability_scaled(self, raw_value):
        val = int(round(float(raw_value)))
        if self.ability_var.get() != val:
            self.ability_var.set(val)
        self._refresh_model_label()
        self._update_estimate()

    def _on_precision_scaled(self, raw_value):
        val = int(round(float(raw_value)))
        if self.precision_var.get() != val:
            self.precision_var.set(val)
        self._refresh_precision_label()
        self._update_estimate()

    def _refresh_model_label(self):
        m = MODEL_TABLE[self.ability_var.get()]
        desc = m["desc_it"] if self.lang == "it" else m["desc_en"]
        params_word = "parametri" if self.lang == "it" else "parameters"
        self.ability_label.configure(
            text=f"{m['label']} ({m['name']}) \u2014 {m['params']} {params_word}.\n{desc}"
        )

    def _refresh_precision_label(self):
        types = self._current_compute_types()
        idx = min(self.precision_var.get(), len(types) - 1)
        ct = types[idx]
        label = ct["label_it"] if self.lang == "it" else ct["label_en"]
        self.precision_label.configure(text=f"{label}  (compute_type=\"{ct['name']}\")")

    def _update_estimate(self):
        if not hasattr(self, "estimate_label"):
            return
        m = MODEL_TABLE[self.ability_var.get()]
        types = self._current_compute_types()
        idx = min(self.precision_var.get(), len(types) - 1)
        ct = types[idx]
        gb = estimate_memory_gb(m, ct)

        device = self._current_device()
        if device == "cuda":
            avail = self.vram_gb
            unit = "VRAM"
        else:
            avail = self.ram_gb
            unit = "RAM"

        warn = ""
        if avail and gb > avail * 0.85:
            warn = self.t("estimate_warn")
        text = (
            f"{self.t('estimate_prefix', unit=unit)}: ~{gb} GB "
            f"({self.t('estimate_available')}: ~{avail} GB){warn}"
        )
        self.estimate_label.configure(text=text, foreground=(self.palette["danger"] if warn else self.palette["success"]))

    def _auto_suggest(self):
        if self.has_cuda and self.vram_gb > 0:
            device_data = "cuda"
            budget = self.vram_gb * 0.8
            types = COMPUTE_TYPES_GPU
        else:
            device_data = "cpu"
            budget = self.ram_gb * 0.5
            types = COMPUTE_TYPES_CPU

        best = None
        for m_idx in range(len(MODEL_TABLE) - 1, -1, -1):
            model = MODEL_TABLE[m_idx]
            for c_idx in range(len(types) - 1, -1, -1):
                gb = estimate_memory_gb(model, types[c_idx])
                if gb <= budget:
                    best = (m_idx, c_idx)
                    break
            if best:
                break
        if best is None:
            best = (0, 0)

        self.device_var.set(device_data)
        self._sync_device_combo_display()
        self._on_device_changed()
        self.ability_var.set(best[0])
        self.precision_var.set(best[1])
        self._refresh_model_label()
        self._refresh_precision_label()
        self._update_estimate()
        self._log(self.t("suggest_applied"))

    def _on_source_toggled(self):
        is_file = self.source_var.get() == "file"
        self.file_path_entry.configure(state="normal" if is_file else "disabled")
        self.file_browse_btn.configure(state="normal" if is_file else "disabled")
        self.url_entry.configure(state="disabled" if is_file else "normal")

    def _browse_file(self):
        path = filedialog.askopenfilename(
            title=self.t("select_file_dialog"),
            filetypes=[
                ("Media files", "*.mp3 *.wav *.m4a *.flac *.ogg *.mp4 *.mkv *.mov *.avi *.webm"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.file_path_var.set(path)

    def _browse_out_dir(self):
        path = filedialog.askdirectory(title=self.t("select_out_dialog"))
        if path:
            self.out_dir_var.set(path)

    def _set_running(self, running: bool):
        self.start_btn.configure(state="disabled" if running else "normal")
        self.stop_btn.configure(state="normal" if running else "disabled")
        if running:
            self.progress.pack(fill="x", pady=(8, 0), before=self.log_box)
            self.progress.start(12)
        else:
            self.progress.stop()
            self.progress.pack_forget()

    def _start(self):
        out_dir = self.out_dir_var.get().strip()
        if not out_dir:
            messagebox.showwarning(self.t("missing_output_title"), self.t("missing_output_msg"))
            return
        os.makedirs(out_dir, exist_ok=True)

        if not ensure_ffmpeg_available():
            messagebox.showwarning(self.t("no_ffmpeg_title"), self.t("no_ffmpeg_msg"))

        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")

        # Fresh cancel flag for this run -- a leftover set() from a prior
        # cancelled job would otherwise make the very next job cancel itself
        # instantly on its first checkpoint.
        self.cancel_event = threading.Event()
        self._set_running(True)

        if self.source_var.get() == "url":
            url = self.url_var.get().strip()
            if not url:
                messagebox.showwarning(self.t("missing_url_title"), self.t("missing_url_msg"))
                self._set_running(False)
                return
            download_dir = os.path.join(out_dir, "downloads")
            self._log(self.t("starting_download", url=url))
            self.active_worker = "download"
            worker = DownloadWorker(url, download_dir, self.result_queue, self.cancel_event)
            self.current_worker_thread = worker
            worker.start()
        else:
            path = self.file_path_var.get().strip()
            if not path or not os.path.isfile(path):
                messagebox.showwarning(self.t("missing_file_title"), self.t("missing_file_msg"))
                self._set_running(False)
                return
            self._begin_transcription(path)

    def _stop(self):
        # Cooperative cancellation only: this signals the running worker's
        # next checkpoint (yt-dlp hook call, or the next decoded segment) to
        # unwind. It cannot interrupt a single blocking call already inside
        # a C/native extension (e.g. mid-write of one whisper decode step),
        # so there can be a short delay between clicking Stop and the
        # "cancelled" message actually arriving.
        if self.active_worker is None:
            return
        self.cancel_event.set()
        self.stop_btn.configure(state="disabled")
        self._log(self.t("stopping"))

    def _begin_transcription(self, audio_path: str):
        self.current_audio_path = audio_path
        m = MODEL_TABLE[self.ability_var.get()]
        types = self._current_compute_types()
        ct = types[min(self.precision_var.get(), len(types) - 1)]
        device = self._current_device()
        lang = self._current_language_code()
        initial_prompt = self._effective_initial_prompt()
        prefix = self.prefix_var.get()
        hotwords = self.hotwords_var.get()
        vad = self.vad_var.get()
        beam = self.beam_var.get()
        threads = self.cpu_threads_var.get()

        self._log(self.t("model_device_line", model=m["name"], device=device, compute=ct["name"]))
        self.active_worker = "transcribe"
        # Downloading a URL enables the Stop button before this method runs;
        # make sure it's still enabled now that we've moved into the
        # transcription phase (in case a fast download disabled it).
        self.stop_btn.configure(state="normal")
        worker = TranscribeWorker(
            audio_path=audio_path,
            model_name=m["name"],
            device=device,
            compute_type=ct["name"],
            language=lang,
            initial_prompt=initial_prompt,
            prefix=prefix,
            hotwords=hotwords,
            vad_filter=vad,
            beam_size=beam,
            cpu_threads=threads,
            result_queue=self.result_queue,
            cancel_event=self.cancel_event,
        )
        self.current_worker_thread = worker
        worker.start()

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.result_queue.get_nowait()
                if kind == "log":
                    self._log(payload)
                elif kind == "download_done":
                    self.active_worker = None
                    self._log(self.t("audio_ready", path=payload))
                    self._begin_transcription(payload)
                elif kind == "transcribe_done":
                    self.active_worker = None
                    segments, meta = payload
                    self._on_transcribe_finished(segments, meta)
                elif kind == "cancelled":
                    self.active_worker = None
                    self.current_worker_thread = None
                    self._set_running(False)
                    self._log(self.t("cancelled_log"))
                elif kind == "failed":
                    self.active_worker = None
                    self._on_worker_failed(payload)
        except queue.Empty:
            pass
        finally:
            self.after(100, self._poll_queue)

    def _on_transcribe_finished(self, segments, meta):
        self._set_running(False)
        self.last_segments = segments
        self.last_meta = meta

        out_dir = self.out_dir_var.get().strip()
        base = os.path.splitext(os.path.basename(self.current_audio_path))[0]
        written = []

        if self.fmt_txt_var.get():
            txt_path = os.path.join(out_dir, base + ".txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                for seg in segments:
                    f.write(seg.text.strip() + "\n")
            written.append(txt_path)

        if self.fmt_srt_var.get():
            srt_path = os.path.join(out_dir, base + ".srt")
            with open(srt_path, "w", encoding="utf-8") as f:
                for i, seg in enumerate(segments, start=1):
                    f.write(f"{i}\n")
                    f.write(f"{format_timestamp_srt(seg.start)} --> {format_timestamp_srt(seg.end)}\n")
                    f.write(seg.text.strip() + "\n\n")
            written.append(srt_path)

        if self.fmt_vtt_var.get():
            vtt_path = os.path.join(out_dir, base + ".vtt")
            with open(vtt_path, "w", encoding="utf-8") as f:
                f.write("WEBVTT\n\n")
                for seg in segments:
                    f.write(f"{format_timestamp_vtt(seg.start)} --> {format_timestamp_vtt(seg.end)}\n")
                    f.write(seg.text.strip() + "\n\n")
            written.append(vtt_path)

        self._log(self.t("done_files_log"))
        for p in written:
            self._log(f"  - {p}")

        messagebox.showinfo(self.t("done_title"), self.t("done_msg") + "\n".join(written))

    def _on_worker_failed(self, message: str):
        self._set_running(False)
        self._log("ERROR: " + message)
        messagebox.showerror(self.t("error_title"), message)


def main():
    app = WhisperGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
