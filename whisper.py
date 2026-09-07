#!/usr/bin/env python3
"""
Faster-Whisper GUI (Qt / PySide6 Edition, self-bootstrapping)
================================================================

A desktop front-end for faster-whisper, built on PySide6 (Qt for Python).


Fully autonomous startup
-------------------------
This script manages its own dependencies. On every run it:
  1. Makes sure a private virtual environment exists (created next to this
     file, or under a per-user data directory if that's not writable), so it
     works out of the box on "externally managed" systems (PEP 668 /
     Debian's `error: externally-managed-environment`) without ever touching
     the system Python or requiring --break-system-packages.
  2. Verifies the required third-party packages (including PySide6) are
     importable inside that venv, and installs/repairs them with the venv's
     own pip if not.
  3. Re-executes itself using the venv's Python interpreter.
  4. At runtime, if `ffmpeg` isn't found on PATH, it transparently falls
     back to a portable build fetched via the `static-ffmpeg` package.

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
- VAD (Voice Activity Detection) filter checkbox with a hover tooltip.
- Interface available in English and Italian (Italian by default).
- Stop button to cancel an in-progress download or transcription.

Run
---
    python3 whisper_gui_qt.py
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

_BOOTSTRAP_MARKER = "WHISPER_GUI_QT_VENV_ACTIVE"
_REQUIRED_PACKAGES = ["faster-whisper", "yt-dlp", "psutil", "static-ffmpeg", "PySide6"]
_OPTIONAL_TORCH_PACKAGE = "torch"


def _default_venv_dir() -> Path:
    """Pick a writable location for the managed venv: next to this script
    if that directory is writable, otherwise a per-user data directory.
    Overridable via the WHISPER_GUI_QT_VENV_DIR environment variable."""
    override = os.environ.get("WHISPER_GUI_QT_VENV_DIR")
    if override:
        return Path(override).expanduser().resolve()

    script_dir = Path(__file__).resolve().parent
    if os.access(script_dir, os.W_OK):
        return script_dir / ".whisper_gui_qt_venv"

    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "whisper-gui-qt" / "venv"


def _venv_python_path(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


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
    modules = ["faster_whisper", "yt_dlp", "psutil", "static_ffmpeg", "PySide6.QtWidgets"]
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

    # Note: unlike the Tkinter edition, there is no OS-level toolkit check
    # here. PySide6 is a self-contained pip wheel (it bundles Qt itself),
    # so it's installed the same way as every other dependency below --
    # no "install this via your OS package manager" step required.

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
import traceback
from datetime import datetime

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QPushButton, QComboBox, QLineEdit, QRadioButton,
    QCheckBox, QSpinBox, QSlider, QProgressBar, QPlainTextEdit, QScrollArea,
    QFileDialog, QMessageBox, QFrame,
)

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
# (Framework-agnostic -- unchanged from the Tkinter edition.)
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
        "media_filter": "Media files (*.mp3 *.wav *.m4a *.flac *.ogg *.mp4 *.mkv *.mov *.avi *.webm)",
        "all_files_filter": "All files (*.*)",
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
        "media_filter": "File multimediali (*.mp3 *.wav *.m4a *.flac *.ogg *.mp4 *.mkv *.mov *.avi *.webm)",
        "all_files_filter": "Tutti i file (*.*)",
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
# Hardware detection helpers (unchanged from the Tkinter edition)
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


class InfoDot(QLabel):
    """Small circular "i" info button with a hover tooltip -- the Qt build-in
    QToolTip mechanism replaces the custom hover-popup class the Tkinter
    edition needed (Tk has no native widget tooltip support)."""

    def __init__(self, text_getter, parent=None, size=18):
        super().__init__("i", parent)
        self.setObjectName("infoDot")
        self.setFixedSize(size, size)
        self.setAlignment(Qt.AlignCenter)
        self.setCursor(Qt.PointingHandCursor)
        self._text_getter = text_getter
        self.refresh_tooltip()

    def refresh_tooltip(self):
        self.setToolTip(self._text_getter())


# ------------------------------------------------------------------------
# Background worker threads. Each is a QThread subclass communicating back
# to the GUI thread via Qt signals -- PySide6 automatically marshals these
# across threads (queued connections), so unlike the Tkinter edition there
# is no manual queue + polling timer required.
#
# Both workers accept a threading.Event `cancel_event`. They check it at
# every safe checkpoint (yt-dlp's progress hook; each transcribed segment)
# and unwind cooperatively -- there is no way to hard-kill a thread, so
# cancellation is always "please stop soon", not "stop now".
# ------------------------------------------------------------------------

class CancelledError(Exception):
    """Raised internally to unwind a worker once cancellation is requested."""


class DownloadWorker(QThread):
    log = Signal(str)
    download_done = Signal(str)
    cancelled = Signal()
    failed = Signal(str)

    def __init__(self, url: str, out_dir: str, cancel_event: "threading.Event", parent=None):
        super().__init__(parent)
        self.url = url
        self.out_dir = out_dir
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
            self.log.emit(f"Downloading... {pct} ({speed})")
        elif d.get("status") == "finished":
            self._final_path = d.get("filename")
            self.log.emit("Download finished, extracting audio...")

    def run(self):
        try:
            import yt_dlp
        except ImportError:
            self.failed.emit("yt-dlp is not installed. Run: pip install yt-dlp")
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
                self.log.emit(f"Fetching info for: {self.url}")
                ydl.download([self.url])

            if self.cancel_event.is_set():
                raise CancelledError("cancelled after download")

            if not self._final_path:
                self.failed.emit("Download completed but no output file was captured.")
                return

            wav_path = os.path.splitext(self._final_path)[0] + ".wav"
            if os.path.exists(wav_path):
                self.download_done.emit(wav_path)
            elif os.path.exists(self._final_path):
                self.download_done.emit(self._final_path)
            else:
                self.failed.emit("Could not locate the downloaded/extracted audio file.")
        except CancelledError:
            self.cancelled.emit()
        except Exception as e:
            # yt-dlp wraps hook exceptions in its own DownloadError; if our
            # CancelledError is the root cause, still report it as a clean
            # cancellation rather than a failure.
            if self.cancel_event.is_set():
                self.cancelled.emit()
            else:
                self.failed.emit(f"Download failed: {e}\n{traceback.format_exc()}")


class TranscribeWorker(QThread):
    log = Signal(str)
    transcribe_done = Signal(object, object)  # (segments: list, meta: dict)
    cancelled = Signal()
    failed = Signal(str)

    def __init__(self, audio_path, model_name, device, compute_type,
                 language, initial_prompt, prefix, hotwords,
                 vad_filter, beam_size, cpu_threads,
                 cancel_event: "threading.Event", parent=None):
        super().__init__(parent)
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
        self.cancel_event = cancel_event

    def run(self):
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            self.failed.emit("faster-whisper is not installed. Run: pip install faster-whisper")
            return

        try:
            if self.cancel_event.is_set():
                self.cancelled.emit()
                return

            self.log.emit(
                f"Loading model '{self.model_name}' on {self.device} ({self.compute_type})..."
            )
            kwargs = dict(device=self.device, compute_type=self.compute_type)
            if self.device == "cpu":
                kwargs["cpu_threads"] = self.cpu_threads

            model = WhisperModel(self.model_name, **kwargs)

            if self.cancel_event.is_set():
                self.cancelled.emit()
                return

            lang = None if self.language == "auto" else self.language
            initial_prompt = self.initial_prompt.strip() or None
            prefix = self.prefix.strip() or None
            hotwords = self.hotwords.strip() or None

            self.log.emit("Starting transcription...")
            segments_gen, info = model.transcribe(
                self.audio_path,
                language=lang,
                initial_prompt=initial_prompt,
                prefix=prefix,
                hotwords=hotwords,
                vad_filter=self.vad_filter,
                beam_size=self.beam_size,
            )

            self.log.emit(
                f"Detected language: {info.language} (probability {info.language_probability:.2f})"
            )

            segments = []
            for seg in segments_gen:
                # segments_gen is lazy: faster-whisper decodes one segment
                # per iteration, so checking here between iterations is a
                # genuine, timely cancellation point (not just at the end).
                if self.cancel_event.is_set():
                    self.cancelled.emit()
                    return
                segments.append(seg)
                self.log.emit(
                    f"[{format_timestamp_vtt(seg.start)} -> {format_timestamp_vtt(seg.end)}] {seg.text.strip()}"
                )

            if self.cancel_event.is_set():
                self.cancelled.emit()
                return

            meta = dict(language=info.language, duration=getattr(info, "duration", None))
            self.transcribe_done.emit(segments, meta)
        except Exception as e:
            if self.cancel_event.is_set():
                self.cancelled.emit()
            else:
                self.failed.emit(f"Transcription failed: {e}\n{traceback.format_exc()}")


# ------------------------------------------------------------------------
# Main application window
# ------------------------------------------------------------------------

class WhisperGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.lang = "it"

        self.has_cuda, self.gpu_name, self.vram_gb = get_gpu_info()
        self.ram_gb = get_ram_gb()

        self.active_worker = None      # "download" | "transcribe" | None
        self.current_worker_thread = None
        self.cancel_event = threading.Event()
        self.current_audio_path = None
        self.last_segments = None
        self.last_meta = None

        self.setWindowTitle("faster-whisper GUI")
        self.resize(1180, 820)
        self.setMinimumSize(760, 480)

        self._build_ui()
        self.retranslate_ui()
        self._on_device_changed()
        self._refresh_model_label()
        self._refresh_precision_combo_targets()
        self._update_estimate()

    def t(self, key, **fmt):
        text = TR[self.lang].get(key, key)
        return text.format(**fmt) if fmt else text

    # ---------------------------- UI BUILD ----------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header bar
        header = QFrame()
        header.setObjectName("headerBar")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(18, 14, 18, 14)
        self.header_label = QLabel()
        self.header_label.setObjectName("headerTitle")
        #self.header_label.setFont(self.fonts["header"])
        header_layout.addWidget(self.header_label)
        header_layout.addStretch(1)

        self.ui_lang_label = QLabel()
        header_layout.addWidget(self.ui_lang_label)
        self.ui_lang_combo = QComboBox()
        self.ui_lang_combo.addItems(["English", "Italiano"])
        self.ui_lang_combo.setCurrentText("Italiano")
        self.ui_lang_combo.setFixedWidth(130)
        self.ui_lang_combo.currentTextChanged.connect(self._on_ui_lang_changed)
        header_layout.addWidget(self.ui_lang_combo)

        root.addWidget(header)

        # Scrollable body -- QScrollArea handles smooth, natively-performant
        # scrolling on every platform; no manual Canvas/scrollregion/wheel-
        # binding plumbing needed here (that was the laggiest part of the
        # Tkinter edition on macOS).
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        scroll.setWidget(body)
        root.addWidget(scroll, 1)

        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(14, 10, 14, 14)
        body_layout.setSpacing(14)

        left_col = QVBoxLayout()
        left_col.setSpacing(10)
        right_col = QVBoxLayout()
        right_col.setSpacing(10)
        body_layout.addLayout(left_col, 1)
        body_layout.addLayout(right_col, 1)

        # Left column: hardware -> model selection -> run panel.
        self.hardware_box = self._build_hardware_box()
        left_col.addWidget(self.hardware_box)
        self.model_box = self._build_model_box()
        left_col.addWidget(self.model_box)
        self.run_box = self._build_run_box()
        left_col.addWidget(self.run_box, 1)

        # Right column: source -> options, sitting alongside the run panel.
        self.source_box = self._build_source_box()
        right_col.addWidget(self.source_box)
        self.options_box = self._build_options_box()
        right_col.addWidget(self.options_box, 1)

    def _title_label(self, text=""):
        lbl = QLabel(text)
        lbl.setProperty("role", "title")
        return lbl

    def _hint_label(self, text=""):
        lbl = QLabel(text)
        lbl.setProperty("role", "hint")
        lbl.setWordWrap(True)
        return lbl

    def _separator(self):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        #line.setStyleSheet(f"color: {self.palette_colors['panel_border']};")
        return line

    def _build_hardware_box(self):
        box = QGroupBox()
        layout = QHBoxLayout(box)
        self.sys_label = QLabel()
        self.sys_label.setWordWrap(True)
        layout.addWidget(self.sys_label, 1)
        self.suggest_btn = QPushButton()
        self.suggest_btn.clicked.connect(self._auto_suggest)
        layout.addWidget(self.suggest_btn)
        return box

    def _build_model_box(self):
        box = QGroupBox()
        layout = QVBoxLayout(box)

        dev_row = QHBoxLayout()
        self.device_label = QLabel()
        self.device_label.setStyleSheet("font-weight: bold;")
        dev_row.addWidget(self.device_label)
        self.device_combo = QComboBox()
        self.device_combo.setFixedWidth(140)
        self.device_combo.currentIndexChanged.connect(self._on_device_combo_selected)
        dev_row.addWidget(self.device_combo)
        dev_row.addSpacing(20)
        self.cpu_threads_label = QLabel()
        dev_row.addWidget(self.cpu_threads_label)
        self.cpu_threads_spin = QSpinBox()
        self.cpu_threads_spin.setRange(1, 64)
        self.cpu_threads_spin.setValue(4)
        self.cpu_threads_spin.setFixedWidth(60)
        dev_row.addWidget(self.cpu_threads_spin)
        dev_row.addStretch(1)
        layout.addLayout(dev_row)

        layout.addWidget(self._separator())

        self.ability_title_label = self._title_label()
        self.ability_title_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.ability_title_label)
        self.ability_slider = QSlider(Qt.Horizontal)
        self.ability_slider.setRange(0, len(MODEL_TABLE) - 1)
        self.ability_slider.setValue(4)
        self.ability_slider.valueChanged.connect(self._on_ability_changed)
        layout.addWidget(self.ability_slider)
        self.ability_label = QLabel()
        self.ability_label.setWordWrap(True)
        layout.addWidget(self.ability_label)

        layout.addWidget(self._separator())

        self.precision_title_label = self._title_label()
        self.precision_title_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.precision_title_label)
        self.precision_slider = QSlider(Qt.Horizontal)
        self.precision_slider.setRange(0, len(COMPUTE_TYPES_GPU) - 1)
        self.precision_slider.setValue(2)
        self.precision_slider.valueChanged.connect(self._on_precision_changed)
        layout.addWidget(self.precision_slider)
        self.precision_label = QLabel()
        self.precision_label.setWordWrap(True)
        layout.addWidget(self.precision_label)

        self.estimate_label = QLabel()
        self.estimate_label.setWordWrap(True)
        #self.estimate_label.setFont(self.fonts["bold"])
        layout.addWidget(self.estimate_label)

        return box

    def _build_source_box(self):
        box = QGroupBox()
        layout = QVBoxLayout(box)

        self.radio_file = QRadioButton()
        self.radio_file.setChecked(True)
        self.radio_file.toggled.connect(self._on_source_toggled)
        layout.addWidget(self.radio_file)

        file_row = QHBoxLayout()
        self.file_path_edit = QLineEdit()
        file_row.addWidget(self.file_path_edit, 1)
        self.file_browse_btn = QPushButton()
        self.file_browse_btn.clicked.connect(self._browse_file)
        file_row.addWidget(self.file_browse_btn)
        layout.addLayout(file_row)

        layout.addWidget(self._separator())

        self.radio_url = QRadioButton()
        layout.addWidget(self.radio_url)
        self.url_edit = QLineEdit()
        self.url_edit.setEnabled(False)
        layout.addWidget(self.url_edit)

        layout.addWidget(self._separator())

        out_row = QHBoxLayout()
        self.output_folder_label = QLabel()
        self.output_folder_label.setStyleSheet("font-weight: bold;")
        out_row.addWidget(self.output_folder_label)
        out_row.addStretch(1)
        self.out_browse_btn = QPushButton()
        self.out_browse_btn.clicked.connect(self._browse_out_dir)
        out_row.addWidget(self.out_browse_btn)
        layout.addLayout(out_row)
        self.out_dir_edit = QLineEdit(os.path.join(os.path.expanduser("~"), "whisper_output"))
        layout.addWidget(self.out_dir_edit)

        return box

    def _build_options_box(self):
        box = QGroupBox()
        layout = QVBoxLayout(box)

        lang_row = QHBoxLayout()
        self.language_label = QLabel()
        self.language_label.setStyleSheet("font-weight: bold;")
        lang_row.addWidget(self.language_label)
        self.lang_combo = QComboBox()
        self.lang_combo.setMinimumWidth(220)
        lang_row.addWidget(self.lang_combo)
        lang_row.addStretch(1)
        layout.addLayout(lang_row)

        layout.addWidget(self._separator())

        self.steering_title_label = self._title_label()
        layout.addWidget(self.steering_title_label)

        # initial_prompt, driven by a named "style" instead of raw example
        # text -- selecting a style resolves to a fitting example prompt
        # behind the scenes. A "Custom" style keeps the raw-text route open.
        self.initial_prompt_label = QLabel()
        self.initial_prompt_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.initial_prompt_label)
        self.style_combo = QComboBox()
        self.style_combo.currentIndexChanged.connect(self._on_style_combo_selected)
        layout.addWidget(self.style_combo)
        self.initial_prompt_hint_label = self._hint_label()
        layout.addWidget(self.initial_prompt_hint_label)

        self.initial_prompt_custom_label = QLabel()
        layout.addWidget(self.initial_prompt_custom_label)
        self.initial_prompt_custom_edit = QLineEdit()
        layout.addWidget(self.initial_prompt_custom_edit)

        # prefix -- forced verbatim start
        self.prefix_label = QLabel()
        self.prefix_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.prefix_label)
        self.prefix_edit = QLineEdit()
        layout.addWidget(self.prefix_edit)
        self.prefix_hint_label = self._hint_label()
        layout.addWidget(self.prefix_hint_label)

        # hotwords -- decoding boost, no context cost
        self.hotwords_label = QLabel()
        self.hotwords_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.hotwords_label)
        self.hotwords_edit = QLineEdit()
        layout.addWidget(self.hotwords_edit)
        self.hotwords_hint_label = self._hint_label()
        layout.addWidget(self.hotwords_hint_label)

        layout.addWidget(self._separator())

        opts_row = QHBoxLayout()
        self.vad_checkbox = QCheckBox()
        self.vad_checkbox.setStyleSheet("font-weight: bold;")
        self.vad_checkbox.setChecked(True)
        opts_row.addWidget(self.vad_checkbox)
        self.vad_info_btn = InfoDot(lambda: self.t("vad_tooltip"))
        opts_row.addWidget(self.vad_info_btn)
        opts_row.addSpacing(18)
        self.beam_label = QLabel()
        opts_row.addWidget(self.beam_label)
        self.beam_spin = QSpinBox()
        self.beam_spin.setRange(1, 10)
        self.beam_spin.setValue(5)
        self.beam_spin.setFixedWidth(60)
        opts_row.addWidget(self.beam_spin)
        opts_row.addStretch(1)
        layout.addLayout(opts_row)

        layout.addWidget(self._separator())

        fmt_row = QHBoxLayout()
        self.formats_label = QLabel()
        self.formats_label.setStyleSheet("font-weight: bold;")
        fmt_row.addWidget(self.formats_label)
        self.fmt_txt_check = QCheckBox(".txt")
        self.fmt_txt_check.setChecked(True)
        fmt_row.addWidget(self.fmt_txt_check)
        self.fmt_srt_check = QCheckBox(".srt")
        self.fmt_srt_check.setChecked(True)
        fmt_row.addWidget(self.fmt_srt_check)
        self.fmt_vtt_check = QCheckBox(".vtt")
        fmt_row.addWidget(self.fmt_vtt_check)
        fmt_row.addStretch(1)
        layout.addLayout(fmt_row)

        layout.addStretch(1)
        self._sync_custom_style_visibility()
        return box

    def _build_run_box(self):
        box = QGroupBox()
        layout = QVBoxLayout(box)

        btn_row = QHBoxLayout()
        self.start_btn = QPushButton()
        self.start_btn.setObjectName("accentButton")
        self.start_btn.clicked.connect(self._start)
        btn_row.addWidget(self.start_btn, 1)
        self.stop_btn = QPushButton()
        self.stop_btn.setObjectName("dangerButton")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)
        btn_row.addWidget(self.stop_btn, 1)
        layout.addLayout(btn_row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setTextVisible(False)
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        #self.log_box.setFont(self.fonts["mono"])
        layout.addWidget(self.log_box, 1)

        return box

    # ---------------------------- TRANSLATION ----------------------------

    def retranslate_ui(self):
        self.setWindowTitle(self.t("window_title"))
        self.header_label.setText(self.t("app_header"))
        self.ui_lang_label.setText(self.t("ui_lang_label"))

        self.hardware_box.setTitle(self.t("group_hardware"))
        gpu_txt = (
            self.t("gpu_detected", name=self.gpu_name, vram=self.vram_gb)
            if self.has_cuda else self.t("gpu_none")
        )
        self.sys_label.setText(f"{gpu_txt}\n{self.t('system_ram', ram=self.ram_gb)}")
        self.suggest_btn.setText(self.t("suggest_btn"))

        self.model_box.setTitle(self.t("group_model"))
        self.device_label.setText(self.t("device_label"))
        self._rebuild_device_combo()
        self.cpu_threads_label.setText(self.t("cpu_threads_label"))
        self.ability_title_label.setText(self.t("ability_title"))
        self.precision_title_label.setText(self.t("precision_title"))

        self.source_box.setTitle(self.t("group_source"))
        self.radio_file.setText(self.t("radio_file"))
        self.radio_url.setText(self.t("radio_url"))
        self.file_browse_btn.setText(self.t("browse"))
        self.out_browse_btn.setText(self.t("browse"))
        self.output_folder_label.setText(self.t("output_folder_label"))
        self.file_path_edit.setPlaceholderText(self.t("file_placeholder"))
        self.url_edit.setPlaceholderText(self.t("url_placeholder"))

        self.options_box.setTitle(self.t("group_options"))
        self.language_label.setText(self.t("language_label"))
        self._rebuild_language_combo()

        self.steering_title_label.setText(self.t("steering_title"))
        self.initial_prompt_label.setText(self.t("initial_prompt_label"))
        self.initial_prompt_hint_label.setText(self.t("initial_prompt_hint"))
        self.initial_prompt_custom_label.setText(self.t("initial_prompt_custom_label"))
        self._rebuild_style_combo()
        self.prefix_label.setText(self.t("prefix_label"))
        self.prefix_hint_label.setText(self.t("prefix_hint"))
        self.hotwords_label.setText(self.t("hotwords_label"))
        self.hotwords_hint_label.setText(self.t("hotwords_hint"))

        self.vad_checkbox.setText(self.t("vad_checkbox"))
        self.vad_info_btn.refresh_tooltip()
        self.beam_label.setText(self.t("beam_label"))
        self.formats_label.setText(self.t("formats_label"))

        self.run_box.setTitle(self.t("group_run"))
        self.start_btn.setText(self.t("start_btn"))
        self.stop_btn.setText(self.t("stop_btn"))

        self._refresh_model_label()
        self._refresh_precision_label()
        self._update_estimate()

    def _on_ui_lang_changed(self, text):
        self.lang = "it" if text == "Italiano" else "en"
        self.retranslate_ui()

    # ---------------------------- DEVICE COMBO ----------------------------

    def _rebuild_device_combo(self):
        current_data = self.device_combo.currentData()
        self.device_combo.blockSignals(True)
        self.device_combo.clear()
        self.device_combo.addItem(self.t("device_gpu"), "cuda")
        self.device_combo.addItem(self.t("device_cpu"), "cpu")
        target = current_data or ("cuda" if self.has_cuda else "cpu")
        idx = self.device_combo.findData(target)
        self.device_combo.setCurrentIndex(idx if idx >= 0 else 1)
        self.device_combo.blockSignals(False)

    # ---------------------------- LANGUAGE COMBO ----------------------------

    def _rebuild_language_combo(self):
        current_code = self.lang_combo.currentData() or "auto"
        self.lang_combo.blockSignals(True)
        self.lang_combo.clear()
        for code, name_en, name_it in LANGUAGES:
            name = name_it if self.lang == "it" else name_en
            display = name if code == "auto" else f"{name} ({code})"
            self.lang_combo.addItem(display, code)
        idx = self.lang_combo.findData(current_code)
        self.lang_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.lang_combo.blockSignals(False)

    def _current_language_code(self):
        return self.lang_combo.currentData() or "auto"

    # ---------------------------- STYLE COMBO ----------------------------

    def _rebuild_style_combo(self):
        current_key = self.style_combo.currentData() or "none"
        self.style_combo.blockSignals(True)
        self.style_combo.clear()
        for preset in STYLE_PRESETS:
            label = preset["label_it"] if self.lang == "it" else preset["label_en"]
            self.style_combo.addItem(label, preset["key"])
        idx = self.style_combo.findData(current_key)
        self.style_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.style_combo.blockSignals(False)
        self._sync_custom_style_visibility()

    def _on_style_combo_selected(self, _index):
        self._sync_custom_style_visibility()

    def _current_style_key(self):
        return self.style_combo.currentData() or "none"

    def _sync_custom_style_visibility(self):
        is_custom = self._current_style_key() == "custom"
        self.initial_prompt_custom_label.setVisible(is_custom)
        self.initial_prompt_custom_edit.setVisible(is_custom)

    def _effective_initial_prompt(self):
        key = self._current_style_key()
        if key == "custom":
            return self.initial_prompt_custom_edit.text()
        preset = next((p for p in STYLE_PRESETS if p["key"] == key), None)
        return preset["prompt"] if preset else ""

    # ---------------------------- LOGIC ----------------------------

    def _log(self, text: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_box.appendPlainText(f"[{ts}] {text}")

    def _current_device(self):
        return self.device_combo.currentData() or "cpu"

    def _current_compute_types(self):
        return COMPUTE_TYPES_GPU if self._current_device() == "cuda" else COMPUTE_TYPES_CPU

    def _refresh_precision_combo_targets(self):
        types = self._current_compute_types()
        self.precision_slider.blockSignals(True)
        self.precision_slider.setRange(0, len(types) - 1)
        default_idx = min(2, len(types) - 1)
        self.precision_slider.setValue(default_idx)
        self.precision_slider.blockSignals(False)
        self._refresh_precision_label()

    def _on_device_combo_selected(self, _index):
        chosen_gpu = self.device_combo.currentData() == "cuda"
        if chosen_gpu and not self.has_cuda:
            QMessageBox.warning(self, self.t("no_gpu_title"), self.t("no_gpu_msg"))
            idx = self.device_combo.findData("cpu")
            self.device_combo.blockSignals(True)
            self.device_combo.setCurrentIndex(idx)
            self.device_combo.blockSignals(False)
        self._on_device_changed()

    def _on_device_changed(self):
        is_cpu = self._current_device() == "cpu"
        self.cpu_threads_spin.setEnabled(is_cpu)
        self._refresh_precision_combo_targets()
        self._update_estimate()

    def _on_ability_changed(self, _value):
        self._refresh_model_label()
        self._update_estimate()

    def _on_precision_changed(self, _value):
        self._refresh_precision_label()
        self._update_estimate()

    def _refresh_model_label(self):
        m = MODEL_TABLE[self.ability_slider.value()]
        desc = m["desc_it"] if self.lang == "it" else m["desc_en"]
        params_word = "parametri" if self.lang == "it" else "parameters"
        self.ability_label.setText(
            f"{m['label']} ({m['name']}) \u2014 {m['params']} {params_word}.\n{desc}"
        )

    def _refresh_precision_label(self):
        types = self._current_compute_types()
        idx = min(self.precision_slider.value(), len(types) - 1)
        ct = types[idx]
        label = ct["label_it"] if self.lang == "it" else ct["label_en"]
        self.precision_label.setText(f"{label}  (compute_type=\"{ct['name']}\")")

    def _update_estimate(self):
        if not hasattr(self, "estimate_label"):
            return
        m = MODEL_TABLE[self.ability_slider.value()]
        types = self._current_compute_types()
        idx = min(self.precision_slider.value(), len(types) - 1)
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
        color = "#e0a13c" if warn else "#4fbf85"
        self.estimate_label.setText(text)
        self.estimate_label.setStyleSheet(f"color: {color}; font-weight: bold;")

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

        idx = self.device_combo.findData(device_data)
        self.device_combo.blockSignals(True)
        self.device_combo.setCurrentIndex(idx if idx >= 0 else 1)
        self.device_combo.blockSignals(False)
        self._on_device_changed()

        self.ability_slider.setValue(best[0])
        self.precision_slider.setValue(best[1])
        self._refresh_model_label()
        self._refresh_precision_label()
        self._update_estimate()
        self._log(self.t("suggest_applied"))

    def _on_source_toggled(self, _checked):
        is_file = self.radio_file.isChecked()
        self.file_path_edit.setEnabled(is_file)
        self.file_browse_btn.setEnabled(is_file)
        self.url_edit.setEnabled(not is_file)

    def _browse_file(self):
        filt = f"{self.t('media_filter')};;{self.t('all_files_filter')}"
        path, _ = QFileDialog.getOpenFileName(self, self.t("select_file_dialog"), "", filt)
        if path:
            self.file_path_edit.setText(path)

    def _browse_out_dir(self):
        path = QFileDialog.getExistingDirectory(self, self.t("select_out_dialog"))
        if path:
            self.out_dir_edit.setText(path)

    def _set_running(self, running: bool):
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.progress.setVisible(running)
        if running:
            self.progress.setRange(0, 0)  # indeterminate/busy mode
        else:
            self.progress.setRange(0, 1)

    def _start(self):
        out_dir = self.out_dir_edit.text().strip()
        if not out_dir:
            QMessageBox.warning(self, self.t("missing_output_title"), self.t("missing_output_msg"))
            return
        os.makedirs(out_dir, exist_ok=True)

        if not ensure_ffmpeg_available():
            QMessageBox.warning(self, self.t("no_ffmpeg_title"), self.t("no_ffmpeg_msg"))

        self.log_box.clear()

        # Fresh cancel flag for this run -- a leftover set() from a prior
        # cancelled job would otherwise make the very next job cancel itself
        # instantly on its first checkpoint.
        self.cancel_event = threading.Event()
        self._set_running(True)

        if self.radio_url.isChecked():
            url = self.url_edit.text().strip()
            if not url:
                QMessageBox.warning(self, self.t("missing_url_title"), self.t("missing_url_msg"))
                self._set_running(False)
                return
            download_dir = os.path.join(out_dir, "downloads")
            self._log(self.t("starting_download", url=url))
            self.active_worker = "download"
            worker = DownloadWorker(url, download_dir, self.cancel_event)
            worker.log.connect(self._log)
            worker.download_done.connect(self._on_download_done)
            worker.cancelled.connect(self._on_cancelled)
            worker.failed.connect(self._on_worker_failed)
            self.current_worker_thread = worker
            worker.start()
        else:
            path = self.file_path_edit.text().strip()
            if not path or not os.path.isfile(path):
                QMessageBox.warning(self, self.t("missing_file_title"), self.t("missing_file_msg"))
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
        self.stop_btn.setEnabled(False)
        self._log(self.t("stopping"))

    def _on_download_done(self, path):
        self.active_worker = None
        self._log(self.t("audio_ready", path=path))
        self._begin_transcription(path)

    def _on_cancelled(self):
        self.active_worker = None
        self.current_worker_thread = None
        self._set_running(False)
        self._log(self.t("cancelled_log"))

    def _begin_transcription(self, audio_path: str):
        self.current_audio_path = audio_path
        m = MODEL_TABLE[self.ability_slider.value()]
        types = self._current_compute_types()
        ct = types[min(self.precision_slider.value(), len(types) - 1)]
        device = self._current_device()
        lang = self._current_language_code()
        initial_prompt = self._effective_initial_prompt()
        prefix = self.prefix_edit.text()
        hotwords = self.hotwords_edit.text()
        vad = self.vad_checkbox.isChecked()
        beam = self.beam_spin.value()
        threads = self.cpu_threads_spin.value()

        self._log(self.t("model_device_line", model=m["name"], device=device, compute=ct["name"]))
        self.active_worker = "transcribe"
        # Downloading a URL enables the Stop button before this method runs;
        # make sure it's still enabled now that we've moved into the
        # transcription phase (in case a fast download disabled it).
        self.stop_btn.setEnabled(True)
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
            cancel_event=self.cancel_event,
        )
        worker.log.connect(self._log)
        worker.transcribe_done.connect(self._on_transcribe_finished)
        worker.cancelled.connect(self._on_cancelled)
        worker.failed.connect(self._on_worker_failed)
        self.current_worker_thread = worker
        worker.start()

    def _on_transcribe_finished(self, segments, meta):
        self.active_worker = None
        self._set_running(False)
        self.last_segments = segments
        self.last_meta = meta

        out_dir = self.out_dir_edit.text().strip()
        base = os.path.splitext(os.path.basename(self.current_audio_path))[0]
        written = []

        if self.fmt_txt_check.isChecked():
            txt_path = os.path.join(out_dir, base + ".txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                for seg in segments:
                    f.write(seg.text.strip() + "\n")
            written.append(txt_path)

        if self.fmt_srt_check.isChecked():
            srt_path = os.path.join(out_dir, base + ".srt")
            with open(srt_path, "w", encoding="utf-8") as f:
                for i, seg in enumerate(segments, start=1):
                    f.write(f"{i}\n")
                    f.write(f"{format_timestamp_srt(seg.start)} --> {format_timestamp_srt(seg.end)}\n")
                    f.write(seg.text.strip() + "\n\n")
            written.append(srt_path)

        if self.fmt_vtt_check.isChecked():
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

        QMessageBox.information(self, self.t("done_title"), self.t("done_msg") + "\n".join(written))

    def _on_worker_failed(self, message: str):
        self.active_worker = None
        self._set_running(False)
        self._log("ERROR: " + message)
        QMessageBox.critical(self, self.t("error_title"), message)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # cross-platform, software-drawn style -- same
                             # rationale as forcing "clam" in the Tk edition:
                             # a fixed look independent of the native theme.
    window = WhisperGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
