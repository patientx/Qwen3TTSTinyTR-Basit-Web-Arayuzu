"""Local web UI for erkamk/qwen3-tts-tiny-tr. Stdlib HTTP server"""
import argparse
import io
import json
import re
import sys
import tempfile
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlparse, parse_qs

import numpy as np
import soundfile as sf
import types, importlib.machinery

# Heavy modules are imported in the background thread (see background_startup)
# so the web UI can come up immediately.
torch = None
Qwen3TTSModel = None


def _stub_unused_modules():
    """disable flash_attn and sox at import time"""

    def make(name, **attrs):
        mod = types.ModuleType(name)
        mod.__spec__ = importlib.machinery.ModuleSpec(name, None)  # keeps importlib.util.find_spec() happy
        mod.__dict__.update(attrs)
        sys.modules[name] = mod
        return mod

    # flash_attn: a None function is exactly what qwen-tts's own fallback sets,
    # so it silently takes the plain PyTorch attention path.
    if "flash_attn" not in sys.modules:
        pkg = make("flash_attn", __path__=[])
        pkg.flash_attn_interface = make("flash_attn.flash_attn_interface", flash_attn_varlen_func=None)

    # sox: only used by the 25Hz tokenizer's XVectorExtractor.
    if "sox" not in sys.modules:
        class Transformer:
            def __init__(self, *a, **k):
                raise RuntimeError("sox is stubbed out; the 25Hz tokenizer is not supported in this setup")
        make("sox", Transformer=Transformer)

    # onnxruntime: only used by the 25Hz tokenizer's XVectorExtractor.
    if "onnxruntime" not in sys.modules:
        make("onnxruntime")

    # transformers' candidate_generator imports roc_curve only for assisted
    # decoding (unused here) but drags in all of scikit-learn (~1.8s) to do it.
    if "sklearn.metrics" not in sys.modules:
        def roc_curve(*a, **k):
            raise RuntimeError("sklearn is stubbed out; assisted generation is not supported here")
        make("sklearn.metrics", roc_curve=roc_curve)



ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
VOICES = ROOT / "voices"
OUTPUTS = ROOT / "outputs"
VOICES.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)

DEFAULT_MODEL = "erkamk/qwen3-tts-tiny-tr"
MAX_UPLOAD = 25 * 1024 * 1024
MAX_JSON = 1024 * 1024
MAX_CHUNK_CHARS = 180

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".wav": "audio/wav",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
}

STATE = {
    "model": None,
    "model_id": "",
    "device": "cpu",
    "dtype": "float32",
    "cancel": False,
    "progress": {"done": 0, "total": 0},
    "ready": False,
    "stage": "Başlatılıyor…",
    "error": "",
}
GEN_LOCK = threading.Lock()
PROMPT_CACHE = {}


class ApiError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status
        self.message = message


# ---------------------------------------------------------------- text prep

ONES = ["", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz"]
TENS = ["", "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan"]
SCALES = ["", "bin", "milyon", "milyar", "trilyon"]


def _below_1000(n):
    h, rest = divmod(n, 100)
    t, o = divmod(rest, 10)
    parts = []
    if h:
        if h > 1:
            parts.append(ONES[h])
        parts.append("yüz")
    if t:
        parts.append(TENS[t])
    if o:
        parts.append(ONES[o])
    return " ".join(parts)


def int_to_tr(n):
    if n == 0:
        return "sıfır"
    groups = []
    while n:
        n, g = divmod(n, 1000)
        groups.append(g)
    if len(groups) > len(SCALES):
        return None
    words = []
    for i in range(len(groups) - 1, -1, -1):
        g = groups[i]
        if not g:
            continue
        if i == 1 and g == 1:
            words.append("bin")
        else:
            words.append(_below_1000(g))
            if i:
                words.append(SCALES[i])
    return " ".join(words)


def _number_repl(m):
    whole = int_to_tr(int(m.group(1)))
    if whole is None:
        return m.group(0)
    if m.group(2) is None:
        return whole
    digits = m.group(2)
    zeros = len(digits) - len(digits.lstrip("0"))
    frac = int_to_tr(int(digits)) if zeros < len(digits) else None
    lead = " ".join(["sıfır"] * zeros)
    spoken = " ".join(x for x in (lead, frac) if x)
    return f"{whole} virgül {spoken}"


def prepare_text(text, fix_case=True, numbers=True):
    text = text.replace("\r", "")
    if fix_case:
        text = text.replace("İ", "i").replace("I", "ı")
    if numbers:
        text = re.sub(r"(?<=\d)\.(?=\d{3}(?!\d))", "", text)
        text = re.sub(r"%\s*(\d)", r"yüzde \1", text)
        text = re.sub(r"(\d)\s*%", r"\1 yüzde", text)
        text = re.sub(r"(\d+)(?:,(\d+))?", _number_repl, text)
    return re.sub(r"[ \t]+", " ", text).strip()


SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+|\n+")
SOFT_BREAK = re.compile(r"[,;:]\s")


def split_text(text, max_chars=MAX_CHUNK_CHARS):
    pieces = [p.strip() for p in SENTENCE_SPLIT.split(text) if p and p.strip()]
    sized = []
    for p in pieces:
        while len(p) > max_chars:
            window = p[:max_chars]
            cut = 0
            for m in SOFT_BREAK.finditer(window):
                if m.end() >= max_chars * 0.4:
                    cut = m.end()
            if not cut:
                cut = window.rfind(" ")
            if cut <= 0:
                cut = max_chars
            sized.append(p[:cut].strip())
            p = p[cut:].strip()
        if p:
            sized.append(p)
    merged = []
    for p in sized:
        if merged and (len(p) < 25 or len(merged[-1]) < 25) and len(merged[-1]) + len(p) + 1 <= 100:
            merged[-1] = f"{merged[-1]} {p}"
        else:
            merged.append(p)
    return merged


# ---------------------------------------------------------------- model

def load_model(model_id, device, dtype_name):
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, dtype_name)
    print(f"Model yükleniyor: {model_id} · {device} · {dtype_name} ...", flush=True)
    t0 = time.time()
    model = Qwen3TTSModel.from_pretrained(model_id, device_map=device, dtype=dtype)
    print(f"Model hazır ({time.time() - t0:.1f} sn)", flush=True)
    STATE.update(model=model, model_id=model_id, device=device, dtype=dtype_name)


def to_numpy(wav):
    if hasattr(wav, "detach"):
        wav = wav.detach().float().cpu().numpy()
    return np.asarray(wav, dtype=np.float32).reshape(-1)


def get_prompt(name, path):
    mtime = path.stat().st_mtime_ns
    cached = PROMPT_CACHE.get(name)
    if cached and cached[0] == mtime:
        return cached[1]
    prompt = STATE["model"].create_voice_clone_prompt(ref_audio=str(path), x_vector_only_mode=True)
    PROMPT_CACHE[name] = (mtime, prompt)
    return prompt


def generate_chunk(chunk, prompt, temperature, subtalker_temperature, repetition_penalty):
    with torch.no_grad():
        wavs, rate = STATE["model"].generate_voice_clone(
            text=chunk,
            language="Auto",
            voice_clone_prompt=prompt,
            non_streaming_mode=True,
            max_new_tokens=25 + 2 * len(chunk),
            temperature=temperature,
            subtalker_temperature=subtalker_temperature,
            repetition_penalty=repetition_penalty,
        )
    return to_numpy(wavs[0]), rate


def warmup():
    """One short generation at startup so the first real request is not slow."""
    print("Isınma çalışması yapılıyor...", flush=True)
    t0 = time.time()
    try:
        existing = sorted(VOICES.glob("*.wav"), key=lambda p: p.stem.lower())
        with tempfile.TemporaryDirectory() as tmp:
            if existing:
                prompt = get_prompt(existing[0].stem, existing[0])
            else:
                ref = Path(tmp) / "warmup.wav"
                t = np.arange(24000 * 3) / 24000
                noise = np.random.default_rng(0).standard_normal(len(t))
                sf.write(str(ref), (0.1 * np.sin(2 * np.pi * 150 * t) + 0.02 * noise).astype(np.float32), 24000)
                prompt = STATE["model"].create_voice_clone_prompt(ref_audio=str(ref), x_vector_only_mode=True)
            generate_chunk("Merhaba", prompt, 0.7, 0.6, 1.0)
        print(f"Isınma tamamlandı ({time.time() - t0:.1f} sn)", flush=True)
    except Exception as exc:
        print(f"Isınma atlandı: {type(exc).__name__}: {exc}", flush=True)


def background_startup(args):
    """Import heavy libraries, load the model and warm it up without blocking the UI."""
    global torch, Qwen3TTSModel
    try:
        t0 = time.time()
        STATE["stage"] = "Kütüphaneler yükleniyor…"
        print("Kütüphaneler yükleniyor...", flush=True)
        _stub_unused_modules()
        import torch as _torch
        torch = _torch
        
        if args.threads > 0:
            torch.set_num_threads(args.threads)
        print(f"CPU thread sayısı: {torch.get_num_threads()}", flush=True)
        
        from qwen_tts import Qwen3TTSModel as _Model
        Qwen3TTSModel = _Model
        print(f"Kütüphaneler hazır ({time.time() - t0:.1f} sn)", flush=True)

        STATE["stage"] = "Model yükleniyor…"
        load_model(args.model, args.device, args.dtype)
        if not args.no_warmup:
            STATE["stage"] = "Isınma çalışması…"
            warmup()
        STATE["ready"] = True
        STATE["stage"] = ""
        print(f"Kullanıma hazır (toplam {time.time() - t0:.1f} sn)", flush=True)
    except Exception as exc:
        traceback.print_exc()
        STATE["error"] = f"{type(exc).__name__}: {exc}"
        STATE["stage"] = "Hata"


def num(req, key, default, lo, hi):
    try:
        value = float(req.get(key, default))
    except (TypeError, ValueError):
        value = default
    return min(max(value, lo), hi)


def synthesize(req):
    if STATE["error"]:
        raise ApiError(500, "Model yüklenemedi: " + STATE["error"])
    if not STATE["ready"]:
        raise ApiError(503, "Model henüz hazır değil, birkaç saniye bekleyin.")
    name = safe_name(str(req.get("voice", "")))
    path = VOICES / f"{name}.wav"
    if not name or not path.is_file():
        raise ApiError(400, "Önce bir referans ses seçin.")

    original = str(req.get("text", ""))
    text = prepare_text(original, bool(req.get("fix_case", True)), bool(req.get("numbers", True)))
    chunks = split_text(text)
    if not chunks:
        raise ApiError(400, "Metin boş.")

    temperature = num(req, "temperature", 0.7, 0.05, 2.0)
    subtalker_temperature = num(req, "subtemp", 0.6, 0.05, 2.0)
    repetition_penalty = num(req, "reppen", 1.0, 0.8, 2.0)
    pause_ms = int(num(req, "pause", 250, 0, 3000))
    seed = int(num(req, "seed", -1, -1, 2**31 - 1))

    if not GEN_LOCK.acquire(blocking=False):
        raise ApiError(409, "Başka bir üretim zaten çalışıyor.")
    try:
        STATE["cancel"] = False
        STATE["progress"] = {"done": 0, "total": len(chunks)}
        t0 = time.time()
        prompt = get_prompt(name, path)
        if seed >= 0:
            torch.manual_seed(seed)

        parts = []
        rate = 24000
        for i, chunk in enumerate(chunks):
            if STATE["cancel"]:
                raise ApiError(409, "İptal edildi.")
            wav, rate = generate_chunk(chunk, prompt, temperature, subtalker_temperature, repetition_penalty)
            if i:
                parts.append(np.zeros(int(rate * pause_ms / 1000), dtype=np.float32))
            parts.append(wav)
            STATE["progress"] = {"done": i + 1, "total": len(chunks)}

        audio = np.concatenate(parts)
        stem = time.strftime("%Y%m%d-%H%M%S") + "-" + name.replace(" ", "_")
        out_path = OUTPUTS / f"{stem}.wav"
        sf.write(str(out_path), audio, rate, subtype="PCM_16")
        elapsed = time.time() - t0
        duration = len(audio) / rate
        meta = {
            "text": original,
            "voice": name,
            "duration": round(duration, 2),
            "elapsed": round(elapsed, 2),
            "created": time.time(),
            "chunks": len(chunks),
        }
        (OUTPUTS / f"{stem}.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        return {
            **meta,
            "file": out_path.name,
            "url": f"/outputs/{quote(out_path.name)}",
            "rtf": round(elapsed / duration, 2) if duration else None,
            "text_used": text,
        }
    finally:
        STATE["progress"] = {"done": 0, "total": 0}
        GEN_LOCK.release()


# ---------------------------------------------------------------- files

def safe_name(raw):
    name = re.sub(r"[^\w\- ]", "_", raw.strip(), flags=re.UNICODE)
    return name.strip(" ._")[:60]


def list_voices():
    items = []
    for p in sorted(VOICES.glob("*.wav"), key=lambda x: x.stem.lower()):
        try:
            info = sf.info(str(p))
        except Exception:
            continue
        items.append({
            "name": p.stem,
            "duration": round(info.duration, 2),
            "url": f"/voices/{quote(p.name)}?v={p.stat().st_mtime_ns}",
        })
    return items


def save_voice(name, body):
    name = safe_name(name)
    if not name:
        raise ApiError(400, "Sese bir ad verin.")
    try:
        data, rate = sf.read(io.BytesIO(body), dtype="float32", always_2d=True)
    except Exception as exc:
        raise ApiError(400, f"Ses dosyası okunamadı: {exc}")
    mono = data.mean(axis=1)
    if len(mono) < rate // 2:
        raise ApiError(400, "Referans ses çok kısa.")
    sf.write(str(VOICES / f"{name}.wav"), mono, rate, subtype="PCM_16")
    PROMPT_CACHE.pop(name, None)
    return name


def list_outputs(limit=40):
    items = []
    files = sorted(OUTPUTS.glob("*.wav"), key=lambda x: x.stat().st_mtime, reverse=True)[:limit]
    for p in files:
        meta = {}
        side = p.with_suffix(".json")
        if side.is_file():
            try:
                meta = json.loads(side.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        if "duration" not in meta:
            try:
                meta["duration"] = round(sf.info(str(p)).duration, 2)
            except Exception:
                meta["duration"] = 0
        meta.setdefault("created", p.stat().st_mtime)
        items.append({**meta, "file": p.name, "url": f"/outputs/{quote(p.name)}"})
    return items


def resolve_in(folder, filename):
    path = (folder / Path(unquote(filename)).name).resolve()
    if path.parent != folder.resolve() or not path.is_file():
        return None
    return path


# ---------------------------------------------------------------- http

class Handler(BaseHTTPRequestHandler):
    server_version = "QwenTTSUI/1.0"

    def log_message(self, fmt, *args):
        pass

    # helpers
    def send_json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path, cache=False):
        size = path.stat().st_size
        start, end, status = 0, size - 1, 200
        rng = self.headers.get("Range")
        if rng:
            m = re.match(r"bytes=(\d*)-(\d*)$", rng.strip())
            if m:
                s, e = m.groups()
                if s == "" and e:
                    start = max(0, size - int(e))
                elif s:
                    start = int(s)
                    if e:
                        end = min(int(e), size - 1)
                if start > end or start >= size:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
                status = 206
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream"))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "max-age=3600" if cache else "no-cache")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        if self.command == "HEAD":
            self.end_headers()
            return
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                data = f.read(min(65536, remaining))
                if not data:
                    break
                self.wfile.write(data)
                remaining -= len(data)

    def read_body(self, limit):
        length = int(self.headers.get("Content-Length") or 0)
        if length > limit:
            raise ApiError(413, "Yüklenen dosya çok büyük.")
        return self.rfile.read(length)

    def dispatch(self, handler):
        try:
            handler()
        except ApiError as exc:
            self.send_json({"error": exc.message}, exc.status)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except Exception as exc:
            traceback.print_exc()
            try:
                self.send_json({"error": f"{type(exc).__name__}: {exc}"}, 500)
            except Exception:
                pass

    # verbs
    def do_GET(self):
        self.dispatch(self.handle_get)

    do_HEAD = do_GET

    def do_POST(self):
        self.dispatch(self.handle_post)

    def do_DELETE(self):
        self.dispatch(self.handle_delete)

    def handle_get(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        if path == "/api/status":
            return self.send_json({
                "model_id": STATE["model_id"],
                "device": STATE["device"],
                "dtype": STATE["dtype"],
                "ready": STATE["ready"],
                "stage": STATE["stage"],
                "error": STATE["error"],
                "busy": GEN_LOCK.locked(),
                "progress": STATE["progress"],
            })
        if path == "/api/voices":
            return self.send_json({"voices": list_voices()})
        if path == "/api/outputs":
            return self.send_json({"outputs": list_outputs()})

        for prefix, folder in (("/voices/", VOICES), ("/outputs/", OUTPUTS)):
            if path.startswith(prefix):
                f = resolve_in(folder, path[len(prefix):])
                if not f:
                    raise ApiError(404, "Bulunamadı.")
                return self.send_file(f)

        rel = "index.html" if path in ("/", "") else path.lstrip("/")
        if rel.startswith("static/"):
            rel = rel[len("static/"):]
        f = resolve_in(STATIC, rel)
        if not f:
            raise ApiError(404, "Bulunamadı.")
        return self.send_file(f)

    def handle_post(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/generate":
            try:
                req = json.loads(self.read_body(MAX_JSON) or b"{}")
            except json.JSONDecodeError:
                raise ApiError(400, "Geçersiz JSON.")
            return self.send_json(synthesize(req))
        if path == "/api/cancel":
            STATE["cancel"] = True
            return self.send_json({"ok": True})
        if path == "/api/voices":
            name = parse_qs(parsed.query).get("name", [""])[0]
            saved = save_voice(name, self.read_body(MAX_UPLOAD))
            return self.send_json({"name": saved})
        raise ApiError(404, "Bulunamadı.")

    def handle_delete(self):
        path = unquote(urlparse(self.path).path)
        for prefix, folder in (("/api/voices/", VOICES), ("/api/outputs/", OUTPUTS)):
            if path.startswith(prefix):
                key = path[len(prefix):]
                if folder is VOICES:
                    key = safe_name(key) + ".wav"
                target = resolve_in(folder, key)
                if not target:
                    raise ApiError(404, "Bulunamadı.")
                target.unlink()
                if folder is OUTPUTS:
                    target.with_suffix(".json").unlink(missing_ok=True)
                else:
                    PROMPT_CACHE.pop(target.stem, None)
                return self.send_json({"ok": True})
        raise ApiError(404, "Bulunamadı.")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="Qwen3-TTS Tiny Türkçe web arayüzü")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="HF repo id or local folder")
    ap.add_argument("--device", default="auto", help="auto, cpu, cuda:0 ...")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--no-warmup", action="store_true", help="ısınmayı atla")
    ap.add_argument("--threads", type=int, default=0, help="CPU thread sayısı (0 = varsayılan)")
    args = ap.parse_args()

    threading.Thread(target=background_startup, args=(args,), daemon=True).start()

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '') else args.host}:{args.port}/"
    print(f"Arayüz açıldı: {url}  (model arka planda yükleniyor, durdurmak için Ctrl+C)", flush=True)
    if not args.no_browser:
        threading.Timer(0.8, webbrowser.open, args=(url,)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
