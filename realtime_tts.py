# realtime_tts.py — RTTS enterprise: no-overlap, instant fallback, direct SAPI5 speak
import threading, time, platform, tempfile, os
from io import BytesIO

import numpy as np
import soundfile as sf
import sounddevice as sd

# TTS utama (bisa error karena torch.load CVE)
try:
    from tts_model import synthesize_speech as synth_torch
except Exception:
    synth_torch = None

# ========= Dropping queue (latest wins) =========
class _DroppingQueue:
    def __init__(self):
        self._item = None
        self._cv = threading.Condition()

    def put_nowait(self, item):
        with self._cv:
            self._item = item
            self._cv.notify()

    def get(self, timeout=None):
        with self._cv:
            if self._item is None:
                self._cv.wait(timeout=timeout)
            item, self._item = self._item, None
            return item

    def flush(self):
        with self._cv:
            self._item = None

speak_q = _DroppingQueue()

# ========= Audio helpers (untuk jalur torch saja) =========
def _pick_output_device():
    try:
        devices = sd.query_devices()
    except Exception:
        return None, None
    candidates = []
    for idx, d in enumerate(devices):
        if d.get("max_output_channels", 0) > 0:
            candidates.append((idx, d.get("name","")))
    preferred = ["speaker", "speakers", "headphone", "headphones", "realtek", "high definition audio", "wasapi", "g733"]
    for idx, name in candidates:
        if any(p in name.lower() for p in preferred):
            return idx, name
    return (candidates[0] if candidates else (None, None))

def _play_wav_bytes_blocking(buf: BytesIO, out_idx=None, pause_check=None):
    """Play audio dengan kemampuan untuk di-interrupt oleh pause_check"""
    buf.seek(0)
    data, sr = sf.read(buf, dtype="float32")
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    
    # Only set OUTPUT device to avoid collision with webcam input
    if out_idx is not None:
        try:
            cur = sd.default.device
        except Exception:
            cur = (None, None)
        sd.default.device = (cur[0], out_idx)
    
    # Play dengan callback untuk check pause
    if pause_check is not None:
        # Non-blocking play dengan manual wait + pause check
        sd.play(data, sr, device=out_idx, blocking=False)
        while sd.get_stream().active:
            if pause_check():
                sd.stop()  # Stop audio immediately
                print("[RTTS] Audio interrupted by pause")
                return
            time.sleep(0.05)
    else:
        # Blocking biasa
        sd.play(data, sr, device=out_idx, blocking=True)

def _tone(duration=0.12, freq=990, sr=24000):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = 0.15 * np.sin(2*np.pi*freq*t).astype(np.float32)
    return y.reshape(-1,1), sr

def _beep(out_idx=None):
    try:
        y, sr = _tone()
        sd.play(y, sr, device=out_idx, blocking=True)
    except Exception:
        pass

# ========= pyttsx3 direct speak (tanpa WAV, tanpa sounddevice) =========
_pyttsx3_engine = None
_pyttsx3_lock = threading.Lock()

def _init_pyttsx3():
    global _pyttsx3_engine
    if _pyttsx3_engine is not None:
        return _pyttsx3_engine
    import pyttsx3
    eng = pyttsx3.init()  # Windows: SAPI5, output ke Default Playback Device OS
    try:
        rate = eng.getProperty('rate')
        eng.setProperty('rate', min(220, rate + 40))
        vol = eng.getProperty('volume')
        eng.setProperty('volume', min(1.0, vol))
    except Exception:
        pass
    _pyttsx3_engine = eng
    return eng

def _sapi5_speak_blocking(text: str, pause_check=None):
    """SAPI5 speak dengan interrupt capability"""
    with _pyttsx3_lock:
        eng = _init_pyttsx3()
        
        # Jika ada pause check dan sudah di-pause, skip
        if pause_check and pause_check():
            return
            
        # Start speaking in separate thread untuk bisa di-interrupt
        speak_done = threading.Event()
        
        def speak_thread():
            try:
                eng.say(text)
                eng.runAndWait()
            except:
                pass
            finally:
                speak_done.set()
        
        t = threading.Thread(target=speak_thread, daemon=True)
        t.start()
        
        # Wait dengan pause check
        if pause_check:
            while not speak_done.is_set():
                if pause_check():
                    try:
                        eng.stop()  # Stop pyttsx3
                        print("[RTTS] SAPI5 interrupted by pause")
                    except:
                        pass
                    return
                time.sleep(0.05)
        else:
            speak_done.wait()

# ========= Worker =========
_worker = None
_stop_evt = threading.Event()
_pause_evt = threading.Event()  # NEW: untuk pause/resume tanpa stop thread
_out_idx = None
_out_name = None

def _is_paused():
    """Helper function untuk check pause status"""
    return _pause_evt.is_set()

def _worker_loop(min_gap=0.0):
    global _out_idx, _out_name

    # Pilih output utk jalur torch (pyttsx3 tidak butuh ini)
    try:
        _out_idx, _out_name = _pick_output_device()
        if _out_idx is not None:
            print(f"[RTTS] Output device: #{_out_idx} — '{_out_name}'")
        else:
            print("[RTTS] Tidak ada output device audio yang valid untuk jalur torch.")
    except Exception as e:
        print(f"[RTTS] Gagal memilih device audio: {e}")

    # Health check: bicara "ready" via fallback pipeline paling kuat
    try:
        print("[RTTS] speak -> ready")
        try:
            # coba torch dulu jika ada dan tidak error
            if synth_torch is not None:
                wav = synth_torch("ready")
                _play_wav_bytes_blocking(wav, out_idx=_out_idx)
            else:
                _sapi5_speak_blocking("ready")
        except Exception:
            # kalau torch gagal, pasti lewat SAPI5
            _sapi5_speak_blocking("ready")
        print("[RTTS] done  -> ready")
    except Exception as e:
        print(f"[RTTS] Health check gagal: {e}")
        _beep(out_idx=_out_idx)

    last_spoken = {}
    while not _stop_evt.is_set():
        try:
            # Cek apakah sedang di-pause
            if _pause_evt.is_set():
                time.sleep(0.1)
                continue
            
            text = speak_q.get(timeout=0.25)
            if not text:
                continue
            
            # Cek lagi sebelum speak (case: baru saja di-pause saat ambil dari queue)
            if _pause_evt.is_set():
                continue
                
            now = time.time()
            if min_gap > 0 and now - last_spoken.get(text, 0.0) < min_gap:
                continue

            print(f"[RTTS] speak -> {text}")
            try:
                # 1) jalur torch (kalau masih jalan)
                if synth_torch is not None:
                    try:
                        wav = synth_torch(text)
                        # Pass pause check function
                        _play_wav_bytes_blocking(wav, out_idx=_out_idx, pause_check=_is_paused)
                    except Exception as e:
                        print(f"[RTTS] Torch TTS gagal: {e}")
                        # 2) fallback langsung SAPI5
                        _sapi5_speak_blocking(text, pause_check=_is_paused)
                else:
                    # langsung SAPI5
                    _sapi5_speak_blocking(text, pause_check=_is_paused)
                
                # Hanya print done jika tidak di-pause
                if not _pause_evt.is_set():
                    print(f"[RTTS] done  -> {text}")
            except Exception as e:
                print(f"[RTTS] Playback error: {e}")
                _beep(out_idx=_out_idx)

            last_spoken[text] = time.time()
        except Exception:
            time.sleep(0.02)

def start_tts_worker(min_gap=0.0):
    """Start worker sekali saja. Default tanpa throttle agar cepat terdengar."""
    global _worker
    if _worker and _worker.is_alive():
        return
    _stop_evt.clear()
    _pause_evt.clear()  # pastikan tidak dalam keadaan pause
    _worker = threading.Thread(target=_worker_loop, kwargs=dict(min_gap=min_gap), daemon=True)
    _worker.start()

def stop_tts_worker(flush=True):
    if flush:
        try:
            speak_q.flush()
        except Exception:
            pass
    _stop_evt.set()

def pause_tts():
    """Pause TTS worker tanpa menghentikan thread. Queue akan di-flush dan audio di-stop."""
    _pause_evt.set()
    try:
        speak_q.flush()
        # Stop semua audio yang sedang playing
        try:
            sd.stop()
        except:
            pass
    except Exception:
        pass
    print("[RTTS] TTS paused, queue flushed, and audio stopped")

def resume_tts():
    """Resume TTS worker."""
    _pause_evt.clear()
    print("[RTTS] TTS resumed")

# ====== Debug utilities ======
def rtts_debug_dump():
    try:
        devices = sd.query_devices()
        print("=== Audio Devices ===")
        for i, d in enumerate(devices):
            print(f"#{i:02d} out_ch={d.get('max_output_channels',0)} in_ch={d.get('max_input_channels',0)} :: {d.get('name','')}")
        print("=====================")
        print(f"[RTTS] selected torch-out: index={_out_idx} name={_out_name}")
    except Exception as e:
        print(f"[RTTS] query_devices error: {e}")

def rtts_self_test():
    try:
        start_tts_worker()
        for w in ["testing", "satu", "dua"]:
            speak_q.put_nowait(w)
            time.sleep(0.4)
        time.sleep(3.0)
    except Exception as e:
        print(f"[RTTS] self_test error: {e}")