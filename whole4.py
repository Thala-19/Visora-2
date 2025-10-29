# ===== Streamlit config (put this FIRST) =====
import streamlit as st

# ===== Imports =====
import os, time, tempfile
from pathlib import Path

import av
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# Optional modules you already used
import whisper
from audiorecorder import audiorecorder

# TTS
from tts_model import synthesize_speech
from realtime_tts import speak_q, start_tts_worker, stop_tts_worker, pause_tts, resume_tts
from gemini_module import gemini_get_response

import threading
from collections import deque

# ===== NEW: YOLO12n =====
from ultralytics import YOLO

# ===== Start TTS worker once =====
start_tts_worker()

# ===== App title =====
st.title("VISORA AI")

# ===== Model path =====
ROOT = Path(__file__).parent
YOLO_WEIGHTS = ROOT / "yolo12n.pt"  # pastikan file ini ada

# ===== Build detector once (global) =====
@st.cache_resource
def load_yolo_model():
    if not YOLO_WEIGHTS.exists():
        st.error(f"Model file missing: {YOLO_WEIGHTS}")
        st.stop()
    model = YOLO(str(YOLO_WEIGHTS))
    # names: dict {class_id:int -> name:str}
    names = model.names
    # Colors for classes
    colors = np.random.uniform(0, 255, size=(len(names), 3))
    return model, names, colors

yolo_model, CLASS_NAMES, Colors = load_yolo_model()

# ===== Helpers =====
def contains_what_am_i_seeing(text: str) -> bool:
    return "What am I seeing?" in text

def safe_pause_tts():
    """Hentikan TTS dengan kompatibilitas berbagai versi pause_tts()."""
    try:
        # Beberapa versi: pause_tts(flush, interrupt) sebagai argumen posisi
        pause_tts(True, True)  # kalau versimu tidak terima argumen, blok ini akan raise TypeError
    except TypeError:
        try:
            pause_tts()  # versi tanpa argumen
        except Exception:
            pass
    except Exception:
        pass

# ===== Audio Recorder UI (kept minimal; NOT TOUCHED) =====
audio = audiorecorder("🎤 Record Your Voice")

if len(audio) > 0:
    st.audio(audio.export().read(), format="audio/wav")

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        audio.export(tmp.name, format="wav")
        temp_audio_path = tmp.name

    # Transcribe with Whisper
    with st.spinner("Transcribing with Whisper..."):
        try:
            model = whisper.load_model("base")
            result = model.transcribe(temp_audio_path)
        finally:
            # Clean up temp file
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)

    st.markdown("**Transcribed Text:**")
    st.markdown(result["text"])

    if contains_what_am_i_seeing(result["text"]):
        audio_bytes = synthesize_speech("Describing what I see.")
        st.audio(audio_bytes)
    else:
        response = gemini_get_response()
        st.markdown(response)
        response_audio = synthesize_speech(response)
        st.audio(response_audio)

# ===== Video Transformer (real-time detect + TTS enqueue) =====
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        # draw state
        self.last_time = {}
        self.cooldown = 2.0  # Increased cooldown to reduce TTS spam
        self.fps_hist = deque(maxlen=30)
        self.t_last = time.time()

        # shared frame/result buffers
        self._lock = threading.Lock()
        self._latest_frame = None
        self._last_annotated = None
        self._labels_in_last_result = set()

        # Control flags
        self._stop = False
        self._processing = False
        
        # start detector thread
        self._worker = threading.Thread(target=self._detector_loop, daemon=True)
        self._worker.start()

    def _detector_loop(self):
        """Runs in background; pulls the newest frame and updates last result."""
        conf_thr = 0.5

        while not self._stop:
            try:
                with self._lock:
                    if self._latest_frame is None or self._processing:
                        time.sleep(0.05)
                        continue
                    frame = self._latest_frame.copy()
                    self._processing = True

                # Run YOLO inference directly on original frame
                # imgsz diatur agar tetap ringan tapi akurat
                results = yolo_model.predict(source=frame, imgsz=640, conf=conf_thr, verbose=False)

                img_display = frame.copy()
                labels_in_frame = set()

                if results and len(results) > 0:
                    r = results[0]
                    if r.boxes is not None and len(r.boxes) > 0:
                        # r.boxes.xyxy: [N,4], r.boxes.conf: [N], r.boxes.cls: [N]
                        for box, score, cls in zip(r.boxes.xyxy.cpu().numpy(),
                                                   r.boxes.conf.cpu().numpy(),
                                                   r.boxes.cls.cpu().numpy()):
                            x1, y1, x2, y2 = box.astype(int)
                            cls_id = int(cls)
                            label = CLASS_NAMES.get(cls_id, str(cls_id))
                            color = Colors[cls_id % len(Colors)]
                            cv2.rectangle(img_display, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(img_display, f"{label} {score:.2f}", (x1, max(0, y1-10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                            labels_in_frame.add(label)

                # Handle TTS for new objects (limit to one per cycle)
                now = time.time()
                for label in sorted(labels_in_frame):
                    if now - self.last_time.get(label, 0) >= self.cooldown:
                        try:
                            speak_q.put_nowait(label)
                            self.last_time[label] = now
                            break
                        except:
                            pass

                with self._lock:
                    self._labels_in_last_result = labels_in_frame
                    self._last_annotated = img_display
                    self._processing = False

            except Exception as e:
                st.error(f"Detection error: {e}")
                with self._lock:
                    self._processing = False
            finally:
                time.sleep(0.1)

    def recv(self, frame):
        """Receive frame and return processed result"""
        try:
            bgr = frame.to_ndarray(format="bgr24")
            
            with self._lock:
                self._latest_frame = bgr
                if self._last_annotated is not None:
                    out = self._last_annotated.copy()
                else:
                    out = bgr.copy()

            # Add FPS overlay
            t_now = time.time()
            dt = t_now - self.t_last
            if dt > 0:
                self.fps_hist.append(1.0 / dt)
            self.t_last = t_now
            
            if self.fps_hist:
                fps = np.mean(self.fps_hist)
                cv2.putText(out, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            return av.VideoFrame.from_ndarray(out, format="bgr24")
            
        except Exception as e:
            st.error(f"Frame processing error: {e}")
            return frame

    def __del__(self):
        """Clean up when processor is destroyed"""
        self._stop = True
        if hasattr(self, '_worker') and self._worker.is_alive():
            self._worker.join(timeout=1.0)

# ===== Streamlit UI (NOT TOUCHED) =====
st.subheader("📷 Kamera Deteksi Objek")

col1, col2 = st.columns(2)
with col1:
    start = st.button("📷 Open Camera")
with col2:
    stop = st.button("🛑 Stop Camera")

# State untuk hook START/STOP merah bawaan webrtc
if "webrtc_prev_playing" not in st.session_state:
    st.session_state["webrtc_prev_playing"] = False
if "camera_active" not in st.session_state:
    st.session_state["camera_active"] = False

if start:
    st.session_state["camera_active"] = True
if stop:
    st.session_state["camera_active"] = False

if st.session_state.get("camera_active", False):
    try:
        ctx = webrtc_streamer(
            key="visora",
            video_processor_factory=VideoProcessor,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 640},
                    "height": {"ideal": 480},
                    "frameRate": {"ideal": 15}
                },
                "audio": False,
            },
            async_processing=True,
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
        )

        # ===== Hook START/STOP merah =====
        cur_playing = bool(ctx and ctx.state.playing)

        # False -> True: START merah ditekan
        if not st.session_state["webrtc_prev_playing"] and cur_playing:
            try:
                resume_tts()
            except Exception:
                pass

        # True -> False: STOP merah ditekan
        if st.session_state["webrtc_prev_playing"] and not cur_playing:
            # Samakan dengan Close Camera: hentikan TTS dan nonaktifkan kamera
            safe_pause_tts()
            st.session_state["camera_active"] = False

        st.session_state["webrtc_prev_playing"] = cur_playing
        # ================================

        if ctx.video_processor:
            st.info("📡 Kamera aktif - deteksi berjalan. TTS akan menyebut objek secara real-time.")
        else:
            st.warning("Kamera belum aktif. Klik 'Open Camera' dan izinkan akses kamera.")
            
    except Exception as e:
        st.error(f"Error starting camera: {e}")
        st.session_state["camera_active"] = False
        st.session_state["webrtc_prev_playing"] = False
        safe_pause_tts()
else:
    st.info("Klik 'Open Camera' untuk memulai deteksi objek.")

# ===== Mode Switcher =====
    col1, col2, col3 = st.columns(3)
    with col1:
        # if OCR mode is desired, st.button's name is st.button("Switch to OCR mode"), otherwise st.button's name is st.button("Switch to Object Detection mode")
        st.button("Switch to OCR mode", on_click=lambda: st.session_state.update({"app_mode": "ocr"}))
    with col2:
        st.button("Switch to Object Detection mode", on_click=lambda: st.session_state.update({"app_mode": "object_detection"}))
    with col3:
        st.button("Refresh Page", on_click=lambda: st.rerun())