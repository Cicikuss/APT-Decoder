"""
NOAA APT Decoder  Python  v3
Translated from noaa-apt (Rust) by martinber
https://github.com/martinber/noaa-apt

CHANGES v3:
  - Gercek zamanli eslesmeli ses: ses bastan oynar, GUI ticker
    her 80ms'de time.time()-baslangic ile cursor'i ilerletir.
    Decode hizi ne olursa olsun ses ve cursor birbirinden BAGIMSIZ
    ve her ikisi de gercek zamana gore calisiyor.
  - Ses backend: sounddevice (tercihli) -> winsound (Windows) ->
    simpleaudio -> subprocess/afplay-aplay (macOS/Linux) -> sessiz
  - WAV dalga formu DIKEY: Y=zaman (ust=baslangic, alt=son),
    X=genlik, kirmizi yatay cizgi = anlik konum
"""

import math
import os
import sys
import time
import threading
import queue
import argparse
import io
import wave
import subprocess
import platform
from typing import Optional

import numpy as np
import scipy.io.wavfile
import scipy.signal
from PIL import Image, ImageDraw

try:
    from PIL import ImageTk
except ImportError:
    ImageTk = None

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    TK_AVAILABLE = True
except Exception:
    tk = ttk = filedialog = messagebox = None
    TK_AVAILABLE = False

# ── Audio backends ──────────────────────────────────────────────────────────

try:
    import sounddevice as sd
    sd.query_devices()          # raises if PortAudio missing
    SD_AVAILABLE = True
except Exception:
    sd = None
    SD_AVAILABLE = False

try:
    import winsound
    WINSOUND_AVAILABLE = True
except ImportError:
    winsound = None
    WINSOUND_AVAILABLE = False

try:
    import simpleaudio as sa
    SA_AVAILABLE = True
except ImportError:
    sa = None
    SA_AVAILABLE = False


# ── AudioPlayer ──────────────────────────────────────────────────────────────

class AudioPlayer:
    """
    Plays a float32 mono array from the beginning in a background thread.
    elapsed_sec is WALL-CLOCK based so it is always accurate regardless of
    how fast/slow the decode pipeline runs.
    """

    def __init__(self, audio: np.ndarray, rate: int, source_path: Optional[str] = None):
        self._audio    = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
        self._rate     = int(rate)
        self._source_path = source_path
        self._start_ts = None          # set by start()
        self._stop_evt = threading.Event()
        self._thread   = None
        self._mem_wave_bytes = None
        self.backend_used = 'none'
        self.last_error = ''
        self.duration  = len(self._audio) / max(1, self._rate)

    def start(self):
        self._stop_evt.clear()
        self._start_ts = time.time()
        self._thread   = threading.Thread(target=self._play, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_evt.set()

    @property
    def elapsed_sec(self) -> float:
        if self._start_ts is None:
            return 0.0
        return max(0.0, min(self.duration, time.time() - self._start_ts))

    @property
    def is_playing(self) -> bool:
        return (self._thread is not None
                and self._thread.is_alive()
                and not self._stop_evt.is_set())

    # internal ----------------------------------------------------------------

    def _pcm16_bytes(self) -> bytes:
        return (self._audio * 32767.0).astype(np.int16).tobytes()

    def _play(self):
        # Try backends in order; fall back automatically on runtime failure.
        attempts = []
        sys_ = platform.system()
        if SD_AVAILABLE:
            attempts.append(('sounddevice', self._play_sd))
        if WINSOUND_AVAILABLE:
            attempts.append(('winsound', self._play_winsound))
        if SA_AVAILABLE:
            attempts.append(('simpleaudio', self._play_sa))
        if sys_ == 'Windows':
            attempts.append(('powershell', self._play_windows_powershell))
        else:
            attempts.append(('subprocess', self._play_subprocess))

        for name, fn in attempts:
            if self._stop_evt.is_set():
                return
            ok = fn()
            if ok:
                self.backend_used = name
                self.last_error = ''
                return

        self.backend_used = 'none'
        if not self.last_error:
            self.last_error = 'No audio backend could play the stream.'
        print(f"[audio] {self.last_error}")

    def _play_sd(self):
        try:
            chunk = 1024
            total = len(self._audio)
            pos   = 0
            with sd.OutputStream(samplerate=self._rate, channels=1,
                                  dtype='float32', blocksize=chunk) as stream:
                while pos < total and not self._stop_evt.is_set():
                    end = min(pos + chunk, total)
                    buf = self._audio[pos:end]
                    if len(buf) < chunk:
                        buf = np.pad(buf, (0, chunk - len(buf)))
                    stream.write(buf.reshape(-1, 1))
                    pos += chunk
            return True
        except Exception as e:
            self.last_error = f"sounddevice: {e}"
            print(f"[audio/sd] {e}")
            return False

    def _play_winsound(self):
        try:
            # Prefer direct file playback on Windows when source path is available.
            if self._source_path and os.path.exists(self._source_path):
                winsound.PlaySound(self._source_path,
                                   winsound.SND_FILENAME | winsound.SND_ASYNC)
            else:
                buf = io.BytesIO()
                with wave.open(buf, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(self._rate)
                    wf.writeframes(self._pcm16_bytes())
                # Keep bytes alive during async playback; otherwise some systems go silent.
                self._mem_wave_bytes = buf.getvalue()
                winsound.PlaySound(self._mem_wave_bytes,
                                   winsound.SND_MEMORY | winsound.SND_ASYNC)
            end_ts = self._start_ts + self.duration
            while time.time() < end_ts and not self._stop_evt.is_set():
                time.sleep(0.05)
            if self._stop_evt.is_set():
                winsound.PlaySound(None, 0)
            self._mem_wave_bytes = None
            return True
        except Exception as e:
            self.last_error = f"winsound: {e}"
            print(f"[audio/winsound] {e}")
            return False

    def _play_sa(self):
        try:
            pcm = (self._audio * 32767.0).astype(np.int16)
            obj = sa.play_buffer(pcm, 1, 2, self._rate)
            while obj.is_playing() and not self._stop_evt.is_set():
                time.sleep(0.05)
            if self._stop_evt.is_set():
                obj.stop()
            return True
        except Exception as e:
            self.last_error = f"simpleaudio: {e}"
            print(f"[audio/sa] {e}")
            return False

    def _play_subprocess(self):
        import tempfile
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                tmp = f.name
            with wave.open(tmp, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self._rate)
                wf.writeframes(self._pcm16_bytes())
            sys_ = platform.system()
            if sys_ == 'Darwin':
                cmd = ['afplay', tmp]
            elif sys_ == 'Linux':
                for player in ('aplay', 'paplay', 'mpv', 'ffplay'):
                    if subprocess.run(['which', player],
                                      capture_output=True).returncode == 0:
                        cmd = [player, tmp]
                        break
                else:
                    self.last_error = 'subprocess: no system player found'
                    print('[audio] no subprocess player found')
                    return False
            else:
                self.last_error = 'subprocess: unsupported platform'
                print('[audio] unsupported platform for subprocess playback')
                return False
            proc = subprocess.Popen(cmd,
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)
            while proc.poll() is None and not self._stop_evt.is_set():
                time.sleep(0.05)
            if self._stop_evt.is_set():
                proc.terminate()
            return True
        except Exception as e:
            self.last_error = f"subprocess: {e}"
            print(f"[audio/subprocess] {e}")
            return False
        finally:
            if tmp and os.path.exists(tmp):
                try:
                    os.unlink(tmp)
                except Exception:
                    pass

    def _play_windows_powershell(self):
        import tempfile
        tmp = None
        proc = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                tmp = f.name
            with wave.open(tmp, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self._rate)
                wf.writeframes(self._pcm16_bytes())

            ps_path = tmp.replace("'", "''")
            cmd = [
                'powershell', '-NoProfile', '-NonInteractive', '-Command',
                f"$p=New-Object System.Media.SoundPlayer('{ps_path}'); $p.PlaySync();"
            ]
            proc = subprocess.Popen(cmd,
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)
            while proc.poll() is None and not self._stop_evt.is_set():
                time.sleep(0.05)
            if self._stop_evt.is_set() and proc.poll() is None:
                proc.terminate()
            return True
        except Exception as e:
            self.last_error = f"powershell: {e}"
            print(f"[audio/powershell] {e}")
            return False
        finally:
            if proc is not None and self._stop_evt.is_set() and proc.poll() is None:
                try:
                    proc.terminate()
                except Exception:
                    pass
            if tmp and os.path.exists(tmp):
                try:
                    os.unlink(tmp)
                except Exception:
                    pass


def _backend_name() -> str:
    if SD_AVAILABLE:       return 'sounddevice'
    if WINSOUND_AVAILABLE: return 'winsound'
    if SA_AVAILABLE:       return 'simpleaudio'
    sys_ = platform.system()
    if sys_ == 'Darwin':   return 'afplay'
    if sys_ == 'Linux':    return 'aplay/paplay'
    return 'none'


# ── APTDecoder ───────────────────────────────────────────────────────────────

class APTDecoder:
    FINAL_RATE  = 4160
    WORK_RATE   = 20800
    IMAGE_WIDTH = 2080
    CARRIER     = 2400

    PX_SYNC   = 39
    PX_SPACE  = 47
    PX_IMAGE  = 909
    PX_TELEM  = 45
    PX_PER_CH = 1040

    _RESAMPLE_ATTEN      = 40.0
    _RESAMPLE_CUTOUT     = 4800.0
    _RESAMPLE_DELTA_FREQ = 500.0
    _DEMOD_ATTEN         = 25.0

    def __init__(self, filename):
        self.filename = filename
        rate, raw = scipy.io.wavfile.read(filename)
        if raw.ndim > 1:
            raw = raw[:, 0]
        if raw.dtype == np.int16:
            sig = raw.astype(np.float32) / 32768.0
        elif raw.dtype == np.int32:
            sig = raw.astype(np.float32) / 2147483648.0
        else:
            sig = raw.astype(np.float32)
        self._input_rate = int(rate)
        self.signal      = sig
        print(f"Yuklendi: {os.path.basename(filename)}  {rate} Hz  {len(sig)/rate:.1f}s")

    @staticmethod
    def _kaiser_beta(a):
        if a > 50.0:  return 0.1102 * (a - 8.7)
        if a >= 21.0: return 0.5842 * (a - 21.0) ** 0.4 + 0.07886 * (a - 21.0)
        return 0.0

    @staticmethod
    def _kaiser_numtaps(a, dHz, rate):
        dr = 2.0 * math.pi * dHz / rate
        n  = int(math.ceil((a - 8.0) / (2.285 * dr))) + 1
        return n if n % 2 == 1 else n + 1

    def _lowpass(self, cutoff, delta, rate, atten=60.0):
        n    = self._kaiser_numtaps(atten, delta, rate)
        nyq  = rate / 2.0
        beta = self._kaiser_beta(atten)
        return scipy.signal.firwin(
            n, min(cutoff, nyq * 0.9999) / nyq,
            window=('kaiser', beta)).astype(np.float32)

    def _bandpass_dc_removal(self, cutout, delta, rate, atten=60.0):
        n    = self._kaiser_numtaps(atten, delta, rate)
        nyq  = rate / 2.0
        beta = self._kaiser_beta(atten)
        return scipy.signal.firwin(
            n, [max(delta/2, 1.0)/nyq, min(cutout, nyq*0.9999)/nyq],
            window=('kaiser', beta), pass_zero=False).astype(np.float32)

    def _generate_sync_frame(self, rate):
        if rate % self.FINAL_RATE:
            raise ValueError("rate must be a multiple of FINAL_RATE")
        pw    = rate // self.FINAL_RATE
        sp    = pw * 2
        guard = [-1]*sp + ([-1]*sp + [1]*sp)*7 + [-1]*(8*pw)
        return np.array(guard, dtype=np.int8)

    @staticmethod
    def _demodulate(sig, carrier, rate):
        phi  = 2.0 * math.pi * carrier / rate
        c2   = math.cos(phi) * 2.0
        s    = abs(math.sin(phi))
        prev = sig[:-1]; curr = sig[1:]
        env  = np.sqrt(np.maximum(0.0, prev*prev + curr*curr - prev*curr*c2)) / s
        return np.concatenate([[0.0], env]).astype(np.float32)

    def _notify(self, cb, event, **kw):
        if cb: cb({'event': event, **kw})

    def _pipeline(self, progress_cb=None, cancel_cb=None):
        def should_cancel():
            return bool(cancel_cb and cancel_cb())

        rate = self._input_rate
        sig  = self.signal.copy()
        self._notify(progress_cb, 'stage', stage='pipeline', progress=0.05, text='Pipeline basladi')
        if should_cancel():
            raise InterruptedError('Decode durduruldu.')

        if rate != self.WORK_RATE:
            g   = math.gcd(rate, self.WORK_RATE)
            sig = scipy.signal.resample_poly(sig, self.WORK_RATE//g, rate//g)
            print(f"-> Resample {rate} -> {self.WORK_RATE} Hz OK")
            self._notify(progress_cb, 'stage', stage='pipeline', progress=0.20, text='Resample tamamlandi')
        else:
            self._notify(progress_cb, 'stage', stage='pipeline', progress=0.20, text='Resample gerekmedi')

        bp  = self._bandpass_dc_removal(self._RESAMPLE_CUTOUT, self._RESAMPLE_DELTA_FREQ,
                                        self.WORK_RATE, self._RESAMPLE_ATTEN)
        sig = scipy.signal.oaconvolve(sig, bp, mode='full')[:len(sig)]
        print(f"-> Bandpass OK (taps={len(bp)})")
        self._notify(progress_cb, 'stage', stage='pipeline', progress=0.40, text='Bandpass uygulandi')
        if should_cancel():
            raise InterruptedError('Decode durduruldu.')

        envelope = self._demodulate(sig, self.CARRIER, self.WORK_RATE)
        print("-> Demodulate OK")
        self._notify(progress_cb, 'stage', stage='pipeline', progress=0.58, text='Demodulasyon tamamlandi')
        if should_cancel():
            raise InterruptedError('Decode durduruldu.')

        lp_cut = float(self.FINAL_RATE) / 2.0
        lp     = self._lowpass(lp_cut, lp_cut/5.0, self.WORK_RATE, self._DEMOD_ATTEN)
        envelope = scipy.signal.oaconvolve(envelope, lp, mode='full')[:len(envelope)]
        print(f"-> Lowpass OK (taps={len(lp)})")
        self._notify(progress_cb, 'stage', stage='pipeline', progress=0.76, text='Son lowpass tamamlandi')
        if should_cancel():
            raise InterruptedError('Decode durduruldu.')

        factor   = self.WORK_RATE // self.FINAL_RATE
        envelope = envelope[::factor].astype(np.float32)
        print(f"-> Decimate x{factor} -> {self.FINAL_RATE} Hz  {len(envelope)/self.FINAL_RATE:.1f}s OK")
        self._notify(progress_cb, 'stage', stage='pipeline', progress=0.92, text='Decimate tamamlandi')
        if should_cancel():
            raise InterruptedError('Decode durduruldu.')
        return envelope

    @staticmethod
    def _digitize(sig, lo=0.5, hi=99.5):
        lo, hi = np.percentile(sig, (lo, hi))
        span   = hi - lo
        if span == 0: return np.zeros(len(sig), dtype=np.uint8)
        return np.clip(np.round(255.0*(sig-lo)/span), 0, 255).astype(np.uint8)

    def _find_sync(self, signal, progress_cb=None, cancel_cb=None):
        self._notify(progress_cb, 'stage', stage='sync', progress=0.0, text='Sync araniyor')
        guard    = self._generate_sync_frame(self.FINAL_RATE).astype(np.float32)
        sprow    = self.IMAGE_WIDTH
        min_dist = sprow * 8 // 10
        corr     = scipy.signal.correlate(signal.astype(np.float32), guard, mode='valid')
        peaks    = [(0, 0.0)]
        for i, c in enumerate(corr.tolist()):
            if cancel_cb and (i % 4096 == 0) and cancel_cb():
                raise InterruptedError('Decode durduruldu.')
            if i - peaks[-1][0] > min_dist:
                while i // sprow > len(peaks):
                    peaks.append((peaks[-1][0] + sprow, 0.0))
                peaks.append((i, c))
            elif c > peaks[-1][1]:
                peaks[-1] = (i, c)
        pos = np.array([p[0] for p in peaks[1:]], dtype=np.int64)
        print(f"-> Sync: {len(pos)} satir")
        self._notify(progress_cb, 'stage', stage='sync', progress=1.0,
                     text=f'Sync: {len(pos)} satir')
        return pos

    def _build_matrix(self, signal, sync_pos, progress_cb=None,
                      realtime_waterfall=False, realtime_speed=1.0,
                      cancel_cb=None):
        w     = self.IMAGE_WIDTH
        valid = [int(p) for p in sync_pos if int(p)+w <= len(signal)]
        if not valid: raise ValueError("Hic gecerli satir olusturulamadi!")
        total  = len(valid)
        matrix = np.empty((total, w), dtype=np.uint8)
        self._notify(progress_cb, 'stage', stage='build', progress=0.0, text='Satirlar olusturuluyor')
        rt_start = None
        rt_first_wav_sec = 0.0
        speed = max(0.05, float(realtime_speed))
        for idx, p in enumerate(valid):
            if cancel_cb and cancel_cb():
                raise InterruptedError('Decode durduruldu.')
            row = signal[p:p+w]
            matrix[idx, :] = row
            wav_sec = float(p)/float(self.FINAL_RATE)

            if realtime_waterfall:
                if rt_start is None:
                    rt_start = time.perf_counter()
                    rt_first_wav_sec = wav_sec
                target = rt_start + (wav_sec - rt_first_wav_sec) / speed
                sleep_s = target - time.perf_counter()
                if sleep_s > 0:
                    time.sleep(sleep_s)

            if progress_cb:
                self._notify(progress_cb, 'row',
                             index=idx, total=total,
                             wav_sample=p,
                             wav_sec=wav_sec,
                             row_data=row.copy(),
                             progress=float(idx+1)/float(total))
        print(f"OK Matris: {matrix.shape}")
        self._notify(progress_cb, 'stage', stage='build', progress=1.0, text='Matris tamamlandi')
        return matrix

    @staticmethod
    def _equalize(arr):
        flat = arr.flatten().astype(np.int32)
        hist, _ = np.histogram(flat, bins=256, range=(0, 255))
        cdf  = hist.cumsum()
        cmin = int(cdf[cdf > 0][0])
        den  = flat.size - cmin
        if den == 0: return arr
        lut  = np.clip(np.round(255.0*(cdf-cmin)/den), 0, 255).astype(np.uint8)
        return lut[arr]

    def _adjust_contrast(self, matrix, mode='histogram', percent=0.98):
        if mode == 'histogram': return self._equalize(matrix)
        p = (1.0 - percent) / 2.0 * 100.0
        lo, hi = (np.percentile(matrix, (p, 100-p)) if mode == 'percent'
                  else (float(matrix.min()), float(matrix.max())))
        span = hi - lo
        if span == 0: return matrix
        return np.clip(np.round(255.0*(matrix.astype(np.float64)-lo)/span),
                       0, 255).astype(np.uint8)

    def _detect_pass_direction(self, matrix):
        guard = self._generate_sync_frame(self.FINAL_RATE).astype(np.float64)
        col_n = self.PX_SYNC
        ref   = np.interp(np.linspace(0,1,col_n), np.linspace(0,1,len(guard)), guard)
        ref   = (ref - ref.mean()) / (ref.std() + 1e-12)
        sl    = max(5, matrix.shape[0] // 10)
        def score(block):
            b = block[:, :col_n].astype(np.float64)
            m = b.mean(axis=1, keepdims=True)
            s = b.std(axis=1, keepdims=True) + 1e-12
            n = min(b.shape[1], len(ref))
            return float((((b-m)/s)[:, :n] @ ref[:n]).mean()) if len(b) else 0.0
        ts, bs = score(matrix[:sl]), score(matrix[-sl:])
        flip   = bs > ts
        print(f"OK Gecis: {'flip' if flip else 'normal'}  top={ts:.3f} bot={bs:.3f}")
        return flip

    def _rotate_channels(self, matrix):
        out = matrix.copy()
        x_a = self.PX_SYNC + self.PX_SPACE
        x_b = x_a + self.PX_PER_CH
        out[:, x_a:x_a+self.PX_IMAGE] = matrix[::-1, x_a:x_a+self.PX_IMAGE]
        out[:, x_b:x_b+self.PX_IMAGE] = matrix[::-1, x_b:x_b+self.PX_IMAGE]
        return out

    def _apply_false_color(self, ch_a, ch_b, palette_file=None):
        if palette_file:
            pal = np.asarray(Image.open(palette_file).convert('RGB'), dtype=np.uint8)
            if pal.shape[:2] != (256, 256):
                raise ValueError(f"Palette 256x256 PNG olmali: {pal.shape[:2]}")
            return pal[ch_b.astype(np.uint8), ch_a.astype(np.uint8)]
        a = ch_a.astype(np.float64) / 255.0
        b = ch_b.astype(np.float64) / 255.0
        R = np.clip(a*0.7 + b*0.5, 0, 1)
        G = np.clip(a*0.6 + (1.0-np.abs(b-0.4))*0.5, 0, 1)
        B = np.clip((1.0-b)*0.8 + a*0.3, 0, 1)
        return np.stack([(R*255).astype(np.uint8),
                         (G*255).astype(np.uint8),
                         (B*255).astype(np.uint8)], axis=2)

    def decode(self, outfile=None, separate_channels=False, auto_rotate=True,
               flip=None, false_color=False, palette_file=None,
               contrast='histogram', contrast_percent=0.98,
               progress_cb=None, show_image=True,
               realtime_waterfall=False, realtime_speed=1.0,
               cancel_cb=None):
        print("=" * 60); print("DECODE BASLIYOR"); print("=" * 60)
        self._notify(progress_cb, 'stage', stage='start', progress=0.0, text='Decode basladi')

        envelope  = self._pipeline(progress_cb=progress_cb, cancel_cb=cancel_cb)
        digitized = self._digitize(envelope)
        self._notify(progress_cb, 'stage', stage='digitize', progress=1.0, text='Digitize tamamlandi')
        if cancel_cb and cancel_cb():
            raise InterruptedError('Decode durduruldu.')

        sync_pos = self._find_sync(digitized.astype(np.float32), progress_cb=progress_cb, cancel_cb=cancel_cb)
        if len(sync_pos) < 5:
            raise ValueError(f"Yeterli sync bulunamadi ({len(sync_pos)} < 5)")

        matrix = self._build_matrix(
            digitized,
            sync_pos,
            progress_cb=progress_cb,
            realtime_waterfall=realtime_waterfall,
            realtime_speed=realtime_speed,
            cancel_cb=cancel_cb,
        )

        should_flip = flip if flip is not None else (auto_rotate and self._detect_pass_direction(matrix))
        if should_flip:
            matrix = self._rotate_channels(matrix)

        ca_s = self.PX_SYNC + self.PX_SPACE
        cb_s = ca_s + self.PX_PER_CH
        kw   = dict(mode=contrast, percent=contrast_percent)

        if separate_channels or false_color:
            ch_a = self._adjust_contrast(matrix[:, ca_s:ca_s+self.PX_IMAGE], **kw)
            ch_b = self._adjust_contrast(matrix[:, cb_s:cb_s+self.PX_IMAGE], **kw)
            if false_color:
                rgb = self._apply_false_color(ch_a, ch_b, palette_file)
                img_fc = Image.fromarray(rgb, 'RGB')
                if outfile:
                    base = outfile.rsplit('.', 1)[0]
                    ext  = outfile.rsplit('.', 1)[-1] if '.' in outfile else 'png'
                    img_fc.save(f"{base}_false_color.{ext}")
                if show_image: img_fc.show()
                self._notify(progress_cb, 'result', image=rgb, text='False-color olustu')
                if not separate_channels: return rgb
            if separate_channels:
                if outfile:
                    base = outfile.rsplit('.', 1)[0]
                    ext  = outfile.rsplit('.', 1)[-1] if '.' in outfile else 'png'
                    Image.fromarray(ch_a, 'L').save(f"{base}_ch_a.{ext}")
                    Image.fromarray(ch_b, 'L').save(f"{base}_ch_b.{ext}")
                preview = np.hstack([ch_a, ch_b])
                if show_image: Image.fromarray(preview, 'L').show()
                self._notify(progress_cb, 'result', image=preview, text='Kanal goruntusu olustu')
                return ch_a, ch_b

        full = self._adjust_contrast(matrix, **kw)
        img  = Image.fromarray(full, 'L')
        if outfile: img.save(outfile)
        if show_image: img.show()
        self._notify(progress_cb, 'result', image=full, text='Goruntu olustu')
        self._notify(progress_cb, 'stage', stage='done', progress=1.0, text='Decode tamamlandi')
        return full


# ── GUI ──────────────────────────────────────────────────────────────────────

class DecoderGUI:
    WAVE_W = 100   # waveform canvas width  (amplitude axis)
    WAVE_H = 560   # waveform canvas height (time axis, top→bottom)

    def __init__(self):
        if not TK_AVAILABLE: raise RuntimeError('Tkinter bulunamadi.')
        if ImageTk is None:  raise RuntimeError('Pillow ImageTk bulunamadi.')

        self.root = tk.Tk()
        self.root.title('NOAA APT Decoder')
        self.root.geometry('1280x800')

        self.event_queue    = queue.Queue()
        self.worker         = None
        self.preview_matrix = None
        self.photo          = None
        self.last_render_ts = 0.0

        self.wave_photo        = None
        self.wave_duration_sec = 0.0
        self.wave_img_id       = None
        self.wave_cursor_id    = None
        self.wave_audio        = None
        self.wave_rate         = 0

        self._player: Optional[AudioPlayer] = None
        self._ticker_running = False
        self._audio_timeline_scale = 1.0

        self._build_ui()
        self._set_waveform_placeholder()
        self.root.after(50, self._process_events)
        self.root.protocol('WM_DELETE_WINDOW', self._on_close)

    def _on_close(self):
        if self._player: self._player.stop()
        self.root.destroy()

    # ── UI ──

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill=tk.X)

        self.wav_var          = tk.StringVar()
        self.out_var          = tk.StringVar()
        self.contrast_var     = tk.StringVar(value='histogram')
        self.contrast_pct_var = tk.StringVar(value='0.98')
        self.palette_var      = tk.StringVar()
        self.separate_var     = tk.BooleanVar(value=False)
        self.false_color_var  = tk.BooleanVar(value=False)
        self.auto_rotate_var  = tk.BooleanVar(value=True)
        self.force_flip_var   = tk.BooleanVar(value=False)
        self.play_audio_var   = tk.BooleanVar(value=True)
        self.waterfall_var    = tk.BooleanVar(value=True)
        self.speed_var        = tk.StringVar(value='1.0')
        self._cancel_evt      = threading.Event()

        for row, (lbl, var, cmd) in enumerate([
            ('WAV Dosyasi:', self.wav_var,    self._browse_wav),
            ('Cikti:',       self.out_var,    self._browse_out),
            ('Palette:',     self.palette_var, self._browse_palette),
        ]):
            ttk.Label(top, text=lbl).grid(row=row, column=0, sticky='w')
            ttk.Entry(top, textvariable=var, width=80).grid(row=row, column=1, sticky='ew', padx=6)
            ttk.Button(top, text='Sec', command=cmd).grid(row=row, column=2, padx=4)
        top.columnconfigure(1, weight=1)

        opts = ttk.Frame(self.root, padding=(10, 0, 10, 0))
        opts.pack(fill=tk.X)
        for col, (txt, var) in enumerate([
            ('Kanal A/B ayri', self.separate_var),
            ('False color',    self.false_color_var),
            ('Auto rotate',    self.auto_rotate_var),
            ('Force flip',     self.force_flip_var),
            ('Sesi cal',       self.play_audio_var),
            ('Waterfall (gercek zamanli)', self.waterfall_var),
        ]):
            ttk.Checkbutton(opts, text=txt, variable=var).grid(row=0, column=col, sticky='w', padx=3)
        ttk.Label(opts, text='Kontrast:').grid(row=1, column=0, sticky='w', padx=3)
        ttk.Combobox(opts, textvariable=self.contrast_var, width=14,
                     values=['histogram','percent','minmax'],
                     state='readonly').grid(row=1, column=1, sticky='w')
        ttk.Label(opts, text='%:').grid(row=1, column=2, sticky='e')
        ttk.Entry(opts, textvariable=self.contrast_pct_var, width=8).grid(row=1, column=3, sticky='w')
        ttk.Label(opts, text='Waterfall hiz:').grid(row=1, column=4, sticky='e', padx=(10, 2))
        ttk.Combobox(opts, textvariable=self.speed_var, width=8,
                 values=['0.5', '1.0', '1.5', '2.0', '4.0', '0.5x', '1x', '2x', '4x'],
                 state='normal').grid(row=1, column=5, sticky='w')

        ctrl = ttk.Frame(self.root, padding=(10, 8, 10, 6))
        ctrl.pack(fill=tk.X)
        self.start_btn = ttk.Button(ctrl, text='▶  Decode Baslat', command=self._start_decode)
        self.start_btn.pack(side=tk.LEFT)
        self.stop_btn = ttk.Button(ctrl, text='⏹  Durdur', command=self._stop_all)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        self.progress = ttk.Progressbar(ctrl, orient='horizontal', mode='determinate', maximum=100)
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        ttk.Label(ctrl, text=f'🔊 {_backend_name()}', foreground='#555').pack(side=tk.LEFT, padx=4)

        self.status_var = tk.StringVar(value='Hazir')
        self.pos_var    = tk.StringVar(value='Ses konumu: 00:00.00')
        ttk.Label(self.root, textvariable=self.status_var, padding=(10, 0)).pack(anchor='w')
        ttk.Label(self.root, textvariable=self.pos_var,    padding=(10, 0)).pack(anchor='w')

        split = ttk.Frame(self.root, padding=10)
        split.pack(fill=tk.BOTH, expand=True)

        # LEFT: narrow vertical waveform panel
        wf = ttk.LabelFrame(split, text='WAV ↕ zaman')
        wf.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
        self.wave_canvas = tk.Canvas(
            wf, width=self.WAVE_W, height=self.WAVE_H,
            bg='#0d1117', highlightthickness=0)
        self.wave_canvas.pack(padx=4, pady=6)

        # RIGHT: decode preview
        df = ttk.LabelFrame(split, text='Canli Donusum')
        df.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.preview_label = ttk.Label(df)
        self.preview_label.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

    # ── Dialogs ──

    def _browse_wav(self):
        p = filedialog.askopenfilename(filetypes=[('WAV','*.wav'),('All','*.*')])
        if p:
            self.wav_var.set(p)
            if not self.out_var.get().strip():
                self.out_var.set(os.path.splitext(p)[0] + '.png')

    def _browse_out(self):
        p = filedialog.asksaveasfilename(defaultextension='.png',
                                         filetypes=[('PNG','*.png'),('All','*.*')])
        if p: self.out_var.set(p)

    def _browse_palette(self):
        p = filedialog.askopenfilename(filetypes=[('PNG','*.png'),('All','*.*')])
        if p: self.palette_var.set(p)

    # ── Helpers ──

    @staticmethod
    def _fmt(sec: float) -> str:
        sec = max(0.0, float(sec))
        m = int(sec // 60)
        return f'{m:02d}:{sec - m*60:05.2f}'

    @staticmethod
    def _parse_speed_value(raw: str) -> float:
        text = (raw or '').strip().lower()
        if text.endswith('x'):
            text = text[:-1].strip()
        val = float(text)
        if val <= 0:
            raise ValueError('Hiz 0dan buyuk olmali.')
        return val

    @staticmethod
    def _resample_for_speed(audio: np.ndarray, speed: float) -> np.ndarray:
        speed = max(0.05, float(speed))
        if abs(speed - 1.0) < 1e-9:
            return np.asarray(audio, dtype=np.float32)
        src = np.asarray(audio, dtype=np.float32)
        n = len(src)
        if n <= 1:
            return src
        out_n = max(1, int(n / speed))
        x_old = np.arange(n, dtype=np.float64)
        x_new = np.linspace(0, n - 1, out_n, dtype=np.float64)
        return np.interp(x_new, x_old, src).astype(np.float32)

    def _stop_audio(self):
        self._ticker_running = False
        if self._player:
            self._player.stop()
            self._player = None

    def _stop_all(self):
        self._cancel_evt.set()
        self._stop_audio()
        if self.worker and self.worker.is_alive():
            self.status_var.set('Durduruluyor...')
        else:
            self.status_var.set('Durduruldu')

    # ── Vertical waveform ──

    def _set_waveform_placeholder(self, text='WAV bekleniyor'):
        img  = Image.new('RGB', (self.WAVE_W, self.WAVE_H), (13, 17, 23))
        draw = ImageDraw.Draw(img)
        draw.rectangle((0, 0, self.WAVE_W-1, self.WAVE_H-1), outline=(40, 52, 68))
        draw.line((self.WAVE_W//2, 0, self.WAVE_W//2, self.WAVE_H-1), fill=(35, 48, 65))
        self.wave_photo = ImageTk.PhotoImage(img)
        self.wave_canvas.delete('all')
        self.wave_img_id    = self.wave_canvas.create_image(0, 0, anchor='nw', image=self.wave_photo)
        self.wave_cursor_id = self.wave_canvas.create_line(
            0, 0, self.WAVE_W, 0, fill='#ff4d4d', width=2)
        self.wave_duration_sec = 0.0

    def _load_waveform_visual(self, wav_path: str):
        """
        Build vertical waveform:
          Y axis = time  (top = 0s, bottom = duration)
          X axis = amplitude (center = 0, symmetric)
        """
        try:
            rate, raw = scipy.io.wavfile.read(wav_path)
            if raw.ndim > 1: raw = raw[:, 0]
            sig = raw.astype(np.float32)
            if raw.dtype == np.int16:   sig /= 32768.0
            elif raw.dtype == np.int32: sig /= 2147483648.0
            mx = np.max(np.abs(sig))
            if mx > 0: sig /= mx

            self.wave_audio        = sig
            self.wave_rate         = int(rate)
            self.wave_duration_sec = len(sig) / max(1, rate)

            bins = self.WAVE_H
            idx  = np.linspace(0, len(sig), bins+1, dtype=np.int64)
            peaks = np.array([
                float(np.max(np.abs(sig[int(idx[i]):int(idx[i+1])])))
                if int(idx[i+1]) > int(idx[i]) else 0.0
                for i in range(bins)
            ], dtype=np.float32)
            peaks = np.clip(peaks, 0.0, 1.0)

            img  = Image.new('RGB', (self.WAVE_W, self.WAVE_H), (10, 14, 20))
            draw = ImageDraw.Draw(img)
            cx   = self.WAVE_W // 2
            draw.line((cx, 0, cx, self.WAVE_H-1), fill=(35, 48, 65), width=1)

            for y, amp in enumerate(peaks.tolist()):
                half  = int(amp * (self.WAVE_W * 0.46))
                x0    = max(0, cx - half)
                x1    = min(self.WAVE_W - 1, cx + half)
                b     = int(50 + amp * 205)
                draw.line((x0, y, x1, y), fill=(0, int(b*0.5), b))

            draw.rectangle((0, 0, self.WAVE_W-1, self.WAVE_H-1), outline=(38, 52, 70))

            self.wave_photo = ImageTk.PhotoImage(img)
            self.wave_canvas.delete('all')
            self.wave_img_id    = self.wave_canvas.create_image(0, 0, anchor='nw', image=self.wave_photo)
            self.wave_cursor_id = self.wave_canvas.create_line(
                0, 0, self.WAVE_W, 0, fill='#ff5a5a', width=2)

        except Exception as e:
            print(f"[waveform] {e}")
            self.wave_audio = None
            self.wave_rate  = 0
            self._set_waveform_placeholder('Waveform hatasi')

    def _update_wave_cursor(self, wav_sec: float):
        if self.wave_cursor_id is None or self.wave_duration_sec <= 0:
            return
        frac = max(0.0, min(1.0, wav_sec / self.wave_duration_sec))
        y    = int(frac * (self.WAVE_H - 1))
        self.wave_canvas.coords(self.wave_cursor_id, 0, y, self.WAVE_W, y)

    # ── Real-time ticker ─────────────────────────────────────────────────────
    #
    #  KEY DESIGN:
    #  AudioPlayer.elapsed_sec = time.time() - _start_ts
    #  This is purely wall-clock — totally independent of decode speed.
    #  The ticker reads this every 80 ms and moves the cursor accordingly.
    #  The decode worker just builds the image; it does NOT drive the cursor.

    def _start_ticker(self):
        self._ticker_running = True
        self._tick()

    def _tick(self):
        if not self._ticker_running:
            return
        if self._player:
            sec = min(self.wave_duration_sec, self._player.elapsed_sec * self._audio_timeline_scale)
            self._update_wave_cursor(sec)
            backend = self._player.backend_used if self._player.backend_used != 'none' else _backend_name()
            self.pos_var.set(
                f'Ses konumu: {self._fmt(sec)}  /  {self._fmt(self.wave_duration_sec)}'
                + (f'  ▶ [{backend}]' if self._player.is_playing else '  ⏹'))
            if not self._player.is_playing and self._player.backend_used == 'none' and self.play_audio_var.get():
                self.status_var.set('Ses calinamadi: ' + (self._player.last_error or 'backend bulunamadi'))
        self.root.after(80, self._tick)

    # ── Preview ──

    def _render_preview(self, matrix):
        if matrix is None or matrix.size == 0:
            return
        img = Image.fromarray(matrix, 'L')
        w, h = img.size
        scale = min(1100 / w, 560 / h, 1.0)
        if scale < 1.0:
            img = img.resize((int(w*scale), int(h*scale)), Image.Resampling.BILINEAR)
        self.photo = ImageTk.PhotoImage(img)
        self.preview_label.configure(image=self.photo)

    # ── Decode ──

    def _start_decode(self):
        if self.worker and self.worker.is_alive():
            messagebox.showinfo('Bilgi', 'Decode islemi zaten calisiyor.')
            return
        wav = self.wav_var.get().strip()
        if not wav or not os.path.exists(wav):
            messagebox.showerror('Hata', 'Gecerli bir WAV dosyasi secin.')
            return

        outfile  = self.out_var.get().strip() or None
        separate = bool(self.separate_var.get())
        fc       = bool(self.false_color_var.get())
        ar       = bool(self.auto_rotate_var.get())
        ff       = bool(self.force_flip_var.get())
        flip     = True if ff else (False if not ar else None)
        palette  = self.palette_var.get().strip() or None
        contrast = self.contrast_var.get().strip() or 'histogram'
        try:
            contrast_pct = float(self.contrast_pct_var.get().strip())
        except Exception:
            messagebox.showerror('Hata', 'Kontrast yuzdesi sayi olmali.')
            return

        self.progress['value'] = 0
        self.status_var.set('Hazirlaniyor...')
        self.preview_matrix = None
        self.last_render_ts = 0.0
        self._cancel_evt.clear()

        try:
            speed_val = self._parse_speed_value(self.speed_var.get())
        except Exception:
            messagebox.showerror('Hata', 'Waterfall hiz degeri gecersiz. Ornek: 1.0, 1.5, 2 veya 2x')
            return

        # Load waveform image (also fills self.wave_audio, self.wave_rate)
        self._load_waveform_visual(wav)
        self._update_wave_cursor(0.0)

        # ── Start audio INDEPENDENTLY, right now ──────────────────────────
        # Audio plays from the beginning at wall-clock speed.
        # Decode runs in a background thread at CPU speed (usually faster).
        # The ticker keeps the cursor in sync with the audio, not the decode.
        self._ticker_running = False
        if self._player:
            self._player.stop()
        self._audio_timeline_scale = 1.0
        if self.play_audio_var.get() and self.wave_audio is not None:
            self._audio_timeline_scale = speed_val
            audio_for_play = self._resample_for_speed(self.wave_audio, speed_val)
            source_path = wav if abs(speed_val - 1.0) < 1e-9 else None
            self._player = AudioPlayer(audio_for_play, self.wave_rate, source_path=source_path)
            self._player.start()    # <-- audio starts here, wall-clock T=0
            self._start_ticker()    # <-- 80ms ticker reads player.elapsed_sec
            self.status_var.set(f'Hazirlaniyor...  Ses backend: {_backend_name()}  hiz: {speed_val:.2f}x')
        else:
            self._player = None
            if self.play_audio_var.get():
                self.status_var.set('Ses backend bulunamadi; sadece goruntu akacak.')

        self.start_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)

        def worker_fn():
            t0 = time.time()
            def cb(evt): self.event_queue.put(evt)
            try:
                APTDecoder(wav).decode(
                    outfile=outfile, separate_channels=separate,
                    auto_rotate=ar, flip=flip, false_color=fc,
                    palette_file=palette, contrast=contrast,
                    contrast_percent=contrast_pct,
                    progress_cb=cb,
                    show_image=False,
                    realtime_waterfall=bool(self.waterfall_var.get()),
                    realtime_speed=speed_val,
                    cancel_cb=lambda: self._cancel_evt.is_set())
                self.event_queue.put({'event': 'finish', 'ok': True,
                                      'elapsed': time.time() - t0})
            except InterruptedError:
                self.event_queue.put({'event': 'finish', 'ok': False,
                                      'cancelled': True,
                                      'error': 'Decode durduruldu.',
                                      'elapsed': time.time() - t0})
            except Exception as exc:
                self.event_queue.put({'event': 'finish', 'ok': False,
                                      'error': str(exc), 'elapsed': time.time() - t0})

        self.worker = threading.Thread(target=worker_fn, daemon=True)
        self.worker.start()

    # ── Event pump ──

    def _process_events(self):
        try:
            while True:
                evt = self.event_queue.get_nowait()
                et  = evt.get('event')

                if et == 'stage':
                    p = float(evt.get('progress', 0.0))
                    s = evt.get('stage', '')
                    if   s == 'pipeline': overall = 5 + p*45
                    elif s == 'digitize': overall = 52
                    elif s == 'sync':     overall = 55 + p*10
                    elif s == 'build':    overall = 65 + p*30
                    elif s == 'done':     overall = 100
                    else:                 overall = self.progress['value']
                    self.progress['value'] = max(0.0, min(100.0, overall))
                    txt = evt.get('text', '')
                    if txt: self.status_var.set(txt)

                elif et == 'row':
                    idx      = int(evt.get('index', 0))
                    total    = int(evt.get('total', 1))
                    row_data = evt.get('row_data')

                    if self.preview_matrix is None:
                        self.preview_matrix = np.zeros(
                            (total, APTDecoder.IMAGE_WIDTH), dtype=np.uint8)
                    if row_data is not None and idx < self.preview_matrix.shape[0]:
                        self.preview_matrix[idx, :] = np.asarray(row_data, dtype=np.uint8)

                    self.status_var.set(f'Donusturuluyor: satir {idx+1}/{total}')
                    self.progress['value'] = 65 + float(idx+1)/max(1,total)*30

                    now = time.time()
                    if now - self.last_render_ts > 0.08 or idx+1 == total:
                        self._render_preview(self.preview_matrix)
                        self.last_render_ts = now

                elif et == 'result':
                    img = evt.get('image')
                    if img is not None:
                        arr = np.asarray(img)
                        if arr.ndim == 3:
                            pil = Image.fromarray(arr.astype(np.uint8), 'RGB')
                            w, h = pil.size
                            scale = min(1100/w, 560/h, 1.0)
                            if scale < 1.0:
                                pil = pil.resize((int(w*scale), int(h*scale)),
                                                 Image.Resampling.BILINEAR)
                            self.photo = ImageTk.PhotoImage(pil)
                            self.preview_label.configure(image=self.photo)
                        else:
                            self._render_preview(arr.astype(np.uint8))

                elif et == 'finish':
                    self.start_btn.configure(state=tk.NORMAL)
                    self.stop_btn.configure(state=tk.NORMAL)
                    self._stop_audio()
                    if evt.get('cancelled'):
                        self.status_var.set(f"Durduruldu ({evt.get('elapsed',0.0):.2f}s)")
                        continue
                    self.progress['value'] = 100 if evt.get('ok') else self.progress['value']
                    if evt.get('ok'):
                        self.status_var.set(f"Decode bitti ({evt.get('elapsed',0.0):.2f}s)")
                    else:
                        self.status_var.set('Decode hatasi')
                        messagebox.showerror('Hata', evt.get('error', 'Bilinmeyen hata'))

        except queue.Empty:
            pass
        finally:
            self.root.after(50, self._process_events)

    def run(self):
        self.root.mainloop()


# ── Headless ─────────────────────────────────────────────────────────────────

def run_headless(opts):
    print("=" * 60)
    print("NOAA APT DECODER")
    print("=" * 60)
    t0   = time.time()
    flip = True if opts.flip else (False if opts.no_rotate else None)
    APTDecoder(opts.filename).decode(
        outfile=opts.outfile, separate_channels=opts.separate, flip=flip,
        false_color=opts.false_color, palette_file=opts.palette,
        contrast=opts.contrast, contrast_percent=opts.contrast_percent,
        show_image=not opts.no_show)
    print(f"\nOK Tamamlandi: {time.time()-t0:.2f}s")


def parse_args(argv):
    p = argparse.ArgumentParser(description='NOAA APT decoder')
    p.add_argument('filename', nargs='?')
    p.add_argument('outfile',  nargs='?')
    p.add_argument('--separate',         action='store_true')
    p.add_argument('--false-color',      action='store_true', dest='false_color')
    p.add_argument('--palette')
    p.add_argument('--flip',             action='store_true')
    p.add_argument('--no-rotate',        action='store_true')
    p.add_argument('--contrast',         choices=['histogram','percent','minmax'], default='histogram')
    p.add_argument('--contrast-percent', type=float, default=0.98)
    p.add_argument('--gui',              action='store_true')
    p.add_argument('--no-show',          action='store_true')
    return p.parse_args(argv)


if __name__ == '__main__':
    opts = parse_args(sys.argv[1:])
    try:
        if opts.gui or not opts.filename:
            if not TK_AVAILABLE: raise RuntimeError('Tkinter bulunamadi.')
            DecoderGUI().run()
        else:
            run_headless(opts)
    except Exception as exc:
        import traceback
        print(f"\nHATA: {exc}")
        traceback.print_exc()
        sys.exit(1)