# NOAA APT Decoder (Python)

Bu proje, NOAA APT (Automatic Picture Transmission) WAV kayıtlarını görüntüye çeviren Python tabanlı bir decoder'dır.

Ana uygulama [main.py](main.py) dosyasındadır. DSP ve sync yaklaşımı, Rust tabanlı referans proje [noaa-apt/noaa-apt](noaa-apt/noaa-apt/README.md) ile uyumlu olacak şekilde Python'a taşınmıştır.

## Neler Var

- Headless (CLI) ve GUI modu
- Gerçek zamanlı waterfall benzeri satır satır görüntü akışı
- WAV waveform paneli ve anlık konum göstergesi
- Ses oynatma ile konum senkronu
- Waterfall/ses hız çarpanı (manuel giriş: `0.5`, `1`, `1.25`, `2x` vb.)
- Decode sırasında durdurma (cancel)
- Kontrast modları: `histogram`, `percent`, `minmax`
- Kanal ayırma (`_ch_a`, `_ch_b`) ve false-color (`_false_color`)

## Gereksinimler

Önerilen ortam:

- Python 3.8+
- NumPy
- SciPy
- Pillow
- Tkinter (GUI için)

Opsiyonel ses backendleri:

- `sounddevice` (tercihli)
- Windows: `winsound` (Python ile gelir)
- `simpleaudio`

Kurulum:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install numpy scipy pillow sounddevice simpleaudio
```

Notlar:

- `sounddevice` kurulu ama PortAudio yoksa backend düşebilir; uygulama otomatik fallback dener.
- Stereo WAV dosyalarında ilk kanal kullanılır.

## Çalıştırma

### GUI

```powershell
python main.py --gui
```

Dosya verilmezse de GUI açılır:

```powershell
python main.py
```

### Headless (CLI)

```powershell
python main.py input.wav output.png
```

Seçenekler:

- `--separate`
- `--false-color`
- `--palette FILE.png`
- `--flip`
- `--no-rotate`
- `--contrast histogram|percent|minmax`
- `--contrast-percent 0.98`
- `--no-show` (headless'ta görüntü penceresi açma)

## GUI Kullanımı

1. WAV dosyasını seçin.
2. Gerekirse çıktı dosyasını ve seçenekleri ayarlayın.
3. `Waterfall (gercek zamanli)` açıkken satırlar gerçek zamanlı akar.
4. `Waterfall hiz` alanına hız girin (`1`, `1.5`, `2x` gibi).
5. `Decode Baslat` ile başlatın.
6. `Durdur` ile decode ve sesi kesebilirsiniz.

## Hız Kontrolü

Tek bir hız değeri hem waterfall akışına hem de ses oynatımına uygulanır.

Örnekler:

- `0.5` -> yavaş
- `1` -> normal
- `2` veya `2x` -> 2 kat hızlı

Geçersiz girişte GUI hata mesajı verir.

## Üretilen Dosyalar

- Normal çıktı: `output.png`
- Ayrı kanallar: `output_ch_a.png`, `output_ch_b.png`
- False-color: `output_false_color.png`

## Kısa Teknik Özet

Pipeline sırası:

1. WAV okuma ve normalize
2. 20800 Hz çalışma oranına resample
3. Bandpass + DC bastırma
4. 2400 Hz AM demodülasyon
5. Lowpass + 4160 Hz'e decimate
6. Sync tespiti
7. Satır çıkarma ve görüntü matrisi
8. Kontrast / kanal ayrımı / false-color

## Sorun Giderme

### Ses yok

- GUI'de `Sesi cal` açık mı kontrol edin.
- Durum satırındaki backend bilgisini kontrol edin.
- Gerekirse `sounddevice` yerine `winsound/simpleaudio` fallback devreye girer.

### `unsupported platform` veya backend hataları

- Uygulama backend fallback zinciri kullanır.
- Windows'ta ek olarak PowerShell tabanlı fallback bulunur.

### `Yeterli sync bulunamadi`

- Kayıt zayıf/gürültülü olabilir.
- NOAA APT dışı bir sinyal olabilir.
- Farklı kayıtla ve kontrast modlarıyla tekrar deneyin.

## Referans

Algoritmik yaklaşım, Rust tabanlı referans projeden esinlenmiştir: [noaa-apt/noaa-apt](noaa-apt/noaa-apt/README.md)