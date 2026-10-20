# Qwen3-TTS Tiny TR - Basit Arayüz

[erkamk/qwen3-tts-tiny-tr](https://huggingface.co/erkamk/qwen3-tts-tiny-tr) modelini bilgisayarında çalıştırmak için basit bir web arayüzü. Metni yazarsın, bir referans ses seçersin, model o sesle Türkçe okur. Her şey kendi bilgisayarında çalışır, ekran kartı gerekmez.

## UYARI : Model ve `qwen-tts` paketi kendi sahiplerine aittir. Kullanmadan önce [model sayfasındaki](https://huggingface.co/erkamk/qwen3-tts-tiny-tr) lisansa bak. Başkasının sesini izinsiz klonlama.

## Ne yapar?

- Yazdığın metni seçtiğin referans sesle seslendirir.
- Referans sesi dosyadan yükleyebilir ya da mikrofondan kaydedebilirsin. 5-10 saniyelik temiz bir ses en iyisidir.
- Uzun metni cümlelere bölüp parça parça üretir ve birleştirir. Üretimi istediğin an durdurabilirsin.
- İstersen sayıları yazıya çevirir (`1.250` → `bin iki yüz elli`).
- Ayarlar: sıcaklık, tekrar cezası, cümleler arası duraklama, tohum (seed).
- Ürettiğin sesler `outputs` klasöründe durur. Arayüzde geçmişten tekrar dinleyebilir, indirebilir ya da silebilirsin.

## Ne kullanır?

- Python 3.12 (kurulum sırasında otomatik indirilir, senin Python'una dokunmaz)
- [qwen-tts](https://pypi.org/project/qwen-tts/), PyTorch, soundfile, numpy
- Python'un kendi `http.server` modülü. Gradio ya da Node gerekmez.
- Sade HTML, CSS ve JavaScript

## Kurulum ve çalıştırma

:: DİKKAT :: Mevcut kurulum (`.bat` dosyası ile) sadece windows 10/11 içindir, linux için manuel kurulum yapabilirsiniz. 

1. Kurulum tamamen otomatiktir, hiçbir şey kurmanıza gerek kalmaz. 
2. https://github.com/patientx/Qwen3TTSTinyTR-Basit-Web-Arayuzu/releases den en son sürümü zip olarak bilgisayarınıza indirin.
3. İndirilen zip dosyasını windows - masaüstü vs dışında bir dizine açın. (tercihen kök dizine yani C:\ - D:\ gibi)
2. `run.bat` dosyasını çalıştırın.

İlk çalıştırmada `install.bat` kendiliğinden çalışır: Python'u ve gerekli paketleri `python_env` klasörüne kurar. Model de ilk seferde Hugging Face'ten indirilir. Bu yüzden ilk açılış uzun sürer ve internet ister. Sonraki açılışlarda internet gerekmez.

Açılınca tarayıcıda `http://127.0.0.1:7860/` adresi kendiliğinden açılır. Sayfa hemen gelir, model arka planda yüklenir. Üst sağdaki kutu durumu gösterir. Yaklaşık 15-25 saniye sonra hazır olur.

## run.bat parametreleri

Parametreleri `run.bat` dosyasının yanına yazıp kullanabilirsiniz ; mesela :

```
run.bat --threads 7
```

(yukarıdaki örnekte 8 çekirdekli bir işlemcinin 7 çekirdeği kullanılır, böylece tam yük binmez ama performans artar)

| Parametre | Varsayılan | Ne işe yarar? |
|---|---|---|
| `--model` | `erkamk/qwen3-tts-tiny-tr` | Eğer hazır indirilmiş model varsa ; yerel klasör |
| `--device` | `auto` | `auto`, `cpu` veya `cuda:0`. `auto` ekran kartı varsa onu, yoksa işlemciyi kullanır -bu sürümde ekran kartı için kurulum otomatik değildir- |
| `--dtype` | `float32` | `float32`, `float16` veya `bfloat16` (aşağıdaki nota bak) |
| `--threads` | `0` | İşlemci thread sayısı. `0` PyTorch'un varsayılanını kullanır (aşağıya bak) |
| `--host` | `127.0.0.1` | Sunucu adresi. Aynı ağdaki başka cihazlar da erişsin istersen `0.0.0.0` yaz |
| `--port` | `7860` | Sunucu portu |
| `--no-browser` | kapalı | Tarayıcıyı kendiliğinden açmaz |
| `--no-warmup` | kapalı | Açılıştaki ısınma çalışmasını atlar. Açılış biraz kısalır ama ilk üretim yavaş olur |

**dtype notu:** İşlemcide en iyisi `float32`. `float16` işlemcide genelde yavaştır. `bfloat16` sadece bunu destekleyen yeni işlemcilerde (Intel Sapphire Rapids ve sonrası, AMD Zen 4 ve sonrası) hızlıdır.

## Thread sayısı (`--threads`)

PyTorch her işlemi birkaç işlemci çekirdeğine bölerek yapar. Verilen sayı hıza doğrudan etki eder.

- Fiziksel çekirdek sayını Görev Yöneticisi'nde, Performans sekmesinde, İşlemci altında "Çekirdek" olarak görebilirsin.
- Genelde en iyi sonuç fiziksel çekirdek sayısında ya da **bir eksiğinde** çıkar. Sebebi: tüm çekirdekleri doldurursan tarayıcı, Windows ve sunucunun kendisi için yer kalmaz, sistem yavaşlayabilir.
- Hyper-threading (mantıksal çekirdekler) bu işte genelde fayda sağlamaz.

Kendi bilgisayarında en iyisini bulmak için aynı metni aynı tohumla (seed) birkaç değerle üret ve sonuç kartındaki tamamlanma süresine bak. **Bu süre ne kadar düşükse o kadar hızlı.**

```
run.bat --threads 6
run.bat --threads 7
run.bat --threads 8
```

Örnek olarak 8 çekirdekli Ryzen 5700X3D'de varsayılan ayarla 18 saniye süren bir metin `--threads 8` ile 14 saniyeye indi. `--threads 7` bunun da üstünde bir sonuç verdi. Senin işlemcinde sonuç farklı olabilir.

Hep aynı değeri kullanmak istersen `run.bat` içindeki şu satırı bul:

```
"%PY%" "%~dp0server.py" %*
```

ve şuna çevir:

```
"%PY%" "%~dp0server.py" --threads 7 %*
```

## Klasörler

```
server.py        sunucu ve model kodu
static/          arayüz (index.html, app.js, style.css)
voices/          kaydettiğin referans sesler
outputs/         ürettiğin sesler ve metin bilgileri
python_env/      kurulumla gelen Python (kendiliğinden oluşur)
install.bat      kurulum
run.bat          başlatma
```

## Başlangıçtaki uyarılar

qwen-tts normalde açılışta `flash-attn` ve `sox` ile ilgili uyarılar basar. Bunlar işlemcide çalışırken gerekmez, çünkü sadece modelin kullanmadığı bir bölüm bunları ister. `server.py` başında bu iki paketin yerine boş birer taklit modül konur, böylece uyarılar çıkmaz ve açılış da biraz hızlanır. Paketlerin kendisine dokunulmaz. Aynı yöntemle `onnxruntime` ve `scikit-learn` içe aktarmaları da atlanır.

Bu yüzden bu kurulumu 25 Hz tokenizer kullanan başka bir model denersen hata alabilirsin.

## Sorun giderme

- **Tarayıcıda "sunucuya ulaşılamıyor" yazıyor:** `run.bat` penceresi kapanmış olabilir. Pencereyi açık tut.
- **Kurulum başarısız oldu:** İnternet bağlantını ve yukarıdaki hata mesajlarını kontrol et, sonra `run.bat` dosyasını tekrar çalıştır.
- **İlk açılış çok uzun sürdü:** Normal. Python paketleri ve model indiriliyor. Sonraki açılışlar çok daha kısa.
- **Açılış yavaş:** Windows Defender'da `python_env` klasörünü dışlama listesine eklemek açılışı belirgin şekilde hızlandırabilir.

## Not

Model ve `qwen-tts` paketi kendi sahiplerine aittir. Kullanmadan önce [model sayfasındaki](https://huggingface.co/erkamk/qwen3-tts-tiny-tr) lisansa bak. Başkasının sesini izinsiz klonlama.
