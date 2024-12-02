# Stop Tabela Algılama (YOLOv8)

Bu proje, **YOLOv8** kullanarak stop tabelalarını algılamak amacıyla geliştirilmiştir. YOLOv8, nesne algılama ve görüntü segmentasyonu için güçlü bir araçtır. Aşağıda, projeyi kurma, çalıştırma ve gerekli ortamı hazırlama adımları bulunmaktadır.

---

## İçindekiler
1. [Gereksinimler](#gereksinimler)
2. [Kurulum](#kurulum)
3. [Veri Seti](#veri-seti)
4. [Eğitim Adımları](#eğitim-adımları)
5. [Tahmin Yapma](#tahmin-yapma)
6. [Sonuçları Görüntüleme](#sonuçları-görüntüleme)
7. [Modeli Roboflow’a Yükleme](#modeli-roboflowa-yükleme)  

## **Ön Koşullar**  
- Python 3.10 veya daha güncel bir sürüm
- NVIDIA GPU (isteğe bağlı, model eğitimini hızlandırmak için önerilir)
- Aşağıdaki Python kütüphaneleri kurulu olmalıdır:  
  - `ultralytics`
  - `torch`
  - `roboflow`

---

## **Adımlar**  

### **1. Ortamı Hazırlama**  
GPU'nun aktif olduğundan emin olun. Eğer Google Colab kullanıyorsanız:  
- Menüden **Edit -> Notebook settings** seçeneğine gidin.  
- **Hardware accelerator** ayarını **GPU** olarak değiştirin.  

### **GPU Durumunu Kontrol Etme**  

GPU'nun doğru çalışıp çalışmadığını kontrol etmek için aşağıdaki komutu kullanabilirsiniz:

!nvidia-smi




# Gerekli kütüphaneler
import os
import random
from roboflow import Roboflow
from ultralytics import YOLO
import glob
from IPython.display import Image, display

pip install ultralytics==8.0.196
from ultralytics import YOLO
#ultralytics check fonksiyonunu çalıştırma
ultralytics.checks()

from roboflow import Roboflow
# Roboflow API anahtarınızı girin
rf = Roboflow(api_key="D6O6tzpNXel87uTZPjNc")
# Roboflow projenize erişim sağlamak
project = rf.workspace("stardust").project("stop-sign-ocamr")
version = project.version(1)
dataset = version.download("yolov8")

# Eğitilmiş YOLOv8 modelini yükleme
model = YOLO(f"{HOME}/runs/detect/train/weights/best.pt")

# Test verisinden rastgele bir resim seçme
test_set_loc = dataset.location + "/test/images/"
random_test_image = random.choice(os.listdir(test_set_loc))
print(f"Running inference on {random_test_image}")

# Seçilen resim üzerinde tahmin yapma
pred = model.predict(os.path.join(test_set_loc, random_test_image), conf=0.25, overlap=30).json()

# Tahmin sonuçlarını yazdırma
print(pred)

# Sonuç resimlerini görselleştirme
base_path = '/content/runs/detect/'
subfolders = [os.path.join(base_path, d) for d in os.listdir(base_path)
              if os.path.isdir(os.path.join(base_path, d)) and d.startswith('predict')]

# En son değiştirilmiş klasörü bulma
latest_folder = max(subfolders, key=os.path.getmtime)

# En son klasördeki resimleri al ve ilk 3 tanesini seç
image_paths = glob.glob(f'{latest_folder}/*.jpg')[:3]

# Resimleri görüntüleme
for image_path in image_paths:
    display(Image(filename=image_path, width=600))
    print("\n")
