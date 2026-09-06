# GUIDE.md — Training Guide (Hinglish)

**Project:** AI-Powered Mobile Road Intelligence & Road Damage Detection System
**Current phase:** Sirf Computer Vision model (YOLO). GPS / IMU / LiDAR / ESP32 / Android abhi NAHI.

Ye guide un logon ke liye hai jo **doosre laptop par model train karenge**.
Data pipeline (inspection → validation → harmonization → duplicate detection → split → preview)
**pehle se complete ho chuka hai**. Aapko sirf **training se aage** ka kaam karna hai.

---

## 0. Ek line mein summary

```
Folder copy karo  →  Python + PyTorch(GPU) install karo  →  1 chhota smoke test  →  Full training  →  Evaluate  →  best.pt wapas bhejo
```

---

## 1. Kya-kya copy karna hai (IMPORTANT)

Poora `road_damage_ai` folder copy karo, **lekin ye do folder chhod sakte ho** (bahut bade hain aur training ke liye zaroori nahi):

| Folder | Size | Copy karein? |
|---|---|---|
| `data/processed/road_damage_combined/images/` | ~1.4 GB | ✅ **HAAN — yahi training data hai** |
| `data/processed/road_damage_combined/labels/` | ~15 MB | ✅ **HAAN — labels** |
| `configs/`, `scripts/`, `utils/`, `requirements.txt` | chhota | ✅ **HAAN** |
| `weights/yolo11n.pt` | ~5 MB | ✅ HAAN (pretrained model — offline bhi chal jaayega) |
| `reports/`, `runs/dataset_preview/` | chhota | ✅ HAAN (reference ke liye) |
| `data/processed/road_damage_combined/staging/` | ~1.4 GB | ❌ Nahi (ye sirf intermediate copy hai) |
| `data/raw/` | ~7.8 GB | ❌ Nahi (raw datasets + RAD videos — training ko nahi chahiye) |

**Total copy size: ~1.5 GB.** Pen drive mein aaram se aa jaayega.

### Minimum working copy ka structure

```
road_damage_ai/
├── configs/
│   ├── data.yaml
│   └── class_mapping.yaml
├── scripts/
├── utils/
├── weights/
│   └── yolo11n.pt
├── data/
│   └── processed/
│       └── road_damage_combined/
│           ├── images/{train,val,test}/
│           └── labels/{train,val,test}/
├── requirements.txt
├── GUIDE.md
└── README.md
```

> ⚠️ **Folder ka naam aur andar ka structure badalna mat.** Scripts `pathlib` se apne aap
> path nikaal lete hain, isliye drive letter (D:, E:, C:) kuch bhi ho — chalega.
> Bas `road_damage_ai` folder ke andar ka structure same rehna chahiye.

### Copy kaise karein (recommended)

Pen drive / external HDD se copy karna sabse aasaan hai.
Windows par CMD/PowerShell se selective copy:

```bash
robocopy "D:\ML_PROJECTS\SIH\road_damage_ai" "E:\road_damage_ai" /E /XD staging raw
```

(`/XD staging raw` ka matlab — `staging` aur `raw` folder skip kar do.)

---

## 2. Training laptop par kya chahiye

| Cheez | Requirement |
|---|---|
| OS | Windows 10/11 ya Linux (dono chalega) |
| Python | 3.10 ya 3.11 (3.12 bhi theek hai) |
| GPU | NVIDIA GPU with CUDA (4 GB VRAM se kaam chal jaata hai, 6–8 GB better) |
| Disk | ~5 GB free |
| Internet | Sirf pehli baar (packages + pretrained weights download ke liye) |

CPU par bhi train ho jaayega **lekin bahut slow** (~20+ ghante). GPU strongly recommended.

---

## 3. Step-by-step setup

### Step 3.1 — Project folder mein jao

```bash
cd E:\road_damage_ai
```

(apna actual path daalna)

### Step 3.2 — Virtual environment banao (recommended)

Windows:

```bash
python -m venv venv
```

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

### Step 3.3 — PyTorch (GPU version) install karo — YE PEHLE

**Ye step sabse important hai.** Agar seedha `pip install -r requirements.txt` chalaoge to
CPU-only PyTorch install ho jaayega aur GPU use nahi hoga.

CUDA 12.x wale GPU ke liye:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

CUDA 11.8 wale purane GPU ke liye:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Kaunsa chahiye pata nahi? Terminal mein `nvidia-smi` chalao — top-right corner mein
"CUDA Version" likha hoga. 12.x ho to pehla, 11.x ho to doosra.

### Step 3.4 — Baaki packages

```bash
pip install -r requirements.txt
```

### Step 3.5 — GPU check karo (skip mat karna)

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
```

Output mein `CUDA: True` aur GPU ka naam dikhna chahiye.
Agar `False` aaye → Step 3.3 dobara karo (pehle `pip uninstall torch torchvision` chala ke).

---

## 4. Training se pehle 2-minute ka verification

### Step 4.1 — Dataset sahi copy hua ya nahi

```bash
python scripts/validate_annotations.py
```

Report yahan banegi: `reports/annotation_validation_report.md`

Report ke aakhir mein `FINAL/train`, `FINAL/val`, `FINAL/test` sections dekho.
**Expected numbers:**

| Split | Images | Annotations |
|---|---|---|
| FINAL/train | 7413 | 17045 |
| FINAL/val | 1482 | 3225 |
| FINAL/test | 988 | 2132 |

`missing_images`, `missing_labels`, `files_with_errors` — sab **0** hone chahiye.

> Agar `FINAL/*` sections dikh hi nahi rahe, matlab `data/processed/road_damage_combined/`
> copy nahi hua. Wapas Step 1 dekho.

### Step 4.2 — Annotation previews dekho (aankhon se check)

`runs/dataset_preview/` folder mein ye images already bani hui hain:

- `combined_train_preview.jpg`
- `combined_val_preview.jpg`
- `combined_test_preview.jpg`

Inhe khol ke dekho — boxes actual potholes/cracks/manholes ke upar honi chahiye.
Ye already verify ho chuka hai, par ek nazar daal lena achha rehta hai.

### Step 4.3 — Smoke test (chhota 1-epoch run)

Poori training se pehle ye chalao — 2-5 minute mein pata chal jaayega ki sab set hai:

```bash
python scripts/train.py --epochs 1 --imgsz 320 --batch 4 --fraction 0.02 --name smoke_test
```

Agar bina error ke chal gaya aur `runs/train/smoke_test/weights/best.pt` ban gaya —
**sab kuch ready hai**. (Iske metrics bekaar honge, normal hai — sirf 1 epoch tha.)

---

## 5. ASLI TRAINING

### `train.py` ab khud crash-safe hai — 4 built-in protections

Colab crashes aur laptop OOM se seekh ke `scripts/train.py` mein ye cheezein
**automatically** hoti hain, kuch bhi extra karna nahi padta:

1. **Auto-resume** — agar `runs/train/<name>/weights/last.pt` pehle se exist karta hai
   (pichli run crash/disconnect hui thi), to **wahi command dobara chalane** se ye khud
   detect karke resume kar lega, epoch 0 se restart nahi karega. Fresh run chahiye ho
   to `--no-auto-resume` lagao.
2. **RAM guard** — background mein har 15 sec RAM check karta hai. 85%+ pe warning,
   93%+ pe critical (gc + CUDA cache clear karta hai). Crash se pehle terminal mein
   pata chal jaata hai kya ho raha hai.
3. **RAM-safe workers** — agar system RAM ≤13 GB hai (Colab jaisa), `--workers`
   automatically 4 pe cap ho jaata hai, chaahe zyada diya ho. 16 GB+ laptop pe koi
   asar nahi.
4. **Offline model fallback** — agar `weights/yolo11n.pt` already local hai, usi ko
   use karega, internet download try nahi karega.
5. **Crash message** — CUDA OOM ya koi bhi crash ho to raw Python traceback ki jagah
   clear message dega: kya hua, `last.pt` safe hai ya nahi, aur exact resume command.

### Main command

```bash
python scripts/train.py --model yolo11n.pt --epochs 100 --batch 16 --imgsz 640 --name road_damage_v1
```

Ye command:
- `yolo11n.pt` pretrained weights download karega (transfer learning — scratch se NAHI)
- GPU khud detect karega
- 100 epochs train karega (patience 20 — improvement ruk gaya to jaldi stop ho jaayega)
- Result `runs/train/road_damage_v1/` mein save karega

### Batch size — apne GPU ke hisaab se

| GPU VRAM | `--batch` | `--imgsz` |
|---|---|---|
| 4 GB (GTX 1650, MX570) | `8` | `640` |
| 6 GB (RTX 2060, 3050) | `16` | `640` |
| 8 GB (RTX 3060 Ti, 4060) | `24` | `640` |
| 12 GB+ (RTX 3060 12GB, 4070, 3090) | `32` | `640` |

**"CUDA out of memory" error aaye to `--batch` aadha kar do.** Simple.

### RAM ki tension mat lo — 16 GB bahut hai

**YOLO poora dataset RAM mein load NAHI karta.** Har batch disk se padha jaata hai,
use hota hai, aur phir chhod diya jaata hai. Dataset 1.4 GB ka hai lekin RAM usage
uspe depend nahi karta.

Is project par actual measure kiya gaya (batch 16, imgsz 640, workers 8):

| Cheez | Value |
|---|---|
| **Peak RAM usage** | **~4.2 GB** |
| System RAM | 16 GB |
| Bacha hua | ~11 GB |

Matlab 16 GB mein aaram se chalega — **8 GB mein bhi chal jaayega**.

#### Colab par crash kyu hota tha (aur yahan kyu nahi hoga)

Colab free tier mein sirf ~12.7 GB RAM milti hai aur crash aksar in wajah se hota hai:

| Colab ki problem | Yahan kya hai |
|---|---|
| `cache=True` / `cache='ram'` lagana — poora dataset RAM mein ghus jaata hai | ❌ Hamare `train.py` mein cache set hi nahi hai (default = disk se padho) |
| 7.8 GB raw zip Colab ke andar extract karna | ❌ Zaroorat nahi — sirf 1.4 GB processed data copy karna hai |
| Colab ka `/content` disk bhar jaana | ❌ Local laptop par 5 GB free chahiye bas |
| Session ka 90-min idle par disconnect hona (ise log "crash" samajh lete hain) | ❌ Local training kabhi disconnect nahi hoti |
| Crash ke baad zero se shuru | ❌ `--resume last.pt` se wahin se continue ho jaata hai |

> ⚠️ **`--cache` flag kabhi mat lagana.** Agar kahin se copy-paste karke
> `cache=True` laga diya, to 9,883 images RAM mein load hone lagengi aur wahi
> Colab wala crash yahan bhi ho jaayega. Default hi sahi hai.

#### Agar phir bhi RAM kam padti dikhe

`--workers` ghata do (har worker apna batch buffer rakhta hai):

```bash
python scripts/train.py --workers 2 --batch 16 --name road_damage_v1
```

Asli constraint **GPU ki VRAM** hai, system RAM nahi. VRAM kam pade to `--batch` ghatao
(upar wali table dekho). "CUDA out of memory" = VRAM, normal "out of memory" = RAM —
dono alag cheezein hain.

### Kitna time lagega (roughly)

| GPU | ~Time (100 epochs) |
|---|---|
| RTX 3060 / 4060 | 1–2 ghante |
| GTX 1650 | 3–5 ghante |
| CPU only | 20+ ghante (avoid karo) |

### Training ke doraan kya dikhega

Har epoch par ye line aayegi:

```
Class   Images  Instances   Box(P)   R   mAP50   mAP50-95
```

**`mAP50` badhna chahiye** epochs ke saath. Bas wahi dekhte raho.

### Agar training beech mein ruk jaaye (bijli, laptop band, crash, etc.)

Ab **wahi original command dobara chalao** — auto-resume khud detect karke continue
karega (upar Section 5 ka point 1 dekho):

```bash
python scripts/train.py --model yolo11n.pt --epochs 100 --batch 16 --imgsz 640 --name road_damage_v1
```

Manually explicit resume bhi kar sakte ho (same effect):

```bash
python scripts/train.py --resume runs/train/road_damage_v1/weights/last.pt
```

### Sab train.py options

| Flag | Default | Kaam |
|---|---|---|
| `--model` | `yolo11n.pt` | Pretrained checkpoint (`yolo11s.pt` bada/behtar par slow) |
| `--epochs` | `100` | Kitne epochs |
| `--batch` | `16` | Batch size |
| `--imgsz` | `640` | Image size |
| `--lr0` | `0.01` | Starting learning rate |
| `--optimizer` | `auto` | SGD / Adam / AdamW / auto |
| `--patience` | `20` | Itne epochs improvement na ho to stop |
| `--workers` | `8` | Data loading threads (**Windows par error aaye to `0` kar do**) |
| `--device` | auto | `0` = GPU, `cpu` = CPU |
| `--seed` | `42` | Reproducibility |
| `--name` | `road_damage_exp` | Run ka naam |
| `--fraction` | `1.0` | Train data ka kitna hissa (smoke test ke liye) |
| `--no-mosaic` | off | Mosaic augmentation band |
| `--resume` | — | Ruke hue run ko continue karo (manual — normally zaroorat nahi, auto-resume khud kar leta hai) |
| `--no-auto-resume` | off | Auto-resume band karo — `last.pt` hote hue bhi fresh run shuru karo |

---

## 5A. Google Colab (T4 GPU) — crash-safe resume plan

T4 mein **16 GB VRAM** hai — 4 GB laptop se 4x zyada, batch 24-32 aaram se chalega. Lekin
Colab free tier **disconnect karta hai** (~90 min idle timeout, ~12 hr max session, GPU
availability din ke hisaab se vary karti hai). Isliye plan disconnect ko **normal maan
ke** banaya gaya hai — resume automatic hona chahiye, manual tracking nahi.

> ⚠️ **RAM alert:** Colab free tier mein sirf **~12 GB system RAM** hoti hai (VRAM alag
> cheez hai — wo GPU ki hai, 16 GB). [Section 5](#ram-ki-tension-mat-lo--16-gb-bahut-hai)
> mein measure kiya gaya 4.2 GB peak **CPU training** ka tha — usme model ka forward/
> backward bhi system RAM use kar raha tha. Colab par GPU training hone se compute
> **VRAM** mein jaata hai, RAM mein nahi — isliye RAM usage waise bhi kam honi chahiye.
> Phir bhi safe margin ke liye: **`--workers 4`** rakho (8 ki jagah — kam workers = kam
> RAM buffering), aur neeche diya RAM-monitor cell background mein chalao taaki crash se
> pehle pata chal jaaye. `--cache` kabhi mat lagana (Section 5 mein wajah likhi hai).

**Golden rule:** `runs/` (checkpoints) **Google Drive** mein jaayen — Colab ki apni disk
(`/content`) session khatam hote hi poori delete ho jaati hai, `last.pt` bhi saath mein.
Dataset training ke liye **local Colab disk** (`/content/data`) mein copy karo — Drive se
seedha training karne par image loading slow hoti hai (network filesystem).

### ✅ Status — ye already ho chuka hai

`road_damage_combined.zip` (processed dataset, ~1.4 GB) Drive par upload ho chuka hai.
Maan lo ye yahan hai: `MyDrive/road_damage_ai/road_damage_combined.zip`
(agar kisi aur folder mein daala hai to neeche har jagah path badal lena).

Ab bacha hua kaam: (1) zip ko unzip karna, (2) chhote code folders (`scripts`, `utils`,
`configs`) upload karna — ye seconds mein ho jaayega, MB level ka hai.

### Step 1 — Drive mount + zip ko unzip karo (Colab cell mein)

```python
from google.colab import drive
drive.mount('/content/drive')

import os
base = '/content/drive/MyDrive/road_damage_ai'
os.makedirs(f'{base}/data/processed', exist_ok=True)

!unzip -q "{base}/road_damage_combined.zip" -d "{base}/data/processed/"
!ls "{base}/data/processed/road_damage_combined"
```

Last line se `images` aur `labels` folder dikhne chahiye — confirm ho jaaye ki unzip
sahi hua.

### Step 2 — Code folders upload (chhote hain — Drive web UI se seedha drag-drop kar sakte ho)

`scripts/`, `utils/`, `configs/`, `requirements.txt` — ye sab tumhare local
`road_damage_ai/` folder se **Drive web UI** (`MyDrive/road_damage_ai/` ke andar)
drag-and-drop karke daal do. Chhota hai (kuch MB), turant ho jaayega.

### Step 3 — Har naye Colab session mein: sab kuch fast local disk pe copy

```python
!cp -r /content/drive/MyDrive/road_damage_ai/data /content/data
!cp -r /content/drive/MyDrive/road_damage_ai/scripts /content/scripts
!cp -r /content/drive/MyDrive/road_damage_ai/utils /content/utils
!cp -r /content/drive/MyDrive/road_damage_ai/configs /content/configs
%cd /content
!pip install ultralytics -q
```

### Step 4 — Pehli training run — output Drive mein likho

`--project` flag se output folder **Drive** mein point karo, taaki disconnect ho bhi jaaye
to `last.pt` safe rahe:

```python
!python scripts/train.py --model yolo11n.pt --epochs 100 --batch 24 --imgsz 640 \
    --workers 4 --project /content/drive/MyDrive/road_damage_ai/runs/train --name road_damage_v1
```

(`--workers 4` — Colab ki 12 GB RAM ke liye safe margin, upar wala RAM alert dekho)

### Step 4.5 — (optional) RAM live monitor, background mein chalao

Training se pehle ek alag cell mein ye chala do — har 30 sec RAM print karega, taaki
crash se pehle warning mil jaaye:

```python
import psutil, time, threading

def monitor_ram():
    while True:
        ram = psutil.virtual_memory()
        print(f"RAM used: {ram.used/1e9:.1f} GB / {ram.total/1e9:.1f} GB ({ram.percent}%)")
        time.sleep(30)

threading.Thread(target=monitor_ram, daemon=True).start()
```

Agar ye consistently 10-11 GB (90%+) dikhaye, `--workers` aur kam karo (`2`) ya
[Section 5](#ram-ki-tension-mat-lo--16-gb-bahut-hai) ki tarah `--batch` kam karo.

### Step 5 — Disconnect ke baad: **Step 4 wala command hi dobara chalao**

`train.py` ab khud checkpoint detect karke resume karta hai ([Section 5](#train-py-ab-khud-crash-safe-hai--4-built-in-protections)
dekho) — alag "auto-resume cell" likhne ki zaroorat nahi. Disconnect hone par:

1. Colab reconnect karo (naya session bhi chalega — data Drive mein safe hai)
2. Step 3 phir se chalao (local copy, 1-2 min — unzip dobara nahi karna, wo Drive mein
   already hai)
3. Step 4 wala **wahi command** phir se chalao — `[auto-resume] Found existing
   checkpoint` print hoga aur khud continue ho jaayega

Explicit resume bhi likh sakte ho agar chaaho (same effect):

```python
!python scripts/train.py --resume /content/drive/MyDrive/road_damage_ai/runs/train/road_damage_v1/weights/last.pt
```

### Batch size resume ke baad change karna ho (VRAM issue par)

`--resume` par CLI se `--batch` override nahi hota (ultralytics `args.yaml` se hi original
values leta hai). Change karna ho to Drive wali `args.yaml` seedhe edit karo:

```python
run_dir = '/content/drive/MyDrive/road_damage_ai/runs/train/road_damage_v1'
!sed -i 's/^batch: .*/batch: 16/' {run_dir}/args.yaml
```

phir Step 5 wala command dobara chalao (auto-resume khud pick kar lega).

### Colab free tier limits (reference)

| Limit | Value |
|---|---|
| Max continuous session | ~12 hours |
| Idle timeout | ~90 min bina activity ke |
| GPU quota | Din ke hisaab se vary, kabhi GPU milta hi nahi |

100 epochs, batch 24, imgsz 640, ~9,900 images — roughly **3-6 hours** T4 par (GPU
availability aur dataset I/O pe depend). Single session mein na ho paaye to upar wala
auto-resume flow use karo — poora training progress kabhi nahi khota, sirf checkpoint
se continue hota hai.

---

## 6. Training ke baad — Evaluation

### Step 6.1 — Test set par evaluate karo

```bash
python scripts/evaluate.py --model runs/train/road_damage_v1/weights/best.pt --split test
```

Ye dega:
- Overall Precision, Recall, mAP50, mAP50-95
- **Per-class metrics** (pothole, crack, manhole, road_damage, speed_bump, unsurfaced_road)
- Confusion matrix + prediction images → `runs/evaluate/eval_results/`
- Report → `reports/evaluation_report.md`

### Step 6.2 — Validation set par bhi (optional comparison)

```bash
python scripts/evaluate.py --model runs/train/road_damage_v1/weights/best.pt --split val --name eval_val
```

### Step 6.3 — Test set ka dataset-wise breakdown

```bash
python scripts/evaluate.py --model runs/train/road_damage_v1/weights/best.pt --split test --cross-dataset
```

---

### Step 6.4 — Cross-dataset generalization test (OPTIONAL, time ho to karna)

Ye check karta hai ki model **naye/anjaan road scenes** par kaam karta hai ya sirf ek
dataset ka style ratt gaya hai. Har experiment ek alag training hai, isliye time lagta hai —
isliye `--epochs 50` kaafi hai.

Pehle configs banao (paths absolute hote hain, isliye naye laptop par ye chalana zaroori hai):

```bash
python scripts/make_cross_dataset_configs.py
```

Phir jo experiment karna ho:

```bash
python scripts/train.py --data configs/cross/exp_b.yaml --epochs 50 --name cross_b
```

```bash
python scripts/evaluate.py --model runs/train/cross_b/weights/best.pt --data configs/cross/exp_b.yaml --split test --name cross_b_eval
```

| Config | Train | Test | Kya check karta hai |
|---|---|---|---|
| `exp_b.yaml` | RAD + Pothole | Road Damage | European roads par transfer hota hai? |
| `exp_c.yaml` | RAD + Road Damage | Pothole | Close-up pothole photos par? |
| `exp_d.yaml` | Road Damage + Pothole | RAD | Indian dashcam scenes par? |

> ⚠️ Ye numbers main model se kaafi kam aayenge — **ye normal hai**.
> Sirf `pothole` class hi teeno datasets mein common hai; `crack`/`manhole` sirf Dataset 2
> mein hain aur `road_damage`/`speed_bump`/`unsurfaced_road` sirf RAD mein.
> Isliye in experiments mein mainly **pothole ka mAP50** dekho.

**Time nahi hai to ye step skip kar sakte ho** — main training (Step 5) hi asli deliverable hai.

---

## 7. Inference test (image aur video)

### Image par

```bash
python scripts/predict.py --source path\to\any_road_image.jpg --model runs/train/road_damage_v1/weights/best.pt
```

### Poore folder par

```bash
python scripts/predict.py --source data/processed/road_damage_combined/images/test --model runs/train/road_damage_v1/weights/best.pt --name test_predictions
```

### Video par (ye demo ke liye sabse important hai)

```bash
python scripts/predict.py --source path\to\road_video.mp4 --model runs/train/road_damage_v1/weights/best.pt --name video_demo
```

Output milega:
- `runs/predict/video_demo/annotated_video.mp4` — boxes + class + confidence + frame number + timestamp
- `runs/predict/video_demo/detections.json` — structured events (future GPS/IMU/LiDAR fusion ke liye ready)

### Webcam se live

```bash
python scripts/predict.py --source 0 --model runs/train/road_damage_v1/weights/best.pt
```

### Confidence threshold badalna

```bash
python scripts/predict.py --source video.mp4 --model runs/train/road_damage_v1/weights/best.pt --conf 0.4
```

(Zyada false positives aa rahe hain → `--conf` badhao, e.g. `0.4`. Detections miss ho rahe hain → ghatao, e.g. `0.15`.)

---

## 8. Training ke baad kaun si files WAPAS leni hain

Training laptop se ye sab copy karke le aao:

```
runs/train/road_damage_v1/
├── weights/
│   ├── best.pt          ← 🔴 SABSE IMPORTANT — final model
│   └── last.pt          ← last epoch (resume ke liye)
├── results.csv          ← har epoch ke metrics
├── results.png          ← training curves
├── confusion_matrix.png
├── args.yaml            ← kis config se train hua
└── val_batch*.jpg       ← sample predictions

runs/evaluate/eval_results/    ← evaluation plots
reports/evaluation_report.md   ← final metrics report
runs/predict/video_demo/       ← video demo output
```

`best.pt` sirf ~5–6 MB ka hoga — WhatsApp/Drive se bhi bhej sakte ho.
**Bas ye ek file bhi aa jaaye to kaam chal jaayega**, baaki sab bonus hai.

---

## 9. Common errors aur fix

| Error | Fix |
|---|---|
| `CUDA out of memory` | `--batch` aadha karo (16 → 8 → 4) |
| `torch.cuda.is_available() == False` | GPU PyTorch install karo (Step 3.3) |
| `Dataset images not found` / galat path | Check karo `data/processed/road_damage_combined/images/train` mein files hain. Folder structure badla mat ho. |
| Windows par `DataLoader worker` crash / hang | `--workers 0` lagao |
| `yolo11n.pt` download nahi ho raha | Internet chalu karo; ya `yolo11n.pt` khud download karke project folder mein rakho aur `--model yolo11n.pt` wahi se uthega |
| `ModuleNotFoundError: ultralytics` | `pip install -r requirements.txt` |
| Training bahut slow | GPU use nahi ho raha — Step 3.5 check karo |
| `UnicodeEncodeError` console par | Ignore karo, ya `set PYTHONIOENCODING=utf-8` |

---

## 10. Agar poora data pipeline dobara chalana ho (normally zaroorat NAHI)

Ye tabhi karna jab aap `data/raw/` bhi copy karke laaye ho aur dataset dobara banana ho.
**Order strictly follow karna:**

```bash
python scripts/inspect_datasets.py
```

```bash
python scripts/validate_annotations.py
```

```bash
python scripts/harmonize_classes.py
```

```bash
python scripts/detect_duplicates.py
```

```bash
python scripts/split_dataset.py
```

```bash
python scripts/visualize_annotations.py
```

Uske baad Step 5 (training).

> ⚠️ `data/raw/` ko **kabhi modify mat karna**. Saara kaam `data/processed/` mein hota hai.

---

## 11. Final model ki details (reference)

| Property | Value |
|---|---|
| Architecture | YOLO11-nano (`yolo11n.pt`), transfer learning |
| Classes | 6 |
| Class list | `0 pothole`, `1 crack`, `2 manhole`, `3 road_damage`, `4 speed_bump`, `5 unsurfaced_road` |
| Train / Val / Test | 7413 / 1482 / 988 images |
| Image size | 640 |
| Model file size | ~5–6 MB |
| Data sources | RAD, Road Damage (Potholes/Cracks/Manholes), Pothole Detection 3900+ |

Class mapping ka poora detail: `configs/class_mapping.yaml`

---

## 12. Iske baad kya (abhi NAHI karna)

Training ho jaane ke baad next phases:

1. `best.pt` → TFLite / NCNN export (Android ke liye)
2. Android app (Phone 1 — camera + YOLO)
3. ESP32 + TF-Luna LiDAR
4. Phone 2 — GPS + network gateway
5. Sensor fusion engine
6. Central server + GIS dashboard

**Abhi sirf ek accha trained `best.pt` chahiye. Bas.**

---

## Quick reference — sirf commands

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

```bash
pip install -r requirements.txt
```

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

```bash
python scripts/validate_annotations.py
```

```bash
python scripts/train.py --epochs 1 --imgsz 320 --batch 4 --fraction 0.02 --name smoke_test
```

```bash
python scripts/train.py --model yolo11n.pt --epochs 100 --batch 16 --imgsz 640 --name road_damage_v1
```

```bash
python scripts/evaluate.py --model runs/train/road_damage_v1/weights/best.pt --split test
```

```bash
python scripts/predict.py --source road_video.mp4 --model runs/train/road_damage_v1/weights/best.pt --name video_demo
```
