# RSNA Pneumonia Detection (Multi-Task: Classification + Localization)

## فكرة المشروع
هذا المشروع يطبّق نموذج **Multi-Task** على صور أشعة الصدر (Chest X-ray) من مسابقة/بيانات **RSNA Pneumonia Detection Challenge** بهدف:

- **تصنيف الحالة**: Pneumonia / No Pneumonia على مستوى الصورة.
- **تحديد موقع الإصابة**: كشف صندوق/صناديق (Bounding Boxes) لمناطق الالتهاب الرئوي.

المشروع مكتوب داخل Notebook ويجمع بين:
- تحميل البيانات عبر `kagglehub`
- قراءة صور DICOM عبر `pydicom`
- نموذج PyTorch من الصفر (بدون pretrained) مع رأس تصنيف ورأس كشف Grid-based.

## الملفات
- `rsna-pneumonia-cls-det.ipynb`
  - Notebook (Colab/محلي) للتجهيز + التدريب + التقييم + حفظ checkpoints.

## كيف يعمل الكود (ملخص)

### 1) تحميل البيانات
- يستخدم `kagglehub` لتنزيل Dataset:
  - `parin30/rsna-pneumonia-detection`
- يبحث عن:
  - `stage_2_train_labels.csv`
  - مجلد `stage_2_train_images` الذي يحتوي ملفات DICOM بصيغة `.dcm`

### 2) تجهيز الـLabels
- ملف `stage_2_train_labels.csv` يحتوي:
  - `patientId`
  - `Target` (0/1)
  - إحداثيات الصناديق `x, y, width, height` عندما تكون `Target=1`
- يبني قاموسين:
  - `target_by_pid[patientId]` (تصنيف)
  - `boxes_by_pid[patientId]` (صناديق)

### 3) قراءة ومعالجة صور DICOM
- `read_dicom(path)`:
  - قراءة `pixel_array`
  - قص قيم intensity باستخدام percentile (1%, 99%) ثم normalization إلى `[0..1]`
- `resize_and_boxes(...)`:
  - تغيير الحجم إلى `512x512`
  - تحويل الصناديق إلى `xyxy` بعد scaling

### 4) Dataset / DataLoader
- Dataset يرجع:
  - `img_t`: Tensor شكل `[1, 512, 512]`
  - `cls`: Tensor شكل `[1]` (0/1)
  - `boxes`: Tensor بشكل `[N,4]` بصيغة `xyxy`
- يوجد `collate_fn` لأن عدد الصناديق متغير بين الصور.

### 5) نموذج MultiTaskNet
Backbone CNN بسيط من ConvBlocks مع downsampling حتى grid صغيرة.
- **Classification head**: `AdaptiveAvgPool -> Linear(…,1)` يعطي `cls_logit`
- **Detection heads**:
  - `det_obj`: objectness logit لكل خلية Grid
  - `det_ltrb`: مسافات `L,T,R,B` لكل خلية (بوحدات stride) مع `softplus` لضمان الإيجابية

### 6) Targets للكشف + الخسارة
- `make_det_targets(...)`: يربط كل GT box بخلية grid التي يقع فيها مركز الصندوق.
- `MultiTaskLoss`:
  - `loss_cls`: `BCEWithLogitsLoss` مع `pos_weight` لأن الإيجابيات أقل
  - `loss_obj`: `BCEWithLogitsLoss` مع `pos_weight` ديناميكي بسبب عدم التوازن في grid
  - `loss_box`: `SmoothL1` للـltrb على الخلايا الموجبة فقط

### 7) التقييم
- تصنيف: `ROC-AUC` باستخدام `roc_auc_score`
- كشف: decoding للصناديق + NMS ثم حساب **أفضل IoU** مقابل GT لكل صورة (متوسط best IoU)

### 8) حفظ النماذج والنتائج
يحفظ داخل `SAVE_DIR`:
- `best_by_auc.pth`
- `best_by_iou.pth`
- `last.pth`
- `history.csv`
- صور plots مثل `plot_val_auc.png` وغيرها

## التشغيل
### المتطلبات
- Python
- PyTorch
- `kagglehub`
- `pydicom`, `opencv-python`, `pandas`, `scikit-learn`, `tqdm`

### تشغيل سريع
1) افتح `rsna-pneumonia-cls-det.ipynb` وشغّل الخلايا بالترتيب.
2) تأكد من مسار البيانات داخل خلية الإعداد:
- `DATA_ROOT`
- `IMG_DIR`
- `CSV_PATH`
