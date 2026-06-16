# AI Radiologist (Pneumonia Detection from Chest X‑Ray) – MobileNetV2 + Gradio

## فكرة المشروع
هذا المشروع يبني نموذج **تصنيف ثنائي** لصور أشعة الصدر (Chest X‑Ray) بهدف الكشف عن:

- `NORMAL`
- `PNEUMONIA`

التنفيذ داخل Notebook واحد ويشمل:
- تنزيل وتجهيز البيانات تلقائيًا.
- تدريب نموذج **MobileNetV2 (Transfer Learning)** باستخدام TensorFlow/Keras.
- خلية اختبار (Sanity Check) للتأكد من صحة التوقعات.
- واجهة **Gradio** لرفع صورة وإظهار التشخيص مع احتمالات الفئات.

## الملفات
- `code.ipynb`
  - Notebook المشروع (تنزيل البيانات + تدريب + Gradio UI).

## البيانات (Dataset)
يتم تنزيل البيانات من HuggingFace Datasets عبر `datasets.load_dataset` مع محاولة اختيار أحد المصدرين:
- `keremberke/chest-xray-classification` (مع `name="full"`)
- (fallback) `mmenendezg/pneumonia_x_ray`

ثم يتم استخراج الصور إلى هيكل مجلدات على نمط Keras:
- `dataset/chest_xray/train/NORMAL`
- `dataset/chest_xray/train/PNEUMONIA`
- `dataset/chest_xray/val/...` (إذا توفر)
- `dataset/chest_xray/test/NORMAL`
- `dataset/chest_xray/test/PNEUMONIA`

ملاحظة: في التدريب داخل الـNotebook تم استخدام:
- `TRAIN_DIR = dataset/chest_xray/train`
- `VAL_DIR = dataset/chest_xray/test` (تم استخدام test كمجموعة validation)

## التدريب (Training)
- المعالجة المسبقة: `mobilenet_v2.preprocess_input`
- Augmentation: دوران بسيط + قلب أفقي.
- Backbone: `MobileNetV2(weights='imagenet', include_top=False)` مع تجميد كامل الطبقات (`trainable=False`).
- Head:
  - `GlobalAveragePooling2D`
  - `Dropout(0.2)`
  - `Dense(1, sigmoid)`
- Loss: `binary_crossentropy`
- Optimizer: `adam`
- epochs: `5`

النموذج يُحفظ باسم:
- `pneumonia_model_final.h5`

## اختبار سريع (Sanity Check)
يوجد كود يختار صور عشوائية من:
- `dataset/chest_xray/test`
ويعرض صورتين (NORMAL وPNEUMONIA) مع نتيجة التنبؤ واحتمالها.

## واجهة Gradio
يوجد واجهة Gradio (Blocks) لرفع صورة أشعة وعرض:
- تقرير HTML ملون (Normal/Pneumonia)
- توزيع الاحتمالات (Label)

### ملاحظة مهمة
كود Gradio في الـNotebook يقوم بعمل Resize إلى `150x150`، بينما التدريب كان على `IMG_SIZE=(160,160)`.
للتوافق الأفضل، يفضّل توحيد حجم الإدخال في الواجهة ليطابق حجم التدريب (160x160) أو استخدام نفس pipeline.

### ملاحظة عن خطأ Event Loop
في مخرجات الـNotebook يظهر خطأ:
- `RuntimeError: <asyncio.locks.Event ...> is bound to a different event loop`
وهذا قد يحدث داخل Colab عند تشغيل Gradio.
حلول شائعة:
- إعادة تشغيل Runtime ثم تشغيل الخلايا بالترتيب.
- تحديث/تثبيت إصدار محدد من Gradio.
- تشغيل الواجهة بـ `share=True` أحيانًا يساعد، أو تجربة `debug=False`.

## التشغيل
1) افتح `code.ipynb` على Google Colab.
2) شغّل الخلايا بالترتيب:
   - تنزيل البيانات
   - تدريب النموذج
   - اختبار سريع
   - تشغيل Gradio
