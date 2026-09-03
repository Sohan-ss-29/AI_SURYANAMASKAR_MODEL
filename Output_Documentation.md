#### **1. Dataset statistics**



**Raw dataset: 2,233 images**

**Clean dataset: 2,188 images**



**Removed:**

**Blurry: 17**

**No pose: 15**

**Exact duplicates: 13**

**Total removed: 45**



**class distribution:**

**Pranamasana             550**

**Hasta Uttanasana        431**

**Padahastasana           286**

**Ashwa Sanchalanasana    161**

**Parvatasana             199**

**Ashtanga Namaskara      440**

**Bhujangasana            121**

**Total                  2188**

#### 

#### **2. Duplicate analysis**



**Potential duplicates analyzed: 316**



**pHash distance:**

**0 → 13 exact duplicates**

**2 → 127 near duplicates**

**4 → 176 near duplicates**

**Only exact duplicates (distance 0) were removed. Near-duplicate images were retained because they can represent meaningful variations in posture, movement, camera position, lighting, etc.**

#### 

#### **3. Dataset splitting**



**Total:       2,188**



**Training:    1,532**

**Validation:    312**

**Testing:      344**

**Custom-video frames were grouped by source video, and each video group was assigned to only one split.**





#### **4. Video leakage verification**



**output:**

**======================================================================**

**DATASET LEAKAGE CHECK**

**======================================================================**



**✅ NO VIDEO GROUP LEAKAGE FOUND**



**Video groups checked: 11**

**======================================================================**





#### **5. MobileNetV2 training log**



**TensorFlow: 2.21.0**



**Training images: 1532**

**Validation images: 312**

**Test images: 344**



**Classes: 7**



**Model: MobileNetV2**

**Pretrained weights: ImageNet**

**Input: 224 × 224 × 3**

**Total parameters: 2,266,951**



**Stage 1 Best Validation Accuracy:**

**89.42%**

**Stage 1 transfer learning achieved the best validation accuracy of 89.42%. Subsequent fine-tuning did not produce an improvement beyond this value.**





#### **6. Final test evaluation**



**Test Dataset: 344 images**



**Test Accuracy:**

**82.56%**

**per-class results:**

**Pranamasana             F1 = 99.57%**

**Hasta Uttanasana        F1 = 82.35%**

**Padahastasana           F1 = 75.61%**

**Ashwa Sanchalanasana    F1 = 67.74%**

**Parvatasana             F1 = 80.60%**

**Ashtanga Namaskara      F1 = 71.84%**

**Bhujangasana            F1 = 53.66%**





#### **7. Project Completion**



**| Component                     | Status  | Completion |**

**| ----------------------------- | ------  | ---------: |**

**| Problem definition / planning | ✅      |       100% |**

**| Dataset collection            | ✅      |       100% |**

**| Custom video collection       | ✅      |       100% |**

**| Frame extraction              | ✅      |       100% |**

**| Dataset cleaning              | ✅      |       100% |**

**| Duplicate analysis            | ✅      |       100% |**

**| MediaPipe validation          | ✅      |       100% |**

**| Clean dataset creation        | ✅      |       100% |**

**| Train/Val/Test splitting      | ✅      |       100% |**

**| Leakage verification          | ✅      |       100% |**

**| MobileNetV2 implementation    | ✅      |       100% |**

**| Transfer learning             | ✅      |       100% |**

**| Fine-tuning experiment        | ✅      |       100% |**

**| Test evaluation               | ✅      |       100% |**

**| Error/confusion analysis      | ⏳      |         0% |**

**| Model improvement/iteration   | ⏳      |         0% |**

**| Real-time webcam inference    | ⏳      |         0% |**

**| MediaPipe posture correctness | ⏳      |         0% |**

**| Sequence/FSM integration      | ⏳      |         0% |**

**| Accuracy/correction feedback  | ⏳      |         0% |**

**| Voice feedback                | ⏳      |         0% |**

**| Final system integration      | ⏳      |         0% |**

**| Final testing                 | ⏳      |         0% |**

**| Documentation/PPT/report      | 🔄      |       \~30% |**





**Stage 1 — Transfer Learning**

&#x20;       **↓**

**15 epochs**

&#x20;       **↓**

**Best validation accuracy = 89.42%**

&#x20;       **↓**

**Stage 2 — Fine-Tuning**

&#x20;       **↓**

**15 epochs**

&#x20;       **↓**

**No improvement beyond 89.42%**

&#x20;       **↓**

**Best model checkpoint retained**

&#x20;       **↓**

**Held-out test evaluation**

&#x20;       **↓**

**82.56% TEST ACCURACY**







