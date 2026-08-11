# 03 - Model Zoo, Architecture Patching & Preprocessing Pipeline

## 1. Overview & Purpose

The benchmark evaluates attribution computational efficiency across a diverse spectrum of deep vision architectures, from lightweight convolutional backbones (e.g., MobileNetV3) to high-capacity residual networks (e.g., ResNet50, RegNetY) and modern Vision Transformers (ViT, Swin). 

Because different architectures impose distinct tensor requirements and backpropagation behaviors—and because certain explainability algorithms (like Captum's DeepLift) encounter graph conflicts with stock Torchvision layers—this module encapsulates model instantiation, compatibility patching, and input transformation pipelines.

---

## 2. Supported Architectures & Model Zoo

The model registry is defined in [`gui/utils/model_loader.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/utils/model_loader.py#L21-L30) using modern Torchvision `Weights` enums:

| Category | Model Identifier | Backbone Architecture | Parameters | Weight Enum | Resolution Behavior |
|:---|:---|:---|:---|:---|:---|
| **CNN** | `resnet50` | Residual Network | 25.6M | `ResNet50_Weights.DEFAULT` | Flexible (32px – 1024px) |
| **CNN** | `convnext-t` | ConvNeXt Tiny | 28.6M | `ConvNeXt_Tiny_Weights.DEFAULT` | Flexible (32px – 1024px) |
| **CNN** | `efficientnet-b0` | EfficientNet B0 | 5.3M | `EfficientNet_B0_Weights.DEFAULT` | Flexible (32px – 1024px) |
| **CNN** | `mobilenet-v3-large`| MobileNet V3 Large | 5.5M | `MobileNet_V3_Large_Weights.IMAGENET1K_V1` | Flexible (32px – 1024px) |
| **CNN** | `densenet121` | DenseNet 121 | 8.0M | `DenseNet121_Weights.IMAGENET1K_V1` | Flexible (32px – 1024px) |
| **Hybrid** | `regnet-y-8gf` | RegNetY 8.0GF | 39.4M | `RegNet_Y_8GF_Weights.IMAGENET1K_V1` | Flexible (32px – 1024px) |
| **Transformer**| `vit-b-16` | Vision Transformer Base (16x16 patch) | 86.6M | `ViT_B_16_Weights.IMAGENET1K_V1` | Fixed (224px only) |
| **Transformer**| `swin-t` | Swin Transformer Tiny | 28.3M | `Swin_T_Weights.IMAGENET1K_V1` | Fixed (224px only) |

---

## 3. DeepLift & Architecture Patching

Stock PyTorch implementations of standard vision models present two major hurdles for gradient- and backpropagation-based XAI methods:

### 3.1 ReLU Instance Reuse in ResNet
In standard Torchvision `Bottleneck` and `BasicBlock` modules, a single `self.relu` instance is invoked multiple times within the same forward pass. When Captum hooks into the module during `DeepLift` or `DeepLiftShap` attribution, this layer reuse causes backward attribution graph tracking errors.

To fix this, [`gui/utils/model_loader.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/utils/model_loader.py#L32-L75) replaces the forward routines with patched implementations (`_patched_bottleneck_forward` and `_patched_basicblock_forward`) and decouples the activation instances:

```python
for module in model.modules():
    if isinstance(module, Bottleneck):
        module.relu1 = torch.nn.ReLU(inplace=False)
        module.relu2 = torch.nn.ReLU(inplace=False)
        module.relu3 = torch.nn.ReLU(inplace=False)
```

### 3.2 In-Place Activation Overrides
Attribution algorithms require pristine intermediate activation tensors to compute reference deltas. `load_model()` recursively traverses all modules in the instantiated network and explicitly enforces `module.inplace = False`.

---

## 4. Input Preprocessing & Image Pipelines

The preprocessing pipeline in [`gui/utils/model_loader.py:preprocess_image()`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/utils/model_loader.py#L123-L151) accommodates both fixed-resolution transformers and variable-resolution CNN benchmarking:

- **Standard ImageNet Normalization**:
  $$\text{Mean} = [0.485, 0.456, 0.406], \quad \text{Std} = [0.229, 0.224, 0.225]$$
- **Dynamic Resolution Scaling**:
  For variable resolution testing, input images are resized maintaining the standard ImageNet aspect ratio ($256/224 \approx 1.14 \times \text{target\_size}$) before center cropping to the exact square dimensions ($N \times N$).
- **Fixed Transformer Constraints**:
  Transformer backbones (`vit-b-16`, `swin-t`) enforce fixed positional embeddings and window partitions. When selected alongside multiple resolutions, the task queuing engine automatically clamps their evaluation to `224px`.

### 4.1 Local ImageNet Label Mapping
Model predictions are decoded using [`gui/utils/label_utils.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/utils/label_utils.py). Rather than relying on external GitHub URLs during benchmark execution, class names are derived directly from the cached `ResNet50_Weights.DEFAULT.meta["categories"]` taxonomy, ensuring completely offline execution.

---

## 5. Key Files & Directory Mapping

- [`gui/utils/model_loader.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/utils/model_loader.py) — Model zoo registry, DeepLift forward patching, and dynamic image transformation pipelines.
- [`gui/utils/label_utils.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/utils/label_utils.py) — Offline ImageNet-1K label decoder and category mapping utility.
- [`gui/utils/loader.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/utils/loader.py) — Workspace image loader, remote URL fetching, and base64 image encoders.

---

## 6. Related Documentation

- [`04-benchmark-configuration.md`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/docs/04-benchmark-configuration.md) — Explains how models and resolutions are staged in the UI.
- [`05-benchmark-execution-engine.md`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/docs/05-benchmark-execution-engine.md) — Details the model caching mechanism that keeps loaded networks in memory across benchmark iterations.
- [`12-in-app-documentation-and-taxonomy.md`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/docs/12-in-app-documentation-and-taxonomy.md) — Reference guide for model backbone parameters and design paradigms.
