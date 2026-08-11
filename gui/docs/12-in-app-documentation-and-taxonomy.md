# 12 - In-App Documentation, Reference Guides & Taxonomy

## 1. Overview & Purpose

To support researchers without requiring them to consult external literature during experimentation, the dashboard includes a built-in technical reference guide rendered in [`gui/views/documentation.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/views/documentation.py). 

The reference guide is backed by a structured taxonomy defined in [`gui/assets/docs_reference.json`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/assets/docs_reference.json), presenting algorithmic mathematical formulations, theoretical computational complexity profiles, backbone architecture innovations, and metric evaluation logic.

---

## 2. In-App Reference Structure

The documentation view organizes technical knowledge into three expandable sections:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   In-App Documentation Taxonomy                        │
├────────────────────────────────────────────────────────────────────────┤
│ 🔬 Section 1: Explainable AI (XAI) Methods                             │
│   • Gradient-based: Saliency, Integrated Gradients, Guided Backprop... │
│   • Reference-based: DeepLift, DeepLift SHAP, Gradient SHAP...         │
│   • Perturbation & Region-based: Occlusion, LIME...                    │
│   • Activation-based: Grad-CAM...                                      │
├────────────────────────────────────────────────────────────────────────┤
│ 🏗️ Section 2: Vision Model Architectures                              │
│   • Convolutional Networks: ResNet50, ConvNeXt, EfficientNet...        │
│   • Vision Transformers: ViT-B/16, Swin Transformer...                 │
│   • Hybrid & Scaled Backbones: RegNetY-8GF, DenseNet121...             │
├────────────────────────────────────────────────────────────────────────┤
│ 📊 Section 3: Benchmark Metrics & Evaluation Strategies                │
│   • Computational: Attribution Runtime, Peak Memory, Energy (TDP)      │
│   • Quality: Gini Index, Deletion AUC, Insertion AUC, Sensitivity...   │
│   • Task Ordering: Balanced, Sequential, Randomized                    │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Taxonomy Specification (`docs_reference.json`)

The taxonomy database in [`gui/assets/docs_reference.json`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/assets/docs_reference.json) adheres to a structured JSON schema:

```json
{
  "xai_methods": [
    {
      "category": "Gradient-Based Attribution",
      "methods": [
        {
          "name": "Integrated Gradients",
          "formulation": "IG_i(x) = (x_i - x'_i) * integral_0^1 (df(x' + alpha*(x - x')) / dx_i) d_alpha",
          "complexity": "O(m * F)",
          "characteristics": "Axiomatically justified (completeness, implementation invariance); uses Riemann summation path."
        }
      ]
    }
  ],
  "models": [
    {
      "category": "Convolutional Neural Networks",
      "models": [
        {
          "name": "resnet50",
          "params": "25.6M",
          "characteristics": "Residual skip connections mitigating vanishing gradients; standard baseline."
        }
      ]
    }
  ],
  "metrics": [
    {
      "category": "Quality & Faithfulness Suite",
      "metrics": [
        {
          "name": "Deletion AUC",
          "unit": "Score (0 to 1, Lower is better)",
          "definition": "Trapezoidal AUC of prediction decay when masking top salient pixels."
        }
      ]
    }
  ]
}
```

---

## 4. Extending the Taxonomy

When contributing new models, attribution algorithms, or quality metrics to the benchmark:
1. **Add the Method/Model/Metric specification** to [`gui/assets/docs_reference.json`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/assets/docs_reference.json).
2. The UI in [`gui/views/documentation.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/views/documentation.py) will automatically render the new entry into its corresponding reference table on launch without requiring layout changes.
3. Update the global options lists in [`gui/config.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/config.py).

---

## 5. Key Files & Directory Mapping

- [`gui/views/documentation.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/views/documentation.py) — In-app documentation view and markdown table compiler.
- [`gui/assets/docs_reference.json`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/assets/docs_reference.json) — Central JSON database for technical formulations, backbone specifications, and metric definitions.
- [`gui/utils/helpers.py:load_docs_reference()`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/utils/helpers.py) — JSON loader and validator.

---

## 6. Related Documentation

- [`01-architecture-and-lifecycle.md`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/docs/01-architecture-and-lifecycle.md) — Explains the application navigation routing to the documentation tab.
- [`03-model-zoo-and-preprocessing.md`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/docs/03-model-zoo-and-preprocessing.md) — Explains the vision models cataloged in the taxonomy.
- [`06-explanation-quality-suite.md`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/docs/06-explanation-quality-suite.md) — Details the mathematical definitions of the quality metrics described in the reference guide.
