# 10 - Export Systems: ReportLab PDF & CSV Generators

## 1. Overview & Purpose

The benchmark includes an automated export pipeline that converts multidimensional benchmark results into standardized, publication-ready artifacts.

This module details the architecture of [`gui/backend/exporter.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/backend/exporter.py), which compiles both **multi-page landscape PDF reports** via **ReportLab** and clean, flat **CSV datasets**.

---

## 2. ReportLab PDF Export Architecture

The PDF report compiler ([`gui/backend/exporter.py:generate_pdf_report()`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/backend/exporter.py)) constructs an A4 landscape document organized into structured functional pages:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   Multi-Page PDF Report Structure                      │
├────────────────────────────────────────────────────────────────────────┤
│ Page 1: Metadata & Executive Summary                                   │
│   • Header with Batch ID, Date Range & Wall Clock Duration             │
│   • Side-by-Side Environment Summary & Benchmark Configuration Tables  │
│   • Custom Algorithm Parameter Mapping (for parameterized runs)        │
│   • Global Performance & Efficiency Aggregation Table                  │
├────────────────────────────────────────────────────────────────────────┤
│ Page 2: Global Analytical Visualizations                               │
│   • Architecture Efficiency Comparison Bar Charts (Runtime & Memory)   │
│   • Multi-dimensional Pareto Frontier & Speed vs Quality Scatter Plots │
├────────────────────────────────────────────────────────────────────────┤
│ Pages 3+: Granular Per-Image Evaluations                               │
│   • Image Index, Backbone Architecture & Model Prediction Header       │
│   • Input Image & Attribution Heatmap Collage Grid                     │
│   • Consolidated Per-Image Metrics Data Table                          │
└────────────────────────────────────────────────────────────────────────┘
```

### 2.1 Dynamic Two-Pass Page Numbering (`NumberedCanvas`)
To calculate total document page count accurately (rendering `"Page X of Y"` and teal accent line borders across all pages), [`gui/backend/exporter.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/backend/exporter.py#L123-L151) implements a custom ReportLab canvas:

```python
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            super().showPage()
        super().save()
```

### 2.2 Table Formatting & Header Abbreviation
To prevent column clipping on dense metric tables, [`gui/backend/exporter.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/backend/exporter.py#L42-L58) maps verbose column titles into compact acronyms (`PDF_COLUMN_HEADER_MAP`) and applies adaptive font sizing based on column counts.

---

## 3. CSV Dataset Export

For external analysis in R, Python, or spreadsheet tools, [`generate_csv_report()`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/backend/exporter.py) outputs a clean, tabular CSV dataset:

- **Metric Normalization**: Standardizes legacy and new column names (`normalize_metric_columns`).
- **Input Resolution Extraction**: Guarantees a numeric `Input Size (px)` column (`add_input_size_column`).
- **Canonical Ordering**: Orders columns logically from identification metadata to runtimes, energy footprints, memory allocations, and quality metrics (`presentation_df`).

---

## 4. Key Files & Directory Mapping

- [`gui/backend/exporter.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/backend/exporter.py) — Unified ReportLab PDF generator, CSV exporter, and table formatting utilities.
- [`gui/views/active_run.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/views/active_run.py) — Triggers PDF/CSV compilation upon batch completion.
- [`gui/views/history.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/views/history.py) — Provides instant re-download buttons for historical exports.

---

## 5. Related Documentation

- [`08-ui-components-and-visualizations.md`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/docs/08-ui-components-and-visualizations.md) — Explains the Matplotlib chart routines exported to Page 2 of the PDF report.
- [`09-session-management-and-history.md`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/docs/09-session-management-and-history.md) — Details the filesystem directories where PDF and CSV exports are stored.
- [`11-headless-cli.md`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/docs/11-headless-cli.md) — Explains how the CLI runner automatically produces these exact export artifacts.
