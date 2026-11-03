"""
Architecture figure (Fig. 1) of the time + spectral `fusion` model (fusion_model.py) as an
editable one-slide PowerPoint: models/figures/architecture.pptx
(the architecture of the base `fusion` model, which the chosen lr1e-4_wd_ls variant shares).

All shapes are native PowerPoint objects (Arial) so every box, label and colour
can be edited. Numbers follow the trained model exactly:
  preprocessing  02_preprocess_labeled_scans.py / full_scan_preprocessing.py (parameters.py)
  temporal       build_encoder: 5 x [Conv1D-BN-ReLU-MaxPool2], 16/32/64/64/64,
                 kernels 9/9/7/5/5, global avg || max pooling -> 128
  spectral       ALS + Ch2 scaling -> center 0.2 s -> FFT 0-300 Hz (61 bins x 3)
                 -> [Conv16 k5 pool, Conv32 k5 pool, Conv32 k3] -> Flatten ->
                 Dropout 0.3 -> FC 64 (ReLU)
  head           concat 192 -> Dropout 0.3 -> FC 32 (ReLU) -> Dropout 0.3 -> FC 1 (sigmoid)
Requires python-pptx (pip install python-pptx).
Usage:  python fig_architecture.py
"""
import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from lxml import etree

import common as mc

OUT = os.path.join(mc.MODEL_FIG_DIR, "architecture.pptx")
FONT = "Arial"

INK = "1F1F1F"
INK2 = "595959"
GREY_FILL, GREY_LINE = "F2F2F2", "A6A6A6"
T_FILL, T_LINE, T_TITLE = "EAF2FB", "6A9BD8", "2E75B6"           # temporal (blue)
T_BLOCK, T_BLOCK_LINE = "B4CDEB", "4F7FBF"
S_FILL, S_LINE, S_TITLE = "EDF6E8", "7FB86A", "3B8A2E"           # spectral (green)
S_BLOCK, S_BLOCK_LINE = "B9DDB0", "5A9A48"
C_FILL, C_LINE, C_TITLE = "F5EEFB", "8E5BBE", "6A3A9A"           # concat (purple)
H_FILL, H_LINE, H_OUT = "FFF7E6", "E6A700", "D9630F"              # head (amber/orange)
WHITE = "FFFFFF"


def rgb(h):
    return RGBColor.from_string(h)


def _font(run, size, bold=False, italic=False, color=INK, baseline=None):
    f = run.font
    f.name, f.size, f.bold, f.italic = FONT, Pt(size), bold, italic
    f.color.rgb = rgb(color)
    if baseline is not None:                          # e.g. -25000 = subscript
        run._r.get_or_add_rPr().set("baseline", str(baseline))


def write(tf, paras, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, margin=0.04):
    """paras: list of paragraphs; each paragraph = list of runs (text, size, opts)
    or a dict {"runs": [...], "bullet": True, "space_after": pt}."""
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for side in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, side, Inches(margin))
    for i, p in enumerate(paras):
        spec = p if isinstance(p, dict) else {"runs": p}
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = spec.get("align", align)
        if spec.get("space_after"):
            para.space_after = Pt(spec["space_after"])
        if spec.get("bullet"):
            pPr = para._p.get_or_add_pPr()
            pPr.set("marL", str(Inches(0.16)))
            pPr.set("indent", str(-Inches(0.13)))
            bf = etree.SubElement(pPr, qn("a:buFont"))
            bf.set("typeface", FONT)
            bu = etree.SubElement(pPr, qn("a:buChar"))
            bu.set("char", "•")
        for text, size, *opt in spec["runs"]:
            r = para.add_run()
            r.text = text
            _font(r, size, **(opt[0] if opt else {}))


def box(sl, x, y, w, h, fill=WHITE, line=INK2, lw=1.0, shape=MSO_SHAPE.ROUNDED_RECTANGLE,
        radius=0.08, paras=None, name=None, **kw):
    s = sl.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if fill is None:
        s.fill.background()
    else:
        s.fill.solid()
        s.fill.fore_color.rgb = rgb(fill)
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = rgb(line)
        s.line.width = Pt(lw)
    s.shadow.inherit = False
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = radius
    if paras:
        write(s.text_frame, paras, **kw)
    if name:
        s.name = name
    return s


def label(sl, x, y, w, h, paras, rotation=0, **kw):
    t = sl.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    write(t.text_frame, paras, **kw)
    t.rotation = rotation
    return t


def arrow(sl, x1, y1, x2, y2, color=INK2, lw=1.5, head=True):
    c = sl.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = rgb(color)
    c.line.width = Pt(lw)
    if head:
        ln = c.line._get_or_add_ln()
        tail = etree.SubElement(ln, qn("a:tailEnd"))
        tail.set("type", "triangle")
        tail.set("w", "med")
        tail.set("len", "med")
    return c


def conv_stack(sl, cx, cy, w, h, n, fill, line, caption):
    """n offset layers (back to front); front layer centred at (cx, cy)."""
    dx, dy = 0.055, 0.055
    for k in range(n - 1, 0, -1):
        box(sl, cx - w / 2 + k * dx, cy - h / 2 - k * dy, w, h, fill=fill, line=line, lw=0.75,
            shape=MSO_SHAPE.RECTANGLE)
    front = box(sl, cx - w / 2, cy - h / 2, w, h, fill=fill, line=line, lw=0.75, shape=MSO_SHAPE.RECTANGLE,
                paras=[[("Conv1D", 9, {"color": INK2})]], margin=0)
    front.text_frame._txBody.bodyPr.set("vert", "vert270")     # vertical text, no rotated text box
    label(sl, cx - 0.33 + (n - 1) * dx / 2, cy + h / 2 + 0.03, 0.66, 0.22, [[(caption, 9.5, {"color": INK})]],
          margin=0)


def embedding(sl, x, y, h, colors, dim, sym, sub, color_brace):
    """Column of neurons with a brace, dimension and symbol."""
    w = 0.3
    box(sl, x, y, w, h, fill=WHITE, line=INK2, lw=0.75, radius=0.25)
    d = 0.19
    n_top = len(colors) - 1
    ys = [y + 0.08 + i * 0.25 for i in range(n_top)]
    for yy, col in zip(ys, colors[:-1]):
        box(sl, x + (w - d) / 2, yy, d, d, fill=col, line=INK2, lw=0.5, shape=MSO_SHAPE.OVAL)
    dots_top = ys[-1] + d + 0.06
    dots_bot = y + h - 0.08 - d - 0.06
    for k in range(3):
        yy = dots_top + (dots_bot - dots_top) * (k + 0.5) / 3 - 0.02
        box(sl, x + w / 2 - 0.02, yy, 0.04, 0.04, fill=INK2, line=None, shape=MSO_SHAPE.OVAL)
    box(sl, x + (w - d) / 2, y + h - 0.08 - d, d, d, fill=colors[-1], line=INK2, lw=0.5, shape=MSO_SHAPE.OVAL)
    br = sl.shapes.add_shape(MSO_SHAPE.RIGHT_BRACE, Inches(x + w + 0.04), Inches(y), Inches(0.11), Inches(h))
    br.fill.background()
    br.line.color.rgb = rgb(color_brace)
    br.line.width = Pt(1.25)
    br.shadow.inherit = False
    label(sl, x + w + 0.15, y + h / 2 - 0.3, 0.42, 0.26, [[(str(dim), 11, {"bold": True, "color": color_brace})]],
          margin=0, align=PP_ALIGN.LEFT)
    if sym:
        label(sl, x + w + 0.15, y + h / 2 + 0.0, 0.42, 0.3,
              [[(sym, 13, {"italic": True, "color": color_brace}),
                (sub, 13, {"italic": True, "color": color_brace, "baseline": -25000})]],
              margin=0, align=PP_ALIGN.LEFT)


def main():
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(6.0)
    sl = prs.slides.add_slide(prs.slide_layouts[6])     # blank

    # ---- raw signal + preprocessing -----------------------------------------
    label(sl, 0.05, 2.35, 0.85, 0.6, [[("Raw", 12)], [("t-DiFC", 12)], [("signal", 12)]], margin=0)
    arrow(sl, 0.1, 3.0, 0.95, 3.0)
    box(sl, 0.95, 1.3, 2.25, 3.4, fill=GREY_FILL, line=GREY_LINE, radius=0.05, anchor=MSO_ANCHOR.TOP, margin=0.08,
        paras=[{"runs": [("Pre-processing", 13, {"bold": True})], "space_after": 1},
               {"runs": [("per file and laser-ON segment", 10.5, {"italic": True, "color": INK2})], "space_after": 6},
               {"runs": [("Laser ON/OFF segmentation (Otsu)", 11)], "bullet": True, "align": PP_ALIGN.LEFT, "space_after": 3},
               {"runs": [("Rolling-median background removal (2 s)", 11)], "bullet": True, "align": PP_ALIGN.LEFT, "space_after": 3},
               {"runs": [("Peak-candidate detection (> 4.5σ)", 11)], "bullet": True, "align": PP_ALIGN.LEFT, "space_after": 3},
               {"runs": [("Peak merging (< 100 ms)", 11)], "bullet": True, "align": PP_ALIGN.LEFT, "space_after": 3},
               {"runs": [("Window extraction (±0.5 s)", 11)], "bullet": True, "align": PP_ALIGN.LEFT, "space_after": 3},
               {"runs": [("Joint Ch1/Ch2 standardization", 11)], "bullet": True, "align": PP_ALIGN.LEFT}],
        name="preprocessing")
    arrow(sl, 3.2, 3.0, 3.35, 3.0, head=False)
    arrow(sl, 3.35, 1.67, 3.35, 4.52, head=False)
    arrow(sl, 3.35, 1.67, 4.1, 1.67)
    arrow(sl, 3.35, 4.52, 4.1, 4.52)
    for yy in (1.37, 4.22):
        label(sl, 3.33, yy, 0.6, 0.26, [[("2×2001", 10.5, {"color": INK})]], margin=0)

    # ---- temporal branch ----------------------------------------------------
    box(sl, 3.95, 0.3, 6.5, 2.45, fill=T_FILL, line=T_LINE, lw=1.25, radius=0.04, name="temporal_branch")
    label(sl, 3.95, 0.32, 6.5, 0.32, [[("Temporal branch", 14, {"bold": True, "color": T_TITLE})]], margin=0)
    box(sl, 4.1, 0.7, 4.05, 1.95, fill=WHITE, line=T_LINE, lw=1.0, radius=0.06, anchor=MSO_ANCHOR.TOP, margin=0.05,
        paras=[[("CNN feature extractor", 12, {"bold": True})],
               [("5 × [Conv1D → BN → ReLU → MaxPool]", 10, {"color": INK2})]], name="temporal_cnn")
    specs_t = [(16, 9, 1.0), (32, 9, 0.88), (64, 7, 0.76), (64, 5, 0.66), (64, 5, 0.56)]
    for i, (f, k, h) in enumerate(specs_t):
        n = {16: 2, 32: 3, 64: 4}[f]
        conv_stack(sl, 4.38 + i * 0.77, 1.82, 0.3, h, n, T_BLOCK, T_BLOCK_LINE, f"{f}, k{k}")
    arrow(sl, 8.15, 1.67, 8.3, 1.67)
    box(sl, 8.3, 1.12, 0.85, 1.1, fill=WHITE, line=T_LINE, lw=1.0,
        paras=[[("Global", 10.5)], [("avg ‖ max", 10.5)], [("pooling", 10.5)]], name="temporal_pool")
    arrow(sl, 9.15, 1.67, 9.3, 1.67)
    embedding(sl, 9.3, 1.05, 1.25, ["A9C4E8"] * 4, 128, "z", "t", T_TITLE)

    # ---- spectral branch ----------------------------------------------------
    box(sl, 3.95, 3.05, 6.5, 2.8, fill=S_FILL, line=S_LINE, lw=1.25, radius=0.04, name="spectral_branch")
    label(sl, 3.95, 3.07, 6.5, 0.32, [[("Spectral branch", 14, {"bold": True, "color": S_TITLE})]], margin=0)
    box(sl, 4.1, 3.65, 1.15, 1.75, fill=WHITE, line=S_LINE, lw=1.0,
        paras=[[("ALS", 11.5, {"bold": True})], [("background", 11.5, {"bold": True})],
               [("removal", 11.5, {"bold": True})], [("+", 11.5)],
               [("Ch2 scaling", 11.5, {"bold": True})]], name="als")
    arrow(sl, 5.25, 4.52, 5.4, 4.52)
    box(sl, 5.4, 3.65, 1.3, 1.75, fill=WHITE, line=S_LINE, lw=1.0,
        paras=[[("Center 0.2 s", 11.5, {"bold": True})], [("FFT, 0–300 Hz", 11.5, {"bold": True})],
               {"runs": [("61 bins × 3:", 10, {"color": INK2})], "space_after": 0},
               [("log|Ch1|, log|Ch2|,", 10, {"color": INK2})], [("cos Δφ", 10, {"color": INK2})]], name="fft")
    arrow(sl, 6.7, 4.52, 6.85, 4.52)
    box(sl, 6.85, 3.5, 1.3, 2.05, fill=WHITE, line=S_LINE, lw=1.0, radius=0.06, anchor=MSO_ANCHOR.TOP, margin=0.05,
        paras=[[("CNN feature", 12, {"bold": True})], [("extractor", 12, {"bold": True})],
               [("3 × [Conv1D → BN → ReLU]", 8.5, {"color": INK2})]], name="spectral_cnn")
    specs_s = [(16, 5, 0.86), (32, 5, 0.74), (32, 3, 0.74)]
    for i, (f, k, h) in enumerate(specs_s):
        conv_stack(sl, 7.05 + i * 0.4, 4.74, 0.22, h, {16: 2, 32: 3}[f], S_BLOCK, S_BLOCK_LINE, f"{f}, k{k}")
    arrow(sl, 8.15, 4.52, 8.3, 4.52)
    box(sl, 8.3, 3.85, 0.85, 1.35, fill=WHITE, line=S_LINE, lw=1.0,
        paras=[[("Flatten", 10.5)], [("Dropout", 10.5)], [("FC 64", 10.5)], [("(ReLU)", 10.5)]],
        name="spectral_fc")
    arrow(sl, 9.15, 4.52, 9.3, 4.52)
    embedding(sl, 9.3, 3.9, 1.25, ["A8D5A2"] * 4, 64, "z", "s", S_TITLE)

    # ---- concatenation --------------------------------------------------------
    arrow(sl, 10.1, 1.67, 10.6, 1.67)
    arrow(sl, 10.1, 4.52, 10.6, 4.52)
    box(sl, 10.6, 1.3, 1.0, 3.6, fill=C_FILL, line=C_LINE, lw=1.25, radius=0.06, anchor=MSO_ANCHOR.TOP, margin=0.04,
        paras=[[("Concatenate", 10, {"bold": True, "color": C_TITLE})]], name="concat")
    embedding(sl, 10.78, 1.78, 2.75, ["A9C4E8"] * 3 + ["A8D5A2"] * 3 + ["A8D5A2"], 192, "", "", C_TITLE)

    # ---- decision head --------------------------------------------------------
    arrow(sl, 11.6, 3.1, 11.75, 3.1)
    box(sl, 11.75, 1.3, 1.5, 3.6, fill=H_FILL, line=H_LINE, lw=1.25, radius=0.06, anchor=MSO_ANCHOR.TOP, margin=0.04,
        paras=[[("Decision head", 12, {"bold": True})]], name="head")
    layers_h = [("Dropout (0.3)", False), ("FC 32 (ReLU)", True), ("Dropout (0.3)", False), ("FC 1 (Sigmoid)", True)]
    y0 = 1.72
    for i, (txt, bold) in enumerate(layers_h):
        yy = y0 + i * 0.6
        box(sl, 11.85, yy, 1.3, 0.42, fill=WHITE, line=H_LINE, lw=1.0, radius=0.12,
            paras=[[(txt, 10.5, {"bold": bold})]])
        if i < len(layers_h) - 1:
            arrow(sl, 12.5, yy + 0.42, 12.5, yy + 0.6, lw=1.25)
    arrow(sl, 12.5, y0 + 3 * 0.6 + 0.42, 12.5, y0 + 3 * 0.6 + 0.62, lw=1.25)
    label(sl, 11.75, 4.18, 1.5, 0.62, [[("P(peak)", 13, {"bold": True, "color": H_OUT})],
                                        [("probability of a peak", 9.5, {"color": H_OUT})]], margin=0)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    prs.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
