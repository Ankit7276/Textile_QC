"""Polished Streamlit interface for the fabric analyzer."""
import json
import os
import tempfile

import pandas as pd
import streamlit as st

from fabric_analyzer import _np, analyze

st.set_page_config(page_title="LoomLab | Fabric quality", page_icon="🧵", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:var(--background-color); color:var(--text-color); }
.block-container { padding-top:1.5rem; max-width:1400px; }
[data-testid="stSidebar"] { background:var(--secondary-background-color); border-right:1px solid color-mix(in srgb, var(--text-color) 12%, transparent); }
.hero { background:linear-gradient(120deg,#183d30,#276b4e); color:#fff; padding:1.45rem 1.7rem; border-radius:18px; margin-bottom:1.2rem; }
.hero h1 { font-family:Manrope,sans-serif; font-size:1.8rem; margin:0; color:#fff; }
.hero p { color:#e0eee6; margin:.35rem 0 0; font-size:.95rem; }
.eyebrow { text-transform:uppercase; letter-spacing:.13em; font-size:.7rem; font-weight:700; opacity:.82; margin-bottom:.45rem; }
div[data-testid="stMetric"] { background:var(--secondary-background-color); color:var(--text-color); border:1px solid color-mix(in srgb, var(--text-color) 13%, transparent); padding:1rem 1.05rem; border-radius:14px; box-shadow:0 2px 8px color-mix(in srgb, var(--text-color) 7%, transparent); }
div[data-testid="stMetricLabel"], div[data-testid="stMetricLabel"] p { color:var(--text-color); opacity:.78; }
div[data-testid="stMetricValue"] { color:var(--text-color); }
.stTabs [data-baseweb="tab-list"] { gap:1.15rem; border-bottom:1px solid color-mix(in srgb, var(--text-color) 14%, transparent); }
.stTabs [data-baseweb="tab"] { padding:.75rem .15rem; color:var(--text-color); }
div[data-testid="stExpander"] { background:var(--secondary-background-color); border:1px solid color-mix(in srgb, var(--text-color) 13%, transparent); border-radius:12px; }
.upload-card { background:var(--secondary-background-color); border:1px dashed color-mix(in srgb, var(--text-color) 30%, transparent); border-radius:16px; padding:2rem; text-align:center; color:var(--text-color); }
.upload-card h3 { color:var(--text-color); margin:.5rem 0; }
.small-note { color:var(--text-color); opacity:.72; font-size:.84rem; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner="Analyzing fabric image…")
def run_analysis(data, suffix, ppi, width_in, ref_rgb, sensitivity):
    with tempfile.TemporaryDirectory() as directory:
        source = os.path.join(directory, "fabric" + suffix)
        overlay_path = os.path.join(directory, "overlay.png")
        with open(source, "wb") as image_file:
            image_file.write(data)
        result = analyze(source, ppi, width_in, ref_rgb, overlay_path, sensitivity)
        with open(overlay_path, "rb") as overlay_file:
            overlay = overlay_file.read()
        return result, overlay


def metric_value(value, decimals=1):
    try:
        return f"{float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return "—"

with st.sidebar:
    st.markdown("## 🧵 LoomLab")
    st.caption("Fabric quality workspace")
    st.divider()
    st.markdown("#### 1 · Fabric image")
    upload = st.file_uploader("Upload a close-up photo", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"], help="For best results, use a sharp, evenly lit image captured straight on.")
    st.markdown("#### 2 · Measurement scale")
    scale_mode = st.radio("How do you know the scale?", ["Photo width", "Pixels per inch"], label_visibility="collapsed", horizontal=True)
    if scale_mode == "Photo width":
        width_in = st.number_input("Photo width (inches)", min_value=0.1, max_value=100.0, value=1.0, step=0.1)
        ppi = None
    else:
        ppi = st.number_input("Pixels per inch (PPI)", min_value=10.0, max_value=5000.0, value=600.0, step=10.0)
        width_in = None
    st.markdown("#### 3 · Detection settings")
    sensitivity = st.slider("Defect sensitivity", min_value=2.0, max_value=10.0, value=5.0, step=0.5, help="Lower values flag more subtle variations and may increase false positives.")
    compare_color = st.checkbox("Compare against target color")
    reference_rgb = None
    if compare_color:
        color_hex = st.color_picker("Target color", "#968264").lstrip("#")
        reference_rgb = tuple(int(color_hex[i:i + 2], 16) for i in (0, 2, 4))
    st.divider()
    st.caption("Analysis runs in this session. Photos are not saved by the app.")

st.markdown('<div class="hero"><div class="eyebrow">Fabric inspection · workspace</div><h1>Fabric quality analysis</h1><p>Measure weave, spot defects, and review fabric color and texture.</p></div>', unsafe_allow_html=True)

if not upload:
    st.markdown('<div class="upload-card"><div style="font-size:2rem">🪡</div><h3>Start with a fabric photo</h3><div>Upload a close-up image from the left panel to see your quality dashboard.</div></div>', unsafe_allow_html=True)
    st.markdown("#### What you can inspect")
    info_cols = st.columns(4)
    for col, icon, title, detail in zip(info_cols, ["〰️", "🔎", "🎨", "▦"], ["Thread count", "Surface defects", "Color profile", "Texture & weave"], ["FFT and peak-based TPI estimates", "Potential holes, stains, and missing yarn", "RGB, HSV, and reference color difference", "Regularity and texture descriptors"]):
        with col:
            st.markdown(f"**{icon} {title}**<br><span class='small-note'>{detail}</span>", unsafe_allow_html=True)
    st.stop()

try:
    result, overlay = run_analysis(upload.getvalue(), os.path.splitext(upload.name)[1] or ".png", ppi, width_in, reference_rgb, sensitivity)
except Exception as error:
    st.error(f"Could not analyze this image: {error}")
    st.info("Try a different image, or check that the scale value is correct.")
    st.stop()

thread = result["thread_count"]
defects = result["defects"]
regularity = result["regularity"]
color = result["color"]
vertical_tpi = thread["vertical_threads"]["tpi_fft"]
horizontal_tpi = thread["horizontal_threads"]["tpi_fft"]

st.markdown(f"**{upload.name}** &nbsp; · &nbsp; {upload.size / (1024 * 1024):.2f} MB &nbsp; · &nbsp; {result['scale_ppi']:.0f} PPI", unsafe_allow_html=True)
summary = st.columns(5)
summary[0].metric("Thread count", f"{thread['thread_count_fft']:.0f}", "threads per inch · FFT")
summary[1].metric("Vertical / horizontal", f"{vertical_tpi:.0f} / {horizontal_tpi:.0f}", "TPI")
summary[2].metric("Potential defects", defects["total"], "review highlighted areas")
summary[3].metric("Weave regularity", f"{regularity['regularity_score']:.2f}", "Lower may indicate irregularity")
if "delta_e_2000_vs_reference" in color:
    summary[4].metric("Color difference", f"ΔE {color['delta_e_2000_vs_reference']:.1f}", "vs target color")
else:
    summary[4].metric("Average RGB", " / ".join(str(round(v)) for v in color["mean_rgb"]), "red / green / blue")

image_tab, defects_tab, texture_tab, color_tab, details_tab = st.tabs(["Overview", f"Defects ({defects['total']})", "Texture & weave", "Color", "Technical details"])
with image_tab:
    original_col, overlay_col = st.columns(2)
    original_col.markdown("#### Original image")
    original_col.image(upload.getvalue(), use_container_width=True)
    overlay_col.markdown("#### Potential defects")
    overlay_col.image(overlay, use_container_width=True)
    if defects["total"]:
        st.warning("Highlighted regions are potential defects. Review them against the original image before making a quality decision.")
    else:
        st.success("No potential defects detected at the current sensitivity.")

with defects_tab:
    count_cols = st.columns(3)
    for col, label in zip(count_cols, ("hole", "stain", "missing_yarn")):
        col.metric(label.replace("_", " ").title(), defects["counts"].get(label, 0))
    if defects["items"]:
        defect_df = pd.DataFrame(defects["items"])
        defect_df["bbox"] = defect_df["bbox"].apply(lambda box: ", ".join(map(str, box)))
        defect_df = defect_df.rename(columns={"type":"Type", "area_mm2":"Area (mm²)", "length_mm":"Length (mm)", "width_mm":"Width (mm)", "circularity":"Circularity", "bbox":"Position (px: x, y, w, h)"})
        shown = st.dataframe(defect_df[["Type", "Area (mm²)", "Length (mm)", "Width (mm)", "Circularity", "Position (px: x, y, w, h)"]].round(2), use_container_width=True, hide_index=True)
        st.download_button("Download defect list (CSV)", defect_df.to_csv(index=False).encode("utf-8"), file_name="fabric_defects.csv", mime="text/csv")
    else:
        st.success("No defects were identified in this image.")
    st.caption("Defect labels are heuristic estimates. Validate findings visually, especially on unfamiliar fabrics.")

with texture_tab:
    left, right = st.columns(2)
    with left:
        st.markdown("#### Weave regularity")
        st.metric("Regularity score", metric_value(regularity["regularity_score"], 3))
        st.caption("A higher autocorrelation score generally indicates a more periodic pattern. This score is not calibrated as a pass/fail grade.")
        for direction in ("vertical_threads", "horizontal_threads"):
            values = thread[direction]
            st.markdown(f"**{direction.replace('_', ' ').title()}** · {values['tpi_fft']:.1f} TPI (FFT), {values['tpi_peaks']:.1f} TPI (peaks)")
    with right:
        st.markdown("#### Texture descriptors")
        st.dataframe(pd.DataFrame(result["texture"]["glcm"], index=["Value"]).T.rename_axis("GLCM measure"), use_container_width=True)
        st.markdown("**Gabor response by orientation**")
        st.dataframe(pd.DataFrame(result["texture"]["gabor"]).T, use_container_width=True)
    st.markdown("#### LBP histogram")
    st.bar_chart(pd.DataFrame({"Frequency": result["texture"]["lbp_hist"]}))

with color_tab:
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("#### Average color")
        rgb = tuple(round(v) for v in color["mean_rgb"])
        st.markdown(f"<div style='height:100px;border-radius:12px;background:rgb{rgb};border:1px solid #ddd'></div>", unsafe_allow_html=True)
        st.caption(f"RGB {rgb[0]}, {rgb[1]}, {rgb[2]} · HSV " + ", ".join(f"{v:.1f}" for v in color["mean_hsv"]))
        if "delta_e_2000_vs_reference" in color:
            st.metric("ΔE 2000 vs target", f"{color['delta_e_2000_vs_reference']:.2f}")
    with c2:
        st.markdown("#### RGB distribution")
        hist_df = pd.DataFrame(color["hist_rgb_8bins"])
        st.bar_chart(hist_df)
        st.caption("Eight-bin channel histograms; lighting and camera settings affect these measurements.")

with details_tab:
    st.markdown("#### Edge and contour measurements")
    edge = {key: value for key, value in result["edges"].items() if key != "hu_moments_log"}
    st.json(edge)
    with st.expander("Full analysis data (JSON)"):
        st.json(result)

st.divider()
json_bytes = json.dumps(result, indent=2, default=_np).encode("utf-8")
st.download_button("Download full analysis (JSON)", json_bytes, file_name="fabric_analysis.json", mime="application/json", type="primary")
st.caption("Use measurements as inspection support. Image quality, lighting, fabric type, and scale calibration can affect results.")

