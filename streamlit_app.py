"""LoomLab: fabric quality analysis, review, reporting, and history."""
import hashlib
import io
import json
import os
import sqlite3
import tempfile
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as PDFImage, PageBreak

from fabric_analyzer import _np, analyze

st.set_page_config(page_title="LoomLab | Fabric quality", page_icon="🧵", layout="wide", initial_sidebar_state="expanded")
DB_PATH = os.environ.get("LOOMLAB_DB_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "loomlab_history.db"))

# Theme-aware styles use Streamlit's active theme tokens. The inspection header is intentionally green in both themes.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
.stApp{background:var(--background-color);color:var(--text-color)}
.block-container{padding-top:1.35rem;max-width:1450px}
[data-testid="stSidebar"]{background:var(--secondary-background-color);border-right:1px solid color-mix(in srgb,var(--text-color) 13%,transparent)}
.hero{background:linear-gradient(120deg,#183d30,#276b4e);color:#fff;padding:1.35rem 1.65rem;border-radius:18px;margin-bottom:1.1rem}
.hero h1{font-family:Manrope,sans-serif;font-size:1.75rem;margin:0;color:#fff}
.hero p{color:#e0eee6;margin:.35rem 0 0;font-size:.94rem}
.eyebrow{text-transform:uppercase;letter-spacing:.13em;font-size:.7rem;font-weight:700;opacity:.82;margin-bottom:.4rem}
div[data-testid="stMetric"]{background:var(--secondary-background-color);color:var(--text-color);border:1px solid color-mix(in srgb,var(--text-color) 13%,transparent);padding:.85rem 1rem;border-radius:14px;box-shadow:0 2px 8px color-mix(in srgb,var(--text-color) 6%,transparent)}
div[data-testid="stMetricLabel"],div[data-testid="stMetricLabel"] p{color:var(--text-color);opacity:.78}
div[data-testid="stMetricValue"]{color:var(--text-color)}
div[data-testid="stExpander"]{background:var(--secondary-background-color);border:1px solid color-mix(in srgb,var(--text-color) 13%,transparent);border-radius:12px}
.stTabs [data-baseweb="tab"]{color:var(--text-color)}
.small-note{color:var(--text-color);opacity:.72;font-size:.84rem}
/* Keep the mobile drawer opaque and above the main page during its slide transition. */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div:first-child,
[data-testid="stSidebarContent"] { background-color:var(--secondary-background-color) !important; opacity:1 !important; }
@media (max-width: 700px) {
  .block-container { padding: .7rem .8rem 2rem; }
  .hero { padding: 1rem 1.1rem; border-radius: 14px; }
  .hero h1 { font-size: 1.35rem; line-height: 1.25; }
  .hero p { font-size: .86rem; }
  [data-testid="stAppViewContainer"] { position:relative; z-index:0 !important; }
  [data-testid="stHeader"] { z-index:999990 !important; }
  [data-testid="stSidebar"] { min-width:min(88vw,340px) !important; max-width:min(88vw,340px) !important; z-index:1000001 !important; isolation:isolate; }
  [data-testid="stSidebar"] > div:first-child { position:relative; z-index:1000001; background-color:var(--secondary-background-color) !important; }
  [data-testid="stSidebar"] > div:first-child { box-shadow: 8px 0 28px rgba(0,0,0,.22); }
  [data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; gap: .65rem !important; }
  [data-testid="stHorizontalBlock"] > [data-testid="column"] { min-width: calc(50% - .4rem) !important; flex: 1 1 calc(50% - .4rem) !important; }
  .stTabs [data-baseweb="tab-list"] { gap: .5rem; overflow-x: auto; flex-wrap: nowrap; }
  .stTabs [data-baseweb="tab"] { white-space: nowrap; padding: .65rem .35rem; }
}
@media (max-width: 430px) {
  [data-testid="stHorizontalBlock"] > [data-testid="column"] { min-width: 100% !important; flex: 1 1 100% !important; }
  div[data-testid="stMetric"] { padding: .7rem .8rem; }
}
</style>
""", unsafe_allow_html=True)


def db_connect():
    connection = sqlite3.connect(DB_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("""CREATE TABLE IF NOT EXISTS inspections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        analysis_key TEXT UNIQUE NOT NULL,
        created_at TEXT NOT NULL,
        batch_name TEXT NOT NULL,
        image_name TEXT NOT NULL,
        image_sha TEXT NOT NULL,
        scale_ppi REAL NOT NULL,
        result_json TEXT NOT NULL,
        original_blob BLOB NOT NULL,
        overlay_blob BLOB NOT NULL,
        review_json TEXT NOT NULL,
        quality_status TEXT NOT NULL
    )""")
    connection.commit()
    return connection


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


def grade_result(result, limits):
    defects = result["defects"]["total"]
    tc = result["thread_count"]["thread_count_fft"]
    regularity = result["regularity"]["regularity_score"]
    reasons = []
    failed = []
    if defects > limits["defects_max"]:
        failed.append(f"{defects} potential defects exceeds limit {limits['defects_max']}")
    elif defects:
        reasons.append(f"{defects} potential defect(s) require visual review")
    if tc < limits["tpi_min"] or tc > limits["tpi_max"]:
        failed.append(f"thread count {tc:.1f} is outside {limits['tpi_min']:.1f}–{limits['tpi_max']:.1f} TPI")
    if regularity < limits["regularity_min"]:
        failed.append(f"regularity {regularity:.2f} is below {limits['regularity_min']:.2f}")
    delta = result["color"].get("delta_e_2000_vs_reference")
    if limits["delta_e_max"] is not None and delta is not None and delta > limits["delta_e_max"]:
        failed.append(f"color difference ΔE {delta:.1f} exceeds {limits['delta_e_max']:.1f}")
    status = "FAIL" if failed else ("REVIEW" if reasons else "PASS")
    return status, failed + reasons


def make_pdf(records, report_title):
    output = io.BytesIO()
    doc = SimpleDocTemplate(output, pagesize=letter, rightMargin=0.55*inch, leftMargin=0.55*inch, topMargin=0.55*inch, bottomMargin=0.55*inch)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="LoomTitle", parent=styles["Title"], textColor=colors.HexColor("#1d5b43"), alignment=TA_CENTER, spaceAfter=8))
    story = [Paragraph("LoomLab Fabric Inspection Report", styles["LoomTitle"]), Paragraph(report_title, styles["Heading2"]), Paragraph("Generated " + datetime.now().strftime("%Y-%m-%d %H:%M"), styles["Normal"]), Spacer(1, 12)]
    for index, row in enumerate(records):
        result = json.loads(row["result_json"])
        tc, defects = result["thread_count"], result["defects"]
        values = [["Image", row["image_name"]], ["Inspection time", row["created_at"]], ["Quality result", row["quality_status"]], ["Scale", f"{row['scale_ppi']:.1f} PPI"], ["Thread count", f"{tc['thread_count_fft']:.1f} TPI"], ["Vertical / horizontal", f"{tc['vertical_threads']['tpi_fft']:.1f} / {tc['horizontal_threads']['tpi_fft']:.1f} TPI"], ["Potential defects", str(defects["total"])], ["Regularity", f"{result['regularity']['regularity_score']:.3f}"]]
        delta = result["color"].get("delta_e_2000_vs_reference")
        if delta is not None:
            values.append(["Color difference", f"ΔE 2000: {delta:.2f}"])
        table = Table(values, colWidths=[1.8*inch, 5.1*inch])
        table.setStyle(TableStyle([("BACKGROUND",(0,0),(0,-1),colors.HexColor("#eaf2ed")),("TEXTCOLOR",(0,0),(-1,-1),colors.HexColor("#20342a")),("GRID",(0,0),(-1,-1),.4,colors.HexColor("#d5e0d9")),("FONTNAME",(0,0),(0,-1),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),9),("PADDING",(0,0),(-1,-1),6)]))
        story += [Paragraph(row["image_name"], styles["Heading2"]), table, Spacer(1, 8)]
        if defects["items"]:
            drows = [["#", "Type", "Area mm²", "Length mm", "Width mm", "Review"]]
            reviews = json.loads(row["review_json"])
            for i, defect in enumerate(defects["items"]):
                review = reviews[i] if i < len(reviews) else {}
                drows.append([str(i+1), review.get("label", defect["type"]), f"{defect['area_mm2']:.2f}", f"{defect['length_mm']:.2f}", f"{defect['width_mm']:.2f}", review.get("status", "Unreviewed")])
            dt = Table(drows, repeatRows=1, colWidths=[.35*inch, 1.2*inch, 1*inch, 1.15*inch, 1.05*inch, 1.2*inch])
            dt.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#276b4e")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),.35,colors.HexColor("#d5e0d9")),("FONTSIZE",(0,0),(-1,-1),8),("PADDING",(0,0),(-1,-1),5)]))
            story += [dt, Spacer(1,8)]
        story += [PDFImage(io.BytesIO(row["overlay_blob"]), width=6.6*inch, height=4.1*inch, kind="proportional"), Spacer(1,8), Paragraph("Automated measurements are estimates and should be verified visually. Results depend on image quality and correct scale calibration.", styles["Italic"])]
        if index < len(records)-1:
            story.append(PageBreak())
    doc.build(story)
    return output.getvalue()


def get_record(record_id):
    with db_connect() as connection:
        return connection.execute("SELECT * FROM inspections WHERE id=?", (record_id,)).fetchone()


def save_review(record_id, defect_index, status, label):
    with db_connect() as connection:
        row = connection.execute("SELECT review_json,result_json FROM inspections WHERE id=?", (record_id,)).fetchone()
        reviews = json.loads(row["review_json"])
        result = json.loads(row["result_json"])
        while len(reviews) < len(result["defects"]["items"]):
            reviews.append({"status":"Unreviewed","label":result["defects"]["items"][len(reviews)]["type"]})
        reviews[defect_index] = {"status":status,"label":label}
        connection.execute("UPDATE inspections SET review_json=? WHERE id=?", (json.dumps(reviews), record_id))
        connection.commit()


with st.sidebar:
    st.markdown("## 🧵 LoomLab")
    st.caption("Fabric inspection workspace")
    st.divider()
    st.markdown("#### 1 · Add fabric photos")
    uploads = st.file_uploader("Upload one or more close-up images", type=["jpg","jpeg","png","bmp","tif","tiff"], accept_multiple_files=True, help="Sharp, evenly lit, straight-on images give more reliable results.")
    batch_name = st.text_input("Inspection / batch name", value="Fabric inspection")
    st.markdown("#### 2 · Calibrate image scale")
    scale_mode = st.radio("Scale method", ["Known photo width", "Known marker distance", "Known PPI"], label_visibility="collapsed")
    ppi = None
    width_in = None
    marker_length_in = None
    if scale_mode == "Known photo width":
        width_in = st.number_input("Fabric width across full image (inches)", min_value=0.1, max_value=100.0, value=1.0, step=0.1, help="Enter the real fabric width represented by the full image width.")
    elif scale_mode == "Known marker distance":
        st.caption("Use the image's native pixel coordinates. Measure the pixel distance between two known points, then enter the known real distance.")
        marker_px = st.number_input("Distance between marker points (pixels)", min_value=1.0, value=100.0, step=1.0)
        marker_length_in = st.number_input("Known distance (inches)", min_value=0.01, value=0.5, step=0.1)
        ppi = marker_px / marker_length_in
        st.caption(f"Calculated scale: {ppi:.1f} PPI")
    else:
        ppi = st.number_input("Known pixels per inch", min_value=10.0, max_value=5000.0, value=600.0, step=10.0)
    st.markdown("#### 3 · Analysis settings")
    sensitivity = st.slider("Defect sensitivity", 2.0, 10.0, 5.0, 0.5, help="Lower values flag subtler variations and may increase false positives.")
    compare_color = st.checkbox("Compare against target color")
    ref_rgb = None
    if compare_color:
        hex_color = st.color_picker("Target color", "#968264").lstrip("#")
        ref_rgb = tuple(int(hex_color[i:i+2],16) for i in (0,2,4))
    with st.expander("Quality thresholds", expanded=False):
        st.caption("Used for a configurable screening result. These are not certified fabric standards.")
        defects_max = st.number_input("Maximum potential defects", min_value=0, value=0, step=1)
        tpi_min = st.number_input("Minimum thread count (TPI)", min_value=0.0, value=0.0, step=1.0)
        tpi_max = st.number_input("Maximum thread count (TPI)", min_value=0.0, value=10000.0, step=1.0)
        regularity_min = st.slider("Minimum regularity score", 0.0, 1.0, 0.0, 0.05)
        delta_e_max = st.number_input("Maximum ΔE (0 disables)", min_value=0.0, value=0.0, step=0.5)
    st.divider()
    st.caption("Saved inspection history is stored in a local SQLite file.")

limits = {"defects_max":defects_max,"tpi_min":tpi_min,"tpi_max":max(tpi_min,tpi_max),"regularity_min":regularity_min,"delta_e_max":delta_e_max if delta_e_max > 0 else None}
st.markdown('<div class="hero"><div class="eyebrow">Inspection · calibration · quality review</div><h1>Fabric quality analysis</h1><p>Analyze samples, review potential defects, compare batches, and keep a searchable inspection record.</p></div>', unsafe_allow_html=True)
analysis_tab, history_tab, guide_tab = st.tabs(["Current inspection", "Saved history", "How to use"])

with analysis_tab:
    if not uploads:
        st.info("Upload one or more fabric photos in the sidebar to begin. The calibration controls can be set before or after upload.")
    else:
        if tpi_max < tpi_min:
            st.error("Maximum TPI must be at least the minimum TPI.")
            st.stop()
        file_results = []
        progress = st.progress(0, text="Preparing inspection…")
        for position, uploaded in enumerate(uploads, start=1):
            data = uploaded.getvalue()
            image_sha = hashlib.sha256(data).hexdigest()
            try:
                result, overlay = run_analysis(data, os.path.splitext(uploaded.name)[1] or ".png", ppi, width_in, ref_rgb, sensitivity)
            except Exception as error:
                st.error(f"{uploaded.name}: could not analyze this image — {error}")
                progress.progress(position / len(uploads), text=f"Processed {position} of {len(uploads)}")
                continue
            status, reasons = grade_result(result, limits)
            key_data = json.dumps({"sha":image_sha,"batch":batch_name,"ppi":round(result["scale_ppi"],4),"sensitivity":sensitivity,"ref":ref_rgb,"limits":limits}, sort_keys=True)
            analysis_key = hashlib.sha256(key_data.encode()).hexdigest()
            with db_connect() as connection:
                row = connection.execute("SELECT id FROM inspections WHERE analysis_key=?", (analysis_key,)).fetchone()
                if row:
                    record_id = row["id"]
                    connection.execute("UPDATE inspections SET batch_name=?,image_name=?,quality_status=? WHERE id=?", (batch_name,uploaded.name,status,record_id))
                else:
                    cursor = connection.execute("INSERT INTO inspections (analysis_key,created_at,batch_name,image_name,image_sha,scale_ppi,result_json,original_blob,overlay_blob,review_json,quality_status) VALUES (?,?,?,?,?,?,?,?,?,?,?)", (analysis_key,datetime.now().strftime("%Y-%m-%d %H:%M:%S"),batch_name,uploaded.name,image_sha,result["scale_ppi"],json.dumps(result,default=_np),data,overlay,json.dumps([]),status))
                    record_id = cursor.lastrowid
                connection.commit()
            file_results.append({"id":record_id,"name":uploaded.name,"data":data,"overlay":overlay,"result":result,"status":status,"reasons":reasons})
            progress.progress(position / len(uploads), text=f"Analyzed {position} of {len(uploads)}")
        progress.empty()
        if file_results:
            summary_rows = []
            for item in file_results:
                result = item["result"]
                summary_rows.append({"Image":item["name"],"Result":item["status"],"Thread count (TPI)":round(result["thread_count"]["thread_count_fft"],1),"Vertical TPI":round(result["thread_count"]["vertical_threads"]["tpi_fft"],1),"Horizontal TPI":round(result["thread_count"]["horizontal_threads"]["tpi_fft"],1),"Potential defects":result["defects"]["total"],"Regularity":round(result["regularity"]["regularity_score"],3),"ΔE":round(result["color"].get("delta_e_2000_vs_reference",float("nan")),2)})
            st.markdown("### Batch comparison")
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
            if len(file_results) > 1:
                st.download_button("Download batch comparison (CSV)",pd.DataFrame(summary_rows).to_csv(index=False).encode(),"fabric_batch_comparison.csv","text/csv")
            selected_name = st.selectbox("Review sample", [item["name"] for item in file_results])
            selected = next(item for item in file_results if item["name"] == selected_name)
            result, defects = selected["result"], selected["result"]["defects"]
            status_color = {"PASS":"green","REVIEW":"orange","FAIL":"red"}[selected["status"]]
            st.markdown(f"### {selected['name']} · :{status_color}[{selected['status']}]")
            if selected["reasons"]:
                for reason in selected["reasons"]:
                    st.warning(reason)
            elif selected["status"] == "PASS":
                st.success("This sample meets the configured screening thresholds.")
            m = st.columns(5)
            m[0].metric("Thread count",f"{result['thread_count']['thread_count_fft']:.0f}","TPI")
            m[1].metric("Potential defects",defects["total"])
            m[2].metric("Regularity",f"{result['regularity']['regularity_score']:.2f}")
            m[3].metric("Scale",f"{result['scale_ppi']:.0f}","PPI")
            delta=result["color"].get("delta_e_2000_vs_reference")
            m[4].metric("Color difference",f"ΔE {delta:.1f}" if delta is not None else "Not set")
            overview, review, texture, color_tab, raw = st.tabs(["Images",f"Defect review ({defects['total']})","Texture & weave","Color","Technical data"])
            with overview:
                left,right=st.columns(2)
                left.image(selected["data"],caption="Original fabric photo",use_container_width=True)
                right.image(selected["overlay"],caption="Potential defects highlighted",use_container_width=True)
                st.caption("Red boxes indicate algorithm candidates. Verify each one against the original photo.")
            with review:
                if defects["items"]:
                    with db_connect() as connection:
                        saved=connection.execute("SELECT review_json FROM inspections WHERE id=?",(selected["id"],)).fetchone()
                    reviews=json.loads(saved["review_json"])
                    while len(reviews)<len(defects["items"]): reviews.append({"status":"Unreviewed","label":defects["items"][len(reviews)]["type"]})
                    defect_idx=st.selectbox("Select candidate",range(len(defects["items"])),format_func=lambda i:f"#{i+1} · {reviews[i].get('label',defects['items'][i]['type'])} · {defects['items'][i]['area_mm2']:.2f} mm²")
                    defect=defects["items"][defect_idx]
                    st.write(f"Position: {defect['bbox']} pixels · Estimated size: {defect['length_mm']:.2f} × {defect['width_mm']:.2f} mm")
                    review_status=st.selectbox("Review decision",["Unreviewed","Confirmed","False positive"],index=["Unreviewed","Confirmed","False positive"].index(reviews[defect_idx].get("status","Unreviewed")))
                    corrected_label=st.selectbox("Defect label",["hole","stain","missing_yarn","other"],index=["hole","stain","missing_yarn","other"].index(reviews[defect_idx].get("label",defect["type"]) if reviews[defect_idx].get("label",defect["type"]) in ["hole","stain","missing_yarn","other"] else "other"))
                    if st.button("Save defect review",type="primary"):
                        save_review(selected["id"],defect_idx,review_status,corrected_label)
                        st.success("Review saved to inspection history.")
                        st.rerun()
                    st.dataframe(pd.DataFrame([{ "Type":d["type"],"Area (mm²)":d["area_mm2"],"Length (mm)":d["length_mm"],"Width (mm)":d["width_mm"],"Position":str(d["bbox"]),"Review":reviews[i].get("status","Unreviewed"),"Reviewed label":reviews[i].get("label",d["type"])} for i,d in enumerate(defects["items"])]).round(2),use_container_width=True,hide_index=True)
                else:
                    st.success("No potential defects were detected for this sample.")
            with texture:
                left,right=st.columns(2)
                left.metric("Weave regularity",f"{result['regularity']['regularity_score']:.3f}")
                left.json(result["regularity"])
                right.dataframe(pd.DataFrame(result["texture"]["glcm"],index=["Value"]).T,use_container_width=True)
                right.dataframe(pd.DataFrame(result["texture"]["gabor"]).T,use_container_width=True)
                st.bar_chart(pd.DataFrame({"LBP frequency":result["texture"]["lbp_hist"]}))
            with color_tab:
                c1,c2=st.columns(2)
                rgb=tuple(round(v) for v in result["color"]["mean_rgb"])
                c1.markdown(f"**Average color · RGB {rgb}**")
                c1.markdown(f"<div style='height:90px;border-radius:12px;background:rgb{rgb};border:1px solid #888'></div>",unsafe_allow_html=True)
                c2.bar_chart(pd.DataFrame(result["color"]["hist_rgb_8bins"]))
                if delta is not None: st.metric("ΔE 2000 vs target",f"{delta:.2f}")
            with raw:
                st.json({"edges":result["edges"],"thread_count":result["thread_count"],"regularity":result["regularity"]})
                with st.expander("Full analysis JSON"): st.json(result)
            st.download_button("Download this sample's analysis (JSON)",json.dumps(result,indent=2,default=_np).encode(),file_name="fabric_analysis.json",mime="application/json")
            with db_connect() as connection:
                record_ids=[item["id"] for item in file_results]
                placeholders=",".join("?" for _ in record_ids)
                batch_rows=connection.execute(f"SELECT * FROM inspections WHERE id IN ({placeholders}) ORDER BY created_at",record_ids).fetchall()
            if batch_rows:
                st.download_button("Download batch PDF report",make_pdf(batch_rows,batch_name),file_name="loomlab_inspection_report.pdf",mime="application/pdf")

with history_tab:
    st.markdown("### Saved inspection history")
    query=st.text_input("Search by batch or image name")
    with db_connect() as connection:
        if query.strip():
            rows=connection.execute("SELECT * FROM inspections WHERE batch_name LIKE ? OR image_name LIKE ? ORDER BY created_at DESC",(f"%{query}%",f"%{query}%")).fetchall()
        else:
            rows=connection.execute("SELECT * FROM inspections ORDER BY created_at DESC").fetchall()
    if rows:
        history_df=pd.DataFrame([{"ID":r["id"],"Date":r["created_at"],"Batch":r["batch_name"],"Image":r["image_name"],"Result":r["quality_status"],"TPI":round(json.loads(r["result_json"])["thread_count"]["thread_count_fft"],1),"Defects":json.loads(r["result_json"])["defects"]["total"]} for r in rows])
        st.dataframe(history_df,use_container_width=True,hide_index=True)
        selected_id=st.selectbox("Open saved inspection",history_df["ID"].tolist(),format_func=lambda x:next(f"#{r['id']} · {r['batch_name']} · {r['image_name']} · {r['created_at']}" for r in rows if r["id"]==x))
        record=get_record(selected_id)
        saved_result=json.loads(record["result_json"])
        left,right=st.columns(2)
        left.image(record["original_blob"],caption=record["image_name"],use_container_width=True)
        right.image(record["overlay_blob"],caption="Saved annotation",use_container_width=True)
        st.json({"quality_status":record["quality_status"],"scale_ppi":record["scale_ppi"],"thread_count":saved_result["thread_count"],"defects":saved_result["defects"],"regularity":saved_result["regularity"]})
        st.download_button("Download saved inspection PDF",make_pdf([record],record["image_name"]),file_name=f"loomlab_inspection_{record['id']}.pdf",mime="application/pdf")
    else:
        st.info("No saved inspections match this search. Results are added here after analysis.")
    st.caption(f"Database file: {DB_PATH}")

with guide_tab:
    st.markdown("""
### Quick user guide
1. Upload one or more sharp, evenly lit fabric photos.
2. Give the batch a useful name (for example, order or sample ID).
3. Calibrate scale. Use full photo width when the real width represented across the image is known. For a marker, measure the pixel distance between two known points in the original image, then enter that pixel distance and its real length. Known PPI is also accepted.
4. Set defect sensitivity and optional target color. Adjust quality thresholds in the sidebar for your inspection policy.
5. Compare samples in the batch table. Open each sample to review its image, defect candidates, and technical metrics.
6. Mark defect candidates Confirmed or False positive, correct their label, then save the review.
7. Download a sample JSON, batch CSV, or PDF report. Search saved results in the history tab.

**Quality result:** FAIL means a configured numeric limit was exceeded. REVIEW means one or more detected defects need human review. PASS means the configured limits were met and no detected candidate requires review. These are configurable screening outcomes, not certified textile grades.

**Calibration:** a wrong scale causes incorrect physical dimensions and may distort size reporting. The thread count image estimate itself is also affected by blur, lighting, weave, and resolution.

**Privacy and persistence:** this version stores images, measurements, and review decisions in a local SQLite database file named `loomlab_history.db`. Keep backups of that file. For a hosted deployment, local app storage may not be durable across restarts or redeploys; use persistent storage or a hosted database before relying on it as a permanent business archive. This app has no user login or per-user access controls.
""")

st.caption("LoomLab is an inspection aid. Verify important findings visually and validate thresholds against your fabrics and process.")
