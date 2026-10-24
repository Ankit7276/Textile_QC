# LoomLab — User Manual

## What the app does

LoomLab analyzes close-up fabric photos and reports estimated thread count, potential surface defects, weave regularity, texture, color, and edge measurements. It also supports batch comparisons, human review of defect candidates, configurable screening thresholds, PDF/CSV/JSON exports, and searchable saved history.

The analysis is an inspection aid. It is not a certified textile test or a replacement for visual quality control.

## Run locally

1. Install Python 3.10 or newer.
2. Open a terminal in the project folder.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start LoomLab:

   ```bash
   streamlit run streamlit_app.py
   ```

5. Open the local address shown in the terminal (usually `http://localhost:8501`).

## Inspect one or more photos

1. Upload one or more JPG, JPEG, PNG, BMP, TIF, or TIFF photos in the sidebar.
2. Give the inspection batch a name, such as a sample ID, fabric lot, or purchase order.
3. Choose one scale method:
   - **Known photo width:** enter the real fabric width represented by the full image width, in inches.
   - **Known marker distance:** enter the distance in pixels between two points on a ruler or known-size marker, plus the real distance in inches. LoomLab calculates PPI from these values. Measure pixel distance in the original image using an image viewer or measurement tool. This PPI is applied to every photo in the current batch, so use consistently scaled photos.
   - **Known PPI:** enter a previously calibrated pixels-per-inch value.
4. Set defect sensitivity. Lower values find more subtle changes and may increase false detections.
5. Optionally choose a target color for ΔE 2000 comparison.
6. In **Quality thresholds**, set the screening limits for defect count, thread count, regularity, and (when color comparison is enabled) color difference.
7. Review the batch comparison table and choose a sample to inspect its images and details.

### Calibration notes

A wrong scale gives wrong physical defect dimensions. For marker calibration, include the marker in the same plane as the fabric and avoid perspective distortion. Do not use a ruler that is noticeably closer to the camera than the fabric. Blur, angle, and image resizing can affect the pixel measurement.

## Review defect candidates

Open **Defect review** for the selected sample. Select a candidate, then choose whether it is confirmed or a false positive and correct its label if needed. Click **Save defect review**. Decisions are stored with the inspection and shown in the history and PDF report.

Initial defect types (hole, stain, missing yarn) are heuristic labels. Review them before using the data for a production decision.

## Understand the quality result

- **FAIL:** one or more configured numeric limits were exceeded.
- **REVIEW:** detected defect candidates need human review, and no configured limit failed.
- **PASS:** the configured limits were met and no defect candidates were detected.

These are user-configured screening results, not industry certification. Calibrate limits against representative samples and your own acceptance criteria. A PASS does not prove that the material is defect-free.

## Export reports

- **Batch comparison CSV:** summary values for all successfully analyzed photos in the batch.
- **Batch PDF report:** sample metrics, quality outcome, defect dimensions and review decisions, and the annotated image for each saved sample with the current batch name.
- **Sample JSON:** all detailed analysis measurements for the selected photo.
- **Saved inspection PDF:** report for the selected historical inspection.

## Search saved inspections

Open **Saved history** and search by batch name or image filename. Select an entry to view its original and annotated images and key measurements. Each analysis is saved automatically and keyed by image contents and analysis settings; rerunning an identical analysis updates the existing entry instead of adding another duplicate.

## Database, privacy, and backups

By default, LoomLab creates `loomlab_history.db` in the project folder. It contains inspection results, uploaded image bytes, annotated images, and review decisions. The included `.gitignore` excludes this database from Git. Keep a secure backup if the history matters. Set `LOOMLAB_DB_PATH` to choose another database file location.

The app does not include user accounts or per-user permissions. Anyone who can access the running app can access its saved inspections. On Streamlit Community Cloud, a local SQLite file may not survive restarts or redeployments and is not a reliable shared database. For durable multi-user production use, configure persistent storage or move history to a hosted database and store images in durable object storage.

## Limitations

Results depend on image scale, focus, resolution, lighting, camera white balance, fabric type, and weave orientation. Defects can be missed, merged, or mislabeled. Color values from ordinary photos are not laboratory color measurements. Always confirm important findings against the original fabric and a calibrated measurement process.

## Run the automated tests

From the project folder, run python -m unittest discover -s tests -v. The tests use synthetic images and do not require sample fabric photos.
