"""Fabric image analyzer.

Features: thread count (FFT + peak counting), defect detection/size,
texture (GLCM, LBP, Gabor), color (RGB/HSV, histogram, Delta E),
edge/contour descriptors, and pattern regularity.

Install: pip install opencv-python scikit-image scipy numpy

Usage:
    python fabric_analyzer.py fabric.jpg --width-in 1.0
    python fabric_analyzer.py fabric.jpg --ppi 600 --ref-image good.jpg \
        --overlay out.png --json out.json
"""
import argparse
import json

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

MM2_PER_IN2 = 645.16
MM_PER_IN = 25.4


# ---------- helpers ----------
def flatten_lighting(gray):
    """Subtract a heavily blurred copy to remove uneven illumination."""
    g = gray.astype(np.float32)
    return g - cv2.GaussianBlur(g, (0, 0), max(g.shape) / 8)


def _peak_freq(profile, min_period=3, max_period=60, pad=8):
    p = profile - profile.mean()
    p = p * np.hanning(len(p))
    n = len(p) * pad  # zero-pad for finer frequency resolution
    spec = np.abs(np.fft.rfft(p, n=n))
    freqs = np.fft.rfftfreq(n)
    m = (freqs > 1 / max_period) & (freqs < 1 / min_period)
    i = np.argmax(spec[m])
    return freqs[m][i], spec[m][i] / (spec[m].mean() + 1e-9)


def _profiles(flat):
    # vertical threads vary along x; horizontal threads vary along y
    return (("vertical_threads", flat.mean(axis=0)),
            ("horizontal_threads", flat.mean(axis=1)))


# ---------- 1. thread count ----------
def thread_count(flat, ppi):
    res, total_fft, total_peaks = {}, 0.0, 0.0
    for name, prof in _profiles(flat):
        f, strength = _peak_freq(prof)
        period = 1 / f
        sm = gaussian_filter1d(prof, period / 6)
        peaks, _ = find_peaks(sm, distance=max(1, int(period * 0.7)),
                              prominence=sm.std() * 0.2)
        span_in = len(prof) / ppi
        res[name] = {"tpi_fft": f * ppi, "tpi_peaks": len(peaks) / span_in,
                     "period_px": period, "fft_peak_strength": strength}
        total_fft += f * ppi
        total_peaks += len(peaks) / span_in
    res["thread_count_fft"] = total_fft
    res["thread_count_peaks"] = total_peaks
    return res


# ---------- 2 & 3. defects: count + size ----------
def detect_defects(bgr, flat, ppi, period_px, k=5.0, min_area_px=40):
    sm = cv2.GaussianBlur(flat, (0, 0), max(1.5, period_px * 0.6))  # suppress weave
    med = np.median(sm)
    mad = np.median(np.abs(sm - med)) * 1.4826 + 1e-6
    z = (sm - med) / mad
    mask = (np.abs(z) > k).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    fabric_lab = np.median(lab.reshape(-1, 3), axis=0)
    defects = []
    img_area = flat.shape[0] * flat.shape[1]
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area_px:
            continue
        in_comp = labels == i
        # Blur widens defects, so tighten to ~40% of the peak deviation
        peak = np.abs(z[in_comp]).max()
        comp = (in_comp & (np.abs(z) > max(k, 0.4 * peak))).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        area, per = cv2.contourArea(c), cv2.arcLength(c, True)
        x, y, bw, bh = cv2.boundingRect(c)
        if area < min_area_px or bw * bh > 0.25 * img_area:
            continue  # too small, or a lighting artifact rather than a defect
        circ = 4 * np.pi * area / per ** 2 if per > 0 else 0.0
        (_, _), (w, h), _ = cv2.minAreaRect(c)
        length, width = max(w, h), min(w, h)
        aspect = length / max(width, 1e-6)
        zmean = float(z[in_comp].mean())
        # Rough heuristic labels; tune on your own fabrics
        reg = lab[in_comp].mean(axis=0)
        d_light = abs(reg[0] - fabric_lab[0])
        d_chroma = float(np.hypot(*(reg[1:] - fabric_lab[1:])))
        if aspect >= 4:
            kind = "missing_yarn"
        elif d_chroma > 0.5 * d_light:   # color shift dominates -> stain
            kind = "stain"
        elif circ >= 0.6:
            kind = "hole"
        else:
            kind = "stain"
        defects.append({
            "type": kind, "bbox": [x, y, bw, bh],
            "area_px": float(area), "area_mm2": area / ppi ** 2 * MM2_PER_IN2,
            "length_mm": length / ppi * MM_PER_IN, "width_mm": width / ppi * MM_PER_IN,
            "circularity": circ, "mean_z": zmean,
        })
    counts = {t: sum(d["type"] == t for d in defects)
              for t in ("hole", "stain", "missing_yarn")}
    return {"total": len(defects), "counts": counts, "items": defects}


# ---------- 4. texture ----------
def texture_features(gray, period_px):
    s = 1500 / max(gray.shape)
    g = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else gray

    glcm = graycomatrix(g // 4, [1, 2, 4], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=64, symmetric=True, normed=True)
    out = {"glcm": {p: float(graycoprops(glcm, p).mean())
                    for p in ("contrast", "energy", "homogeneity", "correlation")}}

    lbp = local_binary_pattern(g, 8, 1, "uniform")
    hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
    out["lbp_hist"] = hist.tolist()

    gf = gray.astype(np.float32)
    lam = float(np.clip(period_px, 4, 20))
    gabor = {}
    for deg in (0, 45, 90, 135):
        kern = cv2.getGaborKernel((31, 31), 4.0, np.deg2rad(deg), lam, 0.5, 0)
        r = cv2.filter2D(gf, -1, kern)
        gabor[f"{deg}deg"] = {"mean_abs": float(np.abs(r).mean()), "std": float(r.std())}
    out["gabor"] = gabor
    return out


# ---------- 5. color ----------
def color_features(bgr, ref_rgb=None):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] *= 2          # OpenCV hue 0-179 -> degrees
    hsv[..., 1:] /= 255.0
    mean_rgb = rgb.reshape(-1, 3).mean(0)
    out = {
        "mean_rgb": mean_rgb.tolist(),
        "std_rgb": rgb.reshape(-1, 3).std(0).tolist(),
        "mean_hsv": hsv.reshape(-1, 3).mean(0).tolist(),
        "std_hsv": hsv.reshape(-1, 3).std(0).tolist(),
        "hist_rgb_8bins": {c: np.histogram(rgb[..., i], bins=8, range=(0, 256),
                                           density=True)[0].tolist()
                           for i, c in enumerate("RGB")},
    }
    if ref_rgb is not None:
        lab = rgb2lab((mean_rgb / 255.0).reshape(1, 1, 3))
        ref = rgb2lab((np.asarray(ref_rgb, float) / 255.0).reshape(1, 1, 3))
        out["delta_e_2000_vs_reference"] = float(deltaE_ciede2000(lab, ref)[0, 0])
    return out


# ---------- 6. edges / contours ----------
def edge_features(gray):
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circs, sols, asps = [], [], []
    for c in cnts:
        a, p = cv2.contourArea(c), cv2.arcLength(c, True)
        if a < 20 or p == 0:
            continue
        hull_a = cv2.contourArea(cv2.convexHull(c))
        _, _, w, h = cv2.boundingRect(c)
        circs.append(4 * np.pi * a / p ** 2)
        sols.append(a / hull_a if hull_a > 0 else 0)
        asps.append(max(w, h) / max(1, min(w, h)))
    hu = cv2.HuMoments(cv2.moments(edges, binaryImage=True)).ravel()
    mean = lambda v: float(np.mean(v)) if v else 0.0
    return {"edge_density": float((edges > 0).mean()), "contour_count": len(cnts),
            "mean_circularity": mean(circs), "mean_solidity": mean(sols),
            "mean_aspect_ratio": mean(asps),
            "hu_moments_log": (-np.sign(hu) * np.log10(np.abs(hu) + 1e-30)).tolist()}


# ---------- 7. pattern regularity ----------
def pattern_regularity(flat):
    out, scores = {}, []
    for name, prof in _profiles(flat):
        p = prof - prof.mean()
        ac = np.correlate(p, p, "full")[len(p) - 1:]
        ac = ac / (ac[0] + 1e-9)
        pk, _ = find_peaks(ac[: len(ac) // 2], height=0)
        peak = float(ac[pk].max()) if len(pk) else 0.0
        _, strength = _peak_freq(prof)
        out[name] = {"autocorr_peak": peak, "fft_peak_strength": float(strength)}
        scores.append(peak)
    out["regularity_score"] = float(np.mean(scores))
    out["low_periodicity"] = bool(out["regularity_score"] < 0.3)  # uncalibrated threshold
    return out


# ---------- pipeline ----------
def analyze(path, ppi=None, width_in=None, ref_rgb=None, overlay_path=None, k=5.0):
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(path)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if ppi is None:
        ppi = gray.shape[1] / width_in
    flat = flatten_lighting(gray)

    tc = thread_count(flat, ppi)
    period = np.mean([tc["vertical_threads"]["period_px"],
                      tc["horizontal_threads"]["period_px"]])
    defects = detect_defects(bgr, flat, ppi, period, k=k)

    if overlay_path:
        vis = bgr.copy()
        for d in defects["items"]:
            x, y, w, h = d["bbox"]
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(vis, d["type"], (x, max(12, y - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imwrite(overlay_path, vis)

    return {"scale_ppi": ppi, "thread_count": tc, "defects": defects,
            "texture": texture_features(gray, period),
            "color": color_features(bgr, ref_rgb),
            "edges": edge_features(gray),
            "regularity": pattern_regularity(flat)}


def _np(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--width-in", type=float, help="real width of the photo, inches")
    g.add_argument("--ppi", type=float, help="pixels per inch of the photo")
    ap.add_argument("--ref-image", help="reference fabric image for Delta E")
    ap.add_argument("--ref-rgb", help="reference color as R,G,B")
    ap.add_argument("--overlay", help="save annotated defect image here")
    ap.add_argument("--json", help="save full results to this JSON file")
    a = ap.parse_args()

    ref = None
    if a.ref_image:
        r = cv2.cvtColor(cv2.imread(a.ref_image), cv2.COLOR_BGR2RGB)
        ref = r.reshape(-1, 3).mean(0)
    elif a.ref_rgb:
        ref = [float(v) for v in a.ref_rgb.split(",")]

    result = analyze(a.image, a.ppi, a.width_in, ref, a.overlay)
    text = json.dumps(result, indent=2, default=_np)
    if a.json:
        open(a.json, "w").write(text)
    tc, d = result["thread_count"], result["defects"]
    print(f"Thread count (FFT / peaks): {tc['thread_count_fft']:.0f} / {tc['thread_count_peaks']:.0f}")
    print(f"Defects: {d['total']} {d['counts']}")
    print(f"Regularity: {result['regularity']['regularity_score']:.2f}"
          f"{'  (LOW)' if result['regularity']['low_periodicity'] else ''}")
    if "delta_e_2000_vs_reference" in result["color"]:
        print(f"Delta E vs reference: {result['color']['delta_e_2000_vs_reference']:.2f}")
    if not a.json:
        print(text)
