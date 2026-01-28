#!/usr/bin/env python3
# tiff_normalization.py
# -*- coding: utf-8 -*-


import os
import gc
import tifffile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy import fftpack, stats
from skimage.measure import shannon_entropy
from skimage.feature import graycomatrix, graycoprops


# Configuration
DATA_DIR = "./zenodo_data"
OUTPUT_DIR = "./results"
CROP_SIZE = 128
STEP_SIZE = 128
CUTOFF_FREQ = 50
MAX_CHANNELS = 8

# Filtration thresholds
MIN_MEAN_INTENSITY = 2.0
MIN_ENTROPY = 0.5
MAX_SATURATION_RATIO = 0.05


def ensure_dirs():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)


def load_channel_names(data_dir):
    path = os.path.join(data_dir, 'panel.tsv')
    mapping = {}
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, sep='\t')
            if 'channel' in df.columns and 'name' in df.columns:
                mapping = dict(zip(df['channel'], df['name']))
        except Exception as e:
            print(f"[WARNING]: Could not load panel.tsv: {e}")
    return mapping


def scan_tissue_sliding_window(image, crop_size=256, step=256):
    """  Scans the image using a sliding window.
    Returns: 
        List of valid crops.
        Dictionary containing scan statistics.
    """
    h, w = image.shape
    valid_crops = []

    print(f"    Scanning image {w}x{h} (window {crop_size}, stride {step})...")
    
    count_total = 0
    count_background = 0
    count_artifact = 0
    
    for y in range(0, h - crop_size + 1, step):
        for x in range(0, w - crop_size + 1, step):
            count_total += 1
            crop = image[y:y+crop_size, x:x+crop_size]     
            
            # Mean intensity for background
            if np.mean(crop) < MIN_MEAN_INTENSITY:
                count_background += 1
                continue
                
            # Saturation artifacts
            max_val = np.max(crop)
            if max_val > 0:
                sat_ratio = np.sum(crop == max_val) / crop.size
                if sat_ratio > MAX_SATURATION_RATIO:
                    count_artifact += 1
                    continue
            
            # Entropy check
            if max_val > 0:
                 crop_u8 = (crop / max_val * 255).astype(np.uint8)
            else:
                 crop_u8 = np.zeros_like(crop, dtype=np.uint8)
                 
            entr = shannon_entropy(crop_u8)
            if entr < MIN_ENTROPY:
                count_background += 1
                continue
                
            valid_crops.append(crop)

    print(f"    Found {len(valid_crops)} valid tiles (out of {count_total} checked).")
    print(f"    Rejected: {count_background} (background/noise), {count_artifact} (artifacts).")

    scan_stats = {
        'Image_Width': w,
        'Image_Height': h,
        'Window_Size': crop_size,
        'Stride': step,
        'Total_Tiles_Checked': count_total,
        'Valid_Tiles': len(valid_crops),
        'Rejected_Background_Noise': count_background,
        'Rejected_Artifacts': count_artifact
    }
    
    return valid_crops, scan_stats


def z_score(img):
    s = np.std(img)
    return (img - np.mean(img)) / s if s > 0 else img - np.mean(img)


def butterworth_mask(shape, cutoff, order=2):
    rows, cols = shape
    y, x = np.ogrid[-rows//2:rows//2, -cols//2:cols//2]
    return 1 / (1 + (np.sqrt(x**2 + y**2) / (cutoff + 1e-5))**(2 * order))


def analyze_crop_spectral(crop, use_z=True, use_bw=True):
    """ Calculates radial profile and Beta slope.
    Flags allow toggling Z-Score and Butterworth filter.
    """
    # Preprocessing
    if use_z:
        img_proc = z_score(crop)
    else:
        img_proc = (crop - np.min(crop)) / (np.ptp(crop) + 1e-8)

    h, w = img_proc.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    img_w = img_proc * win
    
    # FFT
    f_shift = fftpack.fftshift(fftpack.fft2(img_w))
    
    # Filter
    if use_bw:
        f_shift = f_shift * butterworth_mask((h, w), CUTOFF_FREQ)
        
    psd = np.abs(f_shift)**2
    
    # Radial Profile
    y, x = np.indices((h, w))
    center = (h//2, w//2)
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2).astype(int)
    tbin = np.bincount(r.ravel(), psd.ravel())
    nr = np.bincount(r.ravel())
    profile = tbin / (nr + 1e-10)
    profile = profile[1:] 
    
    # Beta Slope
    beta = np.nan
    try:
        freqs = np.arange(1, len(profile) + 1)
        limit = len(profile) // 2
        if limit > 2:
            res = stats.linregress(np.log(freqs[:limit]), np.log(profile[:limit] + 1e-10))
            beta = -res.slope
    except:
        beta = np.nan
        
    return profile, beta

def analyze_crop_texture(crop):
    # GLCM
    mi, ma = crop.min(), crop.max()
    if ma == mi: return None
    crop_u8 = ((crop - mi) / (ma - mi) * 255).astype(np.uint8)
    
    g = graycomatrix(crop_u8, [1], [0, np.pi/2], levels=256, symmetric=True, normed=True)
    return {
        'Contrast': graycoprops(g, 'contrast').mean(),
        'Homogeneity': graycoprops(g, 'homogeneity').mean(),
        'Entropy': shannon_entropy(crop_u8),
        'Mean_Intensity': np.mean(crop)
    }

def process_file_full_scan(filepath, channel_map):
    filename = os.path.splitext(os.path.basename(filepath))[0]
    print(f"\n<== Analyzing file: {filename} ==>")
    ensure_dirs()
    stats_data = []
    scan_logs = []
    
    # Keys in Dict: 'Raw', 'BW', 'Z', 'Z+BW'
    spectra_db = {
        'Raw': {}, 'BW': {}, 'Z': {}, 'Z+BW': {}
    }
    try:
        with tifffile.TiffFile(filepath) as tif:
            for i, page in enumerate(tif.pages):
                if i >= MAX_CHANNELS: break
                
                name = channel_map.get(i, f"Ch{i}")
                label = f"{name} (Ch{i})"
                print(f"\n--> Channel {label}...")
                
                img = page.asarray()
                if img.shape[0] < 500: continue
                
                # Scan
                crops, scan_info = scan_tissue_sliding_window(img, CROP_SIZE, STEP_SIZE)
                scan_info['Channel'] = label
                scan_logs.append(scan_info)
                
                del img
                gc.collect()
                
                if not crops:
                    print("    No tissue found in this channel.")
                    continue

                # Crops
                prof_raw, prof_bw, prof_z, prof_zbw = [], [], [], []
                betas_zbw = []
                textures = []

                for crop in tqdm(crops, desc="    Analyzing crops", leave=False):
                    # Texture
                    tex = analyze_crop_texture(crop)
                    if tex: textures.append(tex)
                    
                    # Raw (No Z, No BW)
                    p1, _ = analyze_crop_spectral(crop, use_z=False, use_bw=False)
                    prof_raw.append(p1)

                    # BW Only
                    p2, _ = analyze_crop_spectral(crop, use_z=False, use_bw=True)
                    prof_bw.append(p2)

                    # Z Only
                    p3, _ = analyze_crop_spectral(crop, use_z=True, use_bw=False)
                    prof_z.append(p3)

                    # Z + BW
                    p4, b4 = analyze_crop_spectral(crop, use_z=True, use_bw=True)
                    prof_zbw.append(p4)
                    if not np.isnan(b4): betas_zbw.append(b4)

                if not prof_zbw:
                    continue

                # Calculate Mean & Std Dev
                def calc_stats(plist):
                    if not plist: return None, None
                    ml = min(len(x) for x in plist)
                    arr = np.array([x[:ml] for x in plist])
                    return np.mean(arr, axis=0), np.std(arr, axis=0)

                spectra_db['Raw'][label] = calc_stats(prof_raw)
                spectra_db['BW'][label] = calc_stats(prof_bw)
                spectra_db['Z'][label] = calc_stats(prof_z)
                spectra_db['Z+BW'][label] = calc_stats(prof_zbw)
                
                # Statistics
                if textures:
                    avg_tex = pd.DataFrame(textures).mean()
                    avg_beta = np.mean(betas_zbw) if betas_zbw else np.nan
                    std_beta = np.std(betas_zbw) if betas_zbw else np.nan
                    
                    stats_entry = {
                        'Channel': label,
                        'N_Crops': len(prof_zbw),
                        'Beta_Mean': avg_beta,
                        'Beta_Std': std_beta,
                        'Entropy': avg_tex['Entropy'],
                        'Contrast': avg_tex['Contrast'],
                        'Homogeneity': avg_tex['Homogeneity'],
                        'Mean_Intensity': avg_tex['Mean_Intensity']
                    }
                    stats_data.append(stats_entry)
                
    except Exception as e:
        print(f"Critical error: {e}")
        return

    if scan_logs:
        df_scan = pd.DataFrame(scan_logs)
        cols = ['Channel'] + [c for c in df_scan.columns if c != 'Channel']
        df_scan = df_scan[cols]
        log_path = os.path.join(OUTPUT_DIR, f"{filename}_scan_logs.csv")
        df_scan.to_csv(log_path, index=False)
        print(f"\nScan logs saved: {log_path}")

    if stats_data:
        df_stats = pd.DataFrame(stats_data)
        print("\n<== Statistics Summary (Z-Score + BW) ==>")
        print(df_stats.to_string(index=False, float_format="%.3f"))
        txt_path = os.path.join(OUTPUT_DIR, f"{filename}_full_stats.csv")
        df_stats.to_csv(txt_path, index=False)
    
    # Plotting 2x2 Grid with variance
    has_data = any(v[0] is not None for v in spectra_db['Z+BW'].values())

    if has_data:
        _, axes = plt.subplots(2, 2, figsize=(16, 12))
        scenarios = [
            ('Raw', axes[0, 0], 'No Z-score; No Butterworth'),
            ('BW', axes[0, 1], 'No Z-score; With Butterworth'),
            ('Z', axes[1, 0], 'With Z-Score, No Butterworth'),
            ('Z+BW', axes[1, 1], 'With Z-Score, With Butterworth')
        ]
        
        # Determine colors
        found_labels = list(spectra_db['Z+BW'].keys())
        colors = plt.cm.tab10(np.linspace(0, 1, len(found_labels)))
        color_map = dict(zip(found_labels, colors))

        for key, ax, title in scenarios:
            ax.set_title(title, weight='bold')
            ax.set_xlabel("Frequency")
            ax.set_ylabel("Power")
            ax.grid(alpha=0.3)
            
            # Plot each channel
            for label, (mean_prof, std_prof) in spectra_db[key].items():
                if mean_prof is not None:
                    x = np.arange(1, len(mean_prof)+1)
                    c = color_map.get(label, 'black')
                    
                    # Line
                    ax.loglog(x, mean_prof, label=label, color=c, lw=1.5)
                    
                    # Variance (Shading)
                    lower = np.maximum(mean_prof - std_prof, 1e-10)
                    upper = mean_prof + std_prof
                    ax.fill_between(x, lower, upper, color=c, alpha=0.15)
            
            if key == 'Raw':
                ax.legend(fontsize='small', loc='lower left')

        plt.suptitle(f"Spectral Analysis Scenarios: {filename}", fontsize=16)
        plt.tight_layout()
        plot_path = os.path.join(OUTPUT_DIR, f"{filename}_spectrum_2x2.png")
        plt.savefig(plot_path, dpi=300)
        print(f"Plot saved: {plot_path}")


def main():
    channel_map = load_channel_names(DATA_DIR)
    files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('qptiff')]
    
    if not files:
        print("No qptiff files found.")
        return None

    for f in files:
        process_file_full_scan(f, channel_map)

if __name__ == "__main__":
    main()
