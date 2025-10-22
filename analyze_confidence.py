import os
import sys
import cv2
import numpy as np
import glob
from tqdm import tqdm

def analyze_confidence_maps(scene_path):
    """
    Analyzes only confidence maps in a given scene directory, checking for both 'confidence'
    and 'geometry_priors' subdirectories, and prints detailed statistics.
    """
    
    possible_dirs = [
        os.path.join(scene_path, "confidence"),
        os.path.join(scene_path, "geometry_priors")
    ]
    
    confidence_dir = None
    for directory in possible_dirs:
        if os.path.isdir(directory):
            confidence_dir = directory
            break

    print("="*80)
    print(f"📊 Analyzing CONFIDENCE Maps for Scene: '{os.path.basename(scene_path)}'")
    
    if confidence_dir is None:
        print("❌ ERROR: Could not find a confidence map directory.")
        print("   Searched for the following folders but none exist:")
        for directory in possible_dirs:
            print(f"     - {directory}")
        return
        
    print(f" searching in: {confidence_dir}")
    print("="*80)

    # --- [ KEY MODIFICATION ] ---
    # Modify search patterns to ONLY match files containing "_confidence"
    search_patterns = [
        os.path.join(confidence_dir, '*_confidence.[pP][nN][gG]'),
        os.path.join(confidence_dir, '*_confidence.[jJ][pP][gG]'),
        os.path.join(confidence_dir, '*_confidence.[jJ][pP][eE][gG]')
    ]
    # --- [ END MODIFICATION ] ---
    
    map_files = []
    for pattern in search_patterns:
        map_files.extend(glob.glob(pattern))

    if not map_files:
        print("❌ WARNING: No confidence maps (e.g., '*_confidence.png') found in the directory.")
        return

    print(f"✅ Found {len(map_files)} confidence maps to analyze.\n")

    stats_per_map = []
    all_means = []
    all_stds = []
    all_low_conf_pcts = []

    for map_path in tqdm(map_files, desc="Processing confidence maps"):
        img = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"   -> Skipping corrupted file: {os.path.basename(map_path)}")
            continue

        img_normalized = img.astype(np.float32) / 255.0

        mean_val = np.mean(img_normalized)
        std_val = np.std(img_normalized)
        min_val = np.min(img_normalized)
        max_val = np.max(img_normalized)
        
        low_confidence_pixels = np.sum(img_normalized < 0.1)
        total_pixels = img_normalized.size
        low_confidence_percentage = (low_confidence_pixels / total_pixels) * 100

        hist, bin_edges = np.histogram(img_normalized, bins=10, range=(0.0, 1.0))
        hist_percentage = (hist / total_pixels) * 100

        stats = {
            "filename": os.path.basename(map_path),
            "min": min_val, "max": max_val, "mean": mean_val, "std": std_val,
            "low_conf_pct": low_confidence_percentage, "hist_pct": hist_percentage
        }
        stats_per_map.append(stats)
        all_means.append(mean_val)
        all_stds.append(std_val)
        all_low_conf_pcts.append(low_confidence_percentage)

    print("\n--- [ Individual Confidence Map Analysis ] ---\n")
    stats_per_map.sort(key=lambda x: x['filename'])
    for stats in stats_per_map:
        print(f"📄 File: {stats['filename']}")
        print(f"   - Stats: Min={stats['min']:.4f}, Max={stats['max']:.4f}, Mean={stats['mean']:.4f}, Std Dev={stats['std']:.4f}")
        print(f"   - Low Confidence (< 0.1): {stats['low_conf_pct']:.2f}% of pixels")
        print(f"   - Distribution Histogram:")
        for i, pct in enumerate(stats['hist_pct']):
            bar = '█' * int(pct / 2)
            print(f"     [{(i)*10:02d}%-{(i+1)*10:02d}%]: {pct:5.2f}%  {bar}")
        print("-" * 40)

    print("\n" + "="*80)
    print("📈 Overall Summary Statistics for CONFIDENCE Maps")
    print("="*80)
    if all_means:
        overall_mean = np.mean(all_means)
        mean_of_stds = np.mean(all_stds)
        overall_low_conf = np.mean(all_low_conf_pcts)
        print(f"Across all {len(map_files)} maps:")
        print(f"  - Average of Mean Confidence Values: {overall_mean:.4f}")
        print(f"  - Average of Standard Deviations:    {mean_of_stds:.4f}")
        print(f"  - Average % of Low Confidence Pixels: {overall_low_conf:.2f}%")
        
        if overall_mean > 0.95:
            print("\n   [💡 Interpretation]: The overall mean confidence is very high (GOOD).")
            print("   This suggests the maps provide only weak differential weighting.")
        elif overall_mean < 0.5:
            print("\n   [💡 Interpretation]: The overall mean confidence is low (POTENTIALLY BAD).")
            print("   This suggests large portions of the images are being significantly down-weighted, potentially harming training.")
        else:
            print("\n   [💡 Interpretation]: The confidence values seem reasonably distributed (GOOD).")
            print("   This suggests the maps are providing a potentially useful and varied signal for training.")

    print("\nAnalysis complete.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_confidence.py <scene_name>")
        sys.exit(1)

    scene_name = sys.argv[1]
    project_root = os.path.dirname(os.path.abspath(__file__))
    target_scene_path = os.path.join(project_root, 'data', scene_name)
    
    analyze_confidence_maps(target_scene_path)