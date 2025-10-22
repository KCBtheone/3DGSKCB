import cv2
import numpy as np
import os
from tqdm import tqdm
import glob

def calculate_average_normal_gradient(image_paths: list) -> float:
    """
    Calculates the average spatial gradient magnitude across a list of normal map images.
    """
    if not image_paths:
        return -1.0

    total_magnitude_avg = 0
    valid_images = 0

    image_iterator = tqdm(image_paths, desc="  Processing images", leave=False, ncols=100)
    for img_path in image_iterator:
        # Read image in color (normals are encoded in RGB)
        normal_map = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if normal_map is None:
            continue

        # Convert to float for gradient calculation
        normal_map_float = normal_map.astype(np.float64) / 255.0
        
        # Calculate gradients in X and Y directions for each channel
        grad_x = cv2.Sobel(normal_map_float, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(normal_map_float, cv2.CV_64F, 0, 1, ksize=3)

        # Calculate the magnitude of the gradient for each pixel
        # Summing the squared gradients across the color channels
        magnitude = np.sqrt(np.sum(grad_x**2 + grad_y**2, axis=2))
        
        # Average magnitude for the current image
        avg_mag = np.mean(magnitude)
        
        if not np.isnan(avg_mag):
            total_magnitude_avg += avg_mag
            valid_images += 1

    if valid_images == 0:
        return -1.0
        
    # Average across all images
    return total_magnitude_avg / valid_images


def generate_report(title: str, results: dict, best_alphas: dict):
    """
    Generates and prints a formatted analysis report.
    """
    if not results:
        print(f"No results to generate report for: {title}")
        return

    sorted_results = sorted(results.items(), key=lambda item: item[1])
    
    max_scene_len = max(len(s) for s in results.keys()) + 2

    print("\n\n" + "="*95)
    print(f"      {title.center(85)}")
    print("="*95)
    print(f"{'Complexity Rank':<18} {'Scene':<{max_scene_len}} {'C_grad Score':<20} {'Actual Best α':<18} {'Correlation Check'}")
    print(f"{'---------------':<18} {'-'* (max_scene_len-1):<{max_scene_len}} {'--------------':<20} {'---------------':<18} {'-----------------'}")

    # For correlation, let's assume higher rank should correspond to lower alpha (negative correlation)
    # We'll check if the top half of scenes have lower alphas than the bottom half on average
    
    for i, (scene, score) in enumerate(sorted_results):
        rank = i + 1
        actual_alpha = best_alphas.get(scene, 'N/A')

        # Simple check: does complexity rank align with alpha magnitude?
        # Low rank (simple geometry) should ideally have high alpha.
        # High rank (complex geometry) should ideally have low alpha.
        correlation = ""
        if isinstance(actual_alpha, float):
            if rank <= 4 and actual_alpha >= 0.30:
                correlation = "✅ Matches (Simple -> High α)"
            elif rank > 4 and actual_alpha <= 0.20:
                correlation = "✅ Matches (Complex -> Low α)"
            elif rank <=4 and actual_alpha <=0.2:
                correlation = "❌ MISMATCH (Simple -> Low α)"
            elif rank > 4 and actual_alpha >= 0.3:
                correlation = "❌ MISMATCH (Complex -> High α)"
        
        print(f"{rank:<18} {scene:<{max_scene_len}} {score:<20.6f} {actual_alpha:<18.2f} {correlation}")
        
    print("="*95)


def main():
    """
    Main function to run the analysis on all specified scenes and normal types.
    """
    # --- Configuration ---
    base_path = "/root/autodl-tmp/gaussian-splatting/data"
    scenes = [
        "electro", "delivery_area", "pipes", "courtyard", 
        "facade", "kicker", "meadow", "office"
    ]
    
    # Your experimentally found best alpha values
    best_alphas = {
        "pipes": 0.20,
        "office": 0.20,
        "delivery_area": 0.40,
        "kicker": 0.10,
        "electro": 0.30,
        "meadow": 0.10,
        "courtyard": 0.40,
        "facade": 0.40
    }
    
    # Directories for the two types of normals
    normal_dirs = {
        "gt_normals": "gt_normals",             # Ground Truth
        "predicted_normals": "geometry_priors"  # COLMAP-predicted (OmniData)
    }
    
    print("Starting Scene Geometric Complexity Analysis using Normal Map Gradients...")
    
    # --- Data Collection Loop ---
    all_results = {}
    for normal_type, dir_name in normal_dirs.items():
        print(f"\n--- Analyzing {normal_type.replace('_', ' ').title()} ---")
        scene_results = {}
        for scene in scenes:
            print(f"Processing scene: {scene}...")
            
            # Use glob to find all normal maps, accommodating different extensions
            image_dir = os.path.join(base_path, scene, dir_name)
            # Find files ending with _normal.png or _normal.jpg
            search_pattern = os.path.join(image_dir, f"*_normal.*") 
            image_paths = glob.glob(search_pattern)

            if not image_paths:
                print(f"  [Warning] No normal maps found for '{scene}' in '{image_dir}'")
                continue
            
            print(f"  Found {len(image_paths)} images to process.")
            
            c_grad = calculate_average_normal_gradient(image_paths)
            
            if c_grad != -1.0:
                scene_results[scene] = c_grad
                print(f"  => Calculated C_grad = {c_grad:.6f}")
        
        all_results[normal_type] = scene_results

    # --- Final Report Generation ---
    if all_results.get("gt_normals"):
        generate_report(
            "Analysis based on GROUND TRUTH Normals", 
            all_results["gt_normals"], 
            best_alphas
        )
    
    if all_results.get("predicted_normals"):
        generate_report(
            "Analysis based on PREDICTED (COLMAP/OmniData) Normals", 
            all_results["predicted_normals"], 
            best_alphas
        )

    print("\n\n[Final Analysis Guide]")
    print("Compare the 'Correlation Check' columns for both reports.")
    print("The report with more '✅ Matches' indicates the superior predictive metric for determining alpha.")
    print("A perfect metric would show a clean negative correlation: rank increases as alpha decreases.")


if __name__ == "__main__":
    main()