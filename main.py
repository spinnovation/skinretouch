import cv2
import numpy as np
import sys
import os
import argparse
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def smooth_skin(roi, p=50, value1=3, value2=1):
    """
    The core High-Pass / Bilateral filter logic for skin smoothing.
    """
    dx = value1 * 5
    fc = value1 * 12.5

    # 1. Bilateral Filter
    temp1 = cv2.bilateralFilter(roi, dx, fc, fc)
    
    # 2. Extract Details (High-Pass)
    temp2 = temp1.astype(np.float32) - roi.astype(np.float32) + 128.0
    
    # 3. Gaussian Blur on Details
    ksize = 2 * value2 - 1
    temp3 = cv2.GaussianBlur(temp2, (ksize, ksize), 0, 0)
    
    # 4. Detail Re-application
    temp4 = roi.astype(np.float32) + 2.0 * temp3 - 255.0
    
    # 5. Alpha Blending
    dst = (roi.astype(np.float32) * (100 - p) + temp4 * p) / 100.0
    dst = np.clip(dst, 0, 255).astype(np.uint8)
    
    return dst

def remove_blemishes(img, skin_mask):
    """
    Removal of blemishes, redness, and spots, while utilizing frequency separation to keep pores later.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    
    # 1. Dark spots (blemishes/freckles) via Black-Hat transform
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    blackhat = cv2.morphologyEx(l_channel, cv2.MORPH_BLACKHAT, kernel)
    # Lower threshold to properly catch and erase skin blemishes and spots
    _, blemish_mask_dark = cv2.threshold(blackhat, 20, 255, cv2.THRESH_BINARY)

    # 2. Red spots (acne/redness) via A-channel local contrast
    a_blur = cv2.GaussianBlur(a, (21, 21), 0)
    red_spikes = cv2.subtract(a, a_blur)
    _, blemish_mask_red = cv2.threshold(red_spikes, 10, 255, cv2.THRESH_BINARY)
    
    # Combine masks
    blemish_mask = cv2.bitwise_or(blemish_mask_dark, blemish_mask_red)
    blemish_mask = cv2.bitwise_and(blemish_mask, skin_mask)
    
    # Cleanup mask
    tiny_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    blemish_mask = cv2.erode(blemish_mask, tiny_kernel, iterations=1)
    blemish_mask = cv2.dilate(blemish_mask, dilate_kernel, iterations=2)
    
    healed_img = img.copy()
    if cv2.countNonZero(blemish_mask) > 0:
        healed_img = cv2.inpaint(img, blemish_mask, 2, cv2.INPAINT_TELEA)
        
    return healed_img

def correct_skin_tone(img, skin_mask):
    """
    Subtle correction of dull skin tone and enhancing natural light and shadow dimensionality.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    
    # Apply CLAHE to Lightness channel for better light and shadow dimensionality
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8,8))
    l_clahe = clahe.apply(l_channel)
    
    # Merge back
    enhanced_lab = cv2.merge((l_clahe, a, b))
    enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    # subtly blend enhanced tone only on the skin (20% opacity for natural look)
    mask_3d = cv2.cvtColor(skin_mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
    final_img = (img.astype(np.float32) * (1.0 - (mask_3d * 0.2))) + (enhanced_img.astype(np.float32) * (mask_3d * 0.2))
    
    return np.clip(final_img, 0, 255).astype(np.uint8)

def process_image(img_path, output_path):
    print(f"[*] Processing {img_path} ...")
    image = cv2.imread(img_path)
    if image is None:
        print(f"[!] Error: Could not load image {img_path}")
        return False

    # Initialize MediaPipe Face Landmarker
    base_options = python.BaseOptions(model_asset_path="face_landmarker.task")
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=10,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
    )

    try:
        detector = vision.FaceLandmarker.create_from_options(options)
    except Exception as e:
        print(f"[!] Missing or invalid face_landmarker.task file! Please download it.")
        print(e)
        return False

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    detection_result = detector.detect(mp_image)

    if not detection_result.face_landmarks:
        print("[-] No face detected. Applying global smoothing...")
        final_image = smooth_skin(image.copy(), p=30)
    else:
        print(f"[+] Detected {len(detection_result.face_landmarks)} face(s). Generating precision masks...")
        
        # Create a global zero mask for all faces
        global_skin_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        for face_landmarks in detection_result.face_landmarks:
            points = []
            for landmark in face_landmarks:
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                points.append((x, y))
            
            points = np.array(points, dtype=np.int32)
            
            # Create convex hull for the face to isolate the skin from the background/hair
            hull = cv2.convexHull(points)
            face_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(face_mask, hull, 255)
            
            # Feather the edges of the mask slightly for natural blending
            face_mask = cv2.GaussianBlur(face_mask, (15, 15), 0)
            
            global_skin_mask = cv2.add(global_skin_mask, face_mask)

        # 1. Subtle Blemish/Redness Correction (Preserving moles and stubble)
        print("[+] Subtle correction of redness and blemishes...")
        healed_image = remove_blemishes(image.copy(), global_skin_mask)
        
        # 2. Tone/Dimensionality Correction
        print("[+] Enhancing tone, light, and shadow dimensionality...")
        toned_image = correct_skin_tone(healed_image, global_skin_mask)

        # 3. Apply High-Resolution Natural Skin Smoothing
        # Lowering parameter value1 to 2 to narrow blur radius, keeping pores crisp.
        print("[+] Applying Natural Frequency-Separation Skin Balancing...")
        smoothed_img = smooth_skin(toned_image, p=45, value1=2) 
        
        # 4. Merge the optimally tuned image with the original utilizing the exact mask
        skin_mask_3d = cv2.cvtColor(global_skin_mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
        
        final_image = (image.astype(np.float32) * (1.0 - skin_mask_3d)) + (smoothed_img.astype(np.float32) * skin_mask_3d)
        final_image = np.clip(final_image, 0, 255).astype(np.uint8)

    cv2.imwrite(output_path, final_image)
    print(f"[*] Processing complete. Saved to '{output_path}'.")
    return True

def main():
    parser = argparse.ArgumentParser(description="AI Face Skin Retouching Tool")
    parser.add_argument("input", nargs="?", default="1.jpeg", help="Input image file path or prefix")
    parser.add_argument("--output", "-o", default="result.png", help="Output image file path")
    
    args = parser.parse_args()
    
    img_path = None
    arg = args.input
    
    if os.path.exists(arg):
        img_path = arg
    else:
        extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
        for ext in extensions:
            if os.path.exists(arg + ext):
                img_path = arg + ext
                break

    if not img_path:
        print(f"[!] Error: Could not find image file matching '{arg}'")
        sys.exit(1)

    process_image(img_path, args.output)

if __name__ == "__main__":
    main()
