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
    Remove moles, acne, and spots using Morphology and Inpainting inside the skin mask.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    
    # 1. Dark spots (moles/freckles) via Black-Hat transform
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    blackhat = cv2.morphologyEx(l_channel, cv2.MORPH_BLACKHAT, kernel)
    _, blemish_mask_dark = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    
    # 2. Red spots (acne) via A-channel local contrast
    a_blur = cv2.GaussianBlur(a, (15, 15), 0)
    red_spikes = cv2.subtract(a, a_blur)
    _, blemish_mask_red = cv2.threshold(red_spikes, 10, 255, cv2.THRESH_BINARY)
    
    # 3. Combine and restrict
    blemish_mask = cv2.bitwise_or(blemish_mask_dark, blemish_mask_red)
    blemish_mask = cv2.bitwise_and(blemish_mask, skin_mask)
    
    # 4. Cleanup and dilate mask to ensure full coverage of the spot
    tiny_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    blemish_mask = cv2.erode(blemish_mask, tiny_kernel, iterations=1)
    blemish_mask = cv2.dilate(blemish_mask, dilate_kernel, iterations=2)
    
    if cv2.countNonZero(blemish_mask) > 0:
        return cv2.inpaint(img, blemish_mask, 3, cv2.INPAINT_TELEA)
    return img

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
        final_image = smooth_skin(image.copy(), p=50)
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

        # 1. Blemish Removal
        print("[+] Erasing moles, spots, and acne via Auto-Inpainting...")
        healed_image = remove_blemishes(image.copy(), global_skin_mask)

        # 2. Apply the smoothing logic on the healed image
        print("[+] Applying Frequency-Separation Skin Smoothing...")
        smoothed_img = smooth_skin(healed_image, p=60)
        
        # 3. Merge the smoothed image and the original image based on the skin mask
        # Note: We merge with `healed_image` as the base so that blemishes remain gone, 
        # but the non-skin parts revert to original unaltered pixels!
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
