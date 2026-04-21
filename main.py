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
    피부를 부드럽게 만드는 핵심 로직 (하이패스 / 쌍방향 필터 활용)
    """
    dx = value1 * 5
    fc = value1 * 12.5

    # 1. 쌍방향 필터 (Bilateral Filter) 적용
    temp1 = cv2.bilateralFilter(roi, dx, fc, fc)
    
    # 2. 질감(디테일) 추출 (하이패스 필터링)
    temp2 = temp1.astype(np.float32) - roi.astype(np.float32) + 128.0
    
    # 3. 추출된 질감에 가우시안 블러 적용
    ksize = 2 * value2 - 1
    temp3 = cv2.GaussianBlur(temp2, (ksize, ksize), 0, 0)
    
    # 4. 부드러워진 피부에 질감 다시 덧입히기
    temp4 = roi.astype(np.float32) + 2.0 * temp3 - 255.0
    
    # 5. 알파 블렌딩 (원본과 스무딩된 이미지 합성)
    dst = (roi.astype(np.float32) * (100 - p) + temp4 * p) / 100.0
    dst = np.clip(dst, 0, 255).astype(np.uint8)
    
    return dst

def remove_blemishes(img, skin_mask):
    """
    모공은 유지하면서 잡티, 붉은기, 점 등을 제거
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    
    # 1. 어두운 점 (잡티/주근깨) 잡기: 블랙햇 (Black-Hat) 변환 활용
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (100, 100))
    blackhat = cv2.morphologyEx(l_channel, cv2.MORPH_BLACKHAT, kernel)
    # 피부 잡티와 점을 정확하게 포착하기 위해 임곗값을 조정
    _, blemish_mask_dark = cv2.threshold(blackhat, 1, 255, cv2.THRESH_BINARY)

    # 2. 붉은 점 (여드름/홍조) 잡기: A-채널의 부분적 대비(Contrast) 활용
    a_blur = cv2.GaussianBlur(a, (100, 100), 0)
    red_spikes = cv2.subtract(a, a_blur)
    _, blemish_mask_red = cv2.threshold(red_spikes, 10, 255, cv2.THRESH_BINARY)
    
    # 잡티 마스크 병합
    blemish_mask = cv2.bitwise_or(blemish_mask_dark, blemish_mask_red)
    blemish_mask = cv2.bitwise_and(blemish_mask, skin_mask)
    
    # 마스크 클린업 작업 (노이즈 제거 및 잡티 영역 확장)
    tiny_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    blemish_mask = cv2.erode(blemish_mask, tiny_kernel, iterations=1)
    blemish_mask = cv2.dilate(blemish_mask, dilate_kernel, iterations=2)
    
    healed_img = img.copy()
    if cv2.countNonZero(blemish_mask) > 0:
        # 감지된 잡티 부분을 주변 피부 픽셀로 감쪽같이 채워넣기(Inpainting)
        healed_img = cv2.inpaint(img, blemish_mask, 10, cv2.INPAINT_TELEA)
        
    return healed_img, blemish_mask

def correct_skin_tone(img, skin_mask):
    """
    자연스러운 빛과 그림자의 입체감을 살려주면서 칙칙한 피부 톤 미세 보정
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    
    # 명암 대비를 주어 빛과 그림자 입체감을 살리기 위해 CLAHE 적용
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8,8))
    l_clahe = clahe.apply(l_channel)
    
    # 채널 다시 합치기
    enhanced_lab = cv2.merge((l_clahe, a, b))
    enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    # 피부 부분에만 강화된 톤을 부드럽게 합성 (자연스러움을 위해 20% 불투명도 적용)
    mask_3d = cv2.cvtColor(skin_mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
    final_img = (img.astype(np.float32) * (1.0 - (mask_3d * 0.5))) + (enhanced_img.astype(np.float32) * (mask_3d * 0.2))
    
    return np.clip(final_img, 0, 255).astype(np.uint8)

def process_image(img_path, output_path):
    print(f"[*] 처리 중: {img_path} ...")
    image = cv2.imread(img_path)
    if image is None:
        print(f"[!] 에러: 이미지를 불러올 수 없습니다. 경로: {img_path}")
        return False

    # MediaPipe Face Landmarker 초기화 작업
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
        print(f"[!] face_landmarker.task 파일이 없거나 잘못되었습니다! 모델 파일을 다운로드 해주세요.")
        print(e)
        return False

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    detection_result = detector.detect(mp_image)

    if not detection_result.face_landmarks:
        print("[-] 얼굴이 감지되지 않았습니다. 전체 이미지 스무딩을 적용합니다...")
        final_image = smooth_skin(image.copy(), p=30)
    else:
        print(f"[+] {len(detection_result.face_landmarks)}개의 얼굴이 감지되었습니다. 정밀 스킨 마스크를 생성합니다...")
        
        # 모든 얼굴을 담을 전체 피부 마스크용 빈 캔버스 생성
        global_skin_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        for face_landmarks in detection_result.face_landmarks:
            points = []
            for landmark in face_landmarks:
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                points.append((x, y))
            
            points = np.array(points, dtype=np.int32)
            
            # 얼굴 윤곽선을 따라 다각형 영역(Convex Hull)을 그려 피부와 배경/머리카락 분리
            hull = cv2.convexHull(points)
            face_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(face_mask, hull, 255)

            # --- 이목구비(눈, 눈썹, 입술) 보호를 위해 마스크에서 제외 ---
            LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
            RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
            LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40]
            LEFT_EYEBROW = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]
            RIGHT_EYEBROW = [285, 295, 282, 283, 276, 300, 293, 334, 296, 336]
            
            features_to_exclude = [LEFT_EYE, RIGHT_EYE, LIPS, LEFT_EYEBROW, RIGHT_EYEBROW]
            for feature_indices in features_to_exclude:
                feature_points = [points[idx] for idx in feature_indices]
                feature_polygon = np.array(feature_points, dtype=np.int32)
                # 약간 여유를 두어 이목구비 외곽까지 안전하게 보호하기 위해 볼록 다각형으로 구멍 뚫기
                feature_hull = cv2.convexHull(feature_polygon)
                cv2.fillConvexPoly(face_mask, feature_hull, 0)
            
            # 윤곽선이 자연스럽게 블렌딩 되도록 마스크 가장자리에 가우시안 블러 적용
            face_mask = cv2.GaussianBlur(face_mask, (15, 15), 0)
            
            global_skin_mask = cv2.add(global_skin_mask, face_mask)

        # 1. 자연스러운 잡티 및 붉은기 교정 (수염 자국과 기미 보존형)
        print("[+] 붉은기 및 국소 잡티를 부드럽게 지우는 중입니다...")
        healed_image, blemish_mask = remove_blemishes(image.copy(), global_skin_mask)
        
        # 2. 피부 톤 및 입체감 교정
        print("[+] 피부 톤과 빛, 그림자의 입체감을 강화하는 중입니다...")
        toned_image = correct_skin_tone(healed_image, global_skin_mask)

        # 3. 고해상도 내추럴 피부 스무딩 적용
        # value1 값을 2로 낮추어 블러 반경을 좁게 유지해 피부 모공의 해상도를 선명하게 살림.
        print("[+] 피부 주파수 분리(Frequency-Separation) 밸런싱을 적용하는 중입니다...")
        smoothed_img = smooth_skin(toned_image, p=45, value1=2) 
        
        # 4. 원본 이미지의 디테일에 최적화 튜닝된 피부 마스크를 기반으로 자연스러운 병합
        skin_mask_3d = cv2.cvtColor(global_skin_mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
        
        final_image = (image.astype(np.float32) * (1.0 - skin_mask_3d)) + (smoothed_img.astype(np.float32) * skin_mask_3d)
        final_image = np.clip(final_image, 0, 255).astype(np.uint8)

        # 사용자가 어디를 힐링(Inpainting) 했는지 볼 수 있도록 인페인팅 영역을 붉은색으로 칠하기
        if cv2.countNonZero(blemish_mask) > 0:
            final_image[blemish_mask > 0] = [0, 0, 255]

    cv2.imwrite(output_path, final_image)
    print(f"[*] 처리 완료! 다음 경로에 저장되었습니다: '{output_path}'")
    return True

def main():
    parser = argparse.ArgumentParser(description="AI 얼굴 피부 보정 툴 (AI Face Skin Retouching Tool)")
    parser.add_argument("input", nargs="?", default="1.jpeg", help="입력할 이미지 파일 경로 또는 이름")
    parser.add_argument("--output", "-o", default="result.png", help="출력될 이미지 파일 경로")
    
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
        print(f"[!] 에러: 일치하는 이미지 파일을 찾을 수 없습니다: '{arg}'")
        sys.exit(1)

    process_image(img_path, args.output)

if __name__ == "__main__":
    main()
