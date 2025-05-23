import os
import json
import cv2
import csv

# ✅ 경로 설정
IMAGE_PATH = r"C:\Users\wecha\project\yolov5\custom_dataset\images\train\Sanggyeonggwan_c1_1.jpeg"
YOLO_LABEL_PATH = r"C:\Users\wecha\project\yolov5\runs\detect\exp34\labels\Sanggyeonggwan_c1_1.txt"
ROI_JSON_PATH = r"C:\Users\wecha\project\yolov5\roi_slots_final_merged.json"
ROI_KEY = "Sanggyeonggwan_c1_1.jpeg"

# ✅ 출력 경로
OUTPUT_CSV_PATH = "occupancy_result.csv"
OUTPUT_IMAGE_PATH = "result_with_roi.jpg"

# ✅ 이미지 열기
image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {IMAGE_PATH}")
img_h, img_w = image.shape[:2]

# ✅ ROI JSON 로드
with open(ROI_JSON_PATH, 'r') as f:
    roi_data = json.load(f)

if ROI_KEY not in roi_data:
    raise KeyError(f"ROI JSON에 '{ROI_KEY}' 키가 없습니다.")
roi_list = roi_data[ROI_KEY]

# ✅ YOLO 결과 로드 (bbox 픽셀 변환)
bboxes = []
with open(YOLO_LABEL_PATH, 'r') as f:
    for line in f.readlines():
        parts = list(map(float, line.strip().split()))
        cls, xc, yc, w, h = parts[:5]
        x_center = xc * img_w
        y_center = yc * img_h
        width = w * img_w
        height = h * img_h
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        bboxes.append((x1, y1, x2, y2, x_center, y_center))

# ✅ Occupied / Free 판단 (중심점 기준)
results = []
for idx, roi in enumerate(roi_list):
    rx1, ry1, rx2, ry2 = roi
    status = 'free'
    for bbox in bboxes:
        _, _, _, _, cx, cy = bbox
        if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
            status = 'occupied'
            break
    results.append((f'slot_{idx+1}', status))

# ✅ CSV 저장
with open(OUTPUT_CSV_PATH, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['slot_id', 'status'])
    for slot_id, status in results:
        writer.writerow([slot_id, status])

# ✅ 이미지 시각화
for idx, roi in enumerate(roi_list):
    x1, y1, x2, y2 = map(int, roi)
    slot_id, status = results[idx]
    color = (0, 255, 0) if status == 'free' else (0, 0, 255)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, slot_id, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

cv2.imwrite(OUTPUT_IMAGE_PATH, image)
print(f"✅ 완료: {OUTPUT_CSV_PATH}, {OUTPUT_IMAGE_PATH}")
