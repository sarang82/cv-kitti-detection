import os
import shutil
from pathlib import Path
import random

# =====================
# 경로 설정
# =====================
RAW_IMG  = Path("/home/ubuntu/kitti-yolo/raw_data/data_object_image_2/training/image_2")
RAW_LBL  = Path("/home/ubuntu/kitti-yolo/raw_data/data_object_label_2/training/label_2")
OUT_BASE = Path("/home/ubuntu/kitti-yolo/data")

# =====================
# 우리가 쓸 3개 클래스만 정의
# YOLO는 클래스를 숫자로 관리: Car=0, Pedestrian=1, Cyclist=2
# =====================
CLASSES = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}

# =====================
# train/val 분리 비율 (8:2)
# =====================
TRAIN_RATIO = 0.8
random.seed(42)  # 재현성을 위해 seed 고정

def kitti_to_yolo_bbox(x1, y1, x2, y2, img_w, img_h):
    """
    KITTI 포맷 (x1,y1,x2,y2 절대 픽셀좌표)
    → YOLO 포맷 (cx,cy,w,h 정규화 0~1)
    """
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    w  = (x2 - x1) / img_w
    h  = (y2 - y1) / img_h
    return cx, cy, w, h

def convert():
    # 출력 폴더 생성
    for split in ["train", "val"]:
        (OUT_BASE / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_BASE / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 전체 이미지 파일 목록
    img_files = sorted(RAW_IMG.glob("*.png"))
    random.shuffle(img_files)

    # train/val 분리
    split_idx  = int(len(img_files) * TRAIN_RATIO)
    train_imgs = img_files[:split_idx]
    val_imgs   = img_files[split_idx:]

    total, skipped = 0, 0

    for split, files in [("train", train_imgs), ("val", val_imgs)]:
        for img_path in files:
            lbl_path = RAW_LBL / img_path.with_suffix(".txt").name

            if not lbl_path.exists():
                skipped += 1
                continue

            yolo_lines = []
            # 이미지 크기 (KITTI는 고정 1242×375)
            img_w, img_h = 1242, 375

            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    cls_name = parts[0]

                    # Car, Pedestrian, Cyclist 외 클래스는 무시
                    if cls_name not in CLASSES:
                        continue

                    cls_id = CLASSES[cls_name]
                    x1, y1, x2, y2 = map(float, parts[4:8])
                    cx, cy, w, h = kitti_to_yolo_bbox(x1, y1, x2, y2, img_w, img_h)
                    yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            # YOLO 라벨 저장
            out_lbl = OUT_BASE / "labels" / split / lbl_path.name
            with open(out_lbl, "w") as f:
                f.write("\n".join(yolo_lines))

            # 이미지 복사
            shutil.copy(img_path, OUT_BASE / "images" / split / img_path.name)
            total += 1

    print(f"✅ 변환 완료: {total}장 처리, {skipped}장 스킵")
    print(f"   train: {len(train_imgs)}장 / val: {len(val_imgs)}장")

if __name__ == "__main__":
    convert()
