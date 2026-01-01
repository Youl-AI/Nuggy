import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

# ---------------------------------------------------------
# ️ 설정 상수
# ---------------------------------------------------------
IMG_SIZE = 1024
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------------------------------------------------------
# ️ 핵심 알고리즘 함수들
# ---------------------------------------------------------

def guided_filter(I, p, r, eps):
    """
    OpenCV 가이디드 필터
    """
    ksize = (2 * r + 1, 2 * r + 1)
    mean_I = cv2.boxFilter(I, cv2.CV_32F, ksize)
    mean_p = cv2.boxFilter(p, cv2.CV_32F, ksize)
    mean_Ip = cv2.boxFilter(I * p, cv2.CV_32F, ksize)
    mean_II = cv2.boxFilter(I * I, cv2.CV_32F, ksize)

    cov_Ip = mean_Ip - mean_I * mean_p
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_32F, ksize)
    mean_b = cv2.boxFilter(b, cv2.CV_32F, ksize)

    q = mean_a * I + mean_b
    return q

def run_inference(model, image, cutoff, gamma, guided_r, guided_eps, min_area_ratio, use_tta=True):
    """
    모델 추론 및 후처리
    """
    orig_w, orig_h = image.size
    
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    
    # 1. TTA (Test Time Augmentation) 추론
    with torch.no_grad():
        img_tensor = transform(image).unsqueeze(0).to(DEVICE)
        preds_1 = model(img_tensor)
        while isinstance(preds_1, (list, tuple)): preds_1 = preds_1[0]
        final_mask = torch.sigmoid(preds_1)

        if use_tta:
            img_flip = image.transpose(Image.FLIP_LEFT_RIGHT)
            img_flip_tensor = transform(img_flip).unsqueeze(0).to(DEVICE)
            preds_2 = model(img_flip_tensor)
            while isinstance(preds_2, (list, tuple)): preds_2 = preds_2[0]
            mask_2 = torch.sigmoid(preds_2)
            mask_2 = torch.flip(mask_2, dims=[3]) 
            final_mask = (final_mask + mask_2) / 2.0

    # 텐서 -> 넘파이 변환
    pred_mask = final_mask.squeeze().cpu().numpy()
    if pred_mask.max() != pred_mask.min():
        pred_mask = (pred_mask - pred_mask.min()) / (pred_mask.max() - pred_mask.min())

    # 2. Guided Filter
    src_img_pil = image.resize((IMG_SIZE, IMG_SIZE)).convert("L")
    src_img = np.array(src_img_pil).astype(np.float32) / 255.0
    guidance_mask = pred_mask.astype(np.float32)
    
    refined_mask = guided_filter(I=src_img, p=guidance_mask, r=guided_r, eps=guided_eps)
    pred_mask = refined_mask

    # 3. Island Removal
    pred_mask[pred_mask < cutoff] = 0.0
    temp_mask = (pred_mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(temp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        clean_mask = np.zeros_like(temp_mask)
        total_area = temp_mask.shape[0] * temp_mask.shape[1]
        min_area = total_area * min_area_ratio
        for contour in contours:
            if cv2.contourArea(contour) > min_area:
                cv2.drawContours(clean_mask, [contour], -1, 255, thickness=cv2.FILLED)
        pred_mask = np.where(clean_mask > 0, pred_mask, 0.0)

    # 4. Gamma Correction & Finalize
    pred_mask = np.power(pred_mask, gamma)
    pred_mask[pred_mask > 0.95] = 1.0
    
    # 결과 이미지 생성
    pred_mask = (pred_mask * 255).astype(np.uint8)
    mask_img = Image.fromarray(pred_mask).convert("L")
    mask_img = mask_img.resize((orig_w, orig_h), resample=Image.BILINEAR)
    
    result_img = image.copy()
    result_img.putalpha(mask_img)
    
    return result_img, mask_img
