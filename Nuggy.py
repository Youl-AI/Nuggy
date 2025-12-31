import streamlit as st
import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import io

# ---------------------------------------------------------
# 🛠️ [설정] 웹에서도 최강의 성능을 유지합니다
# ---------------------------------------------------------
MODEL_PATH = "./checkpoints/best_finetuned_model.pth"
IMG_SIZE = 1024
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# (성능 파라미터 - 아까 맞춘 최적값)
NOISE_CUTOFF = 0.2      # 배경 노이즈 제거 기준
MIN_AREA_RATIO = 0.001   # 작은 먼지 제거 기준
GAMMA = 0.5              # 선명도 보정
GUIDED_R = 4             # 털 디테일 반경
GUIDED_EPS = 1e-4        # 털 디테일 민감도
USE_TTA = True           # 고성능 모드

# ---------------------------------------------------------
# 🧩 함수 정의 (모델 로드 & 알고리즘)
# ---------------------------------------------------------

# 1. 모델 로드 (캐싱을 통해 속도 향상)
@st.cache_resource
def load_model():
    # 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_path = os.path.join(current_dir, 'DIS')
    import sys
    if project_root_path not in sys.path:
        sys.path.append(project_root_path)

    try:
        from models.isnet import ISNetDIS
        model = ISNetDIS().to(DEVICE)
        
        if not os.path.exists(MODEL_PATH):
            return None
            
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        return model
    except Exception as e:
        return None

# 2. 가이디드 필터 (OpenCV 직접 구현)
def guided_filter(I, p, r, eps):
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

# 3. 추론 엔진 (Masterpiece 로직 적용)
def run_inference(model, image):
    orig_w, orig_h = image.size
    
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    
    # (A) TTA 추론
    with torch.no_grad():
        img_tensor = transform(image).unsqueeze(0).to(DEVICE)
        preds_1 = model(img_tensor)
        while isinstance(preds_1, (list, tuple)): preds_1 = preds_1[0]
        mask_1 = torch.sigmoid(preds_1)
        final_mask = mask_1

        if USE_TTA:
            img_flip = image.transpose(Image.FLIP_LEFT_RIGHT)
            img_flip_tensor = transform(img_flip).unsqueeze(0).to(DEVICE)
            preds_2 = model(img_flip_tensor)
            while isinstance(preds_2, (list, tuple)): preds_2 = preds_2[0]
            mask_2 = torch.sigmoid(preds_2)
            mask_2 = torch.flip(mask_2, dims=[3]) 
            final_mask = (mask_1 + mask_2) / 2.0

    pred_mask = final_mask.squeeze().cpu().numpy()
    if pred_mask.max() != pred_mask.min():
        pred_mask = (pred_mask - pred_mask.min()) / (pred_mask.max() - pred_mask.min())

    # (B) Guided Filter
    src_img_pil = image.resize((IMG_SIZE, IMG_SIZE)).convert("L")
    src_img = np.array(src_img_pil).astype(np.float32) / 255.0
    guidance_mask = pred_mask.astype(np.float32)
    refined_mask = guided_filter(I=src_img, p=guidance_mask, r=GUIDED_R, eps=GUIDED_EPS)
    pred_mask = refined_mask

    # (C) Island Removal (먼지 청소)
    pred_mask[pred_mask < NOISE_CUTOFF] = 0.0
    temp_mask = (pred_mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(temp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        clean_mask = np.zeros_like(temp_mask)
        total_area = temp_mask.shape[0] * temp_mask.shape[1]
        min_area = total_area * MIN_AREA_RATIO
        for contour in contours:
            if cv2.contourArea(contour) > min_area:
                cv2.drawContours(clean_mask, [contour], -1, 255, thickness=cv2.FILLED)
        pred_mask = np.where(clean_mask > 0, pred_mask, 0.0)

    # (D) 마무리
    pred_mask = np.power(pred_mask, GAMMA)
    pred_mask[pred_mask > 0.95] = 1.0
    
    # 이미지 복원
    pred_mask = (pred_mask * 255).astype(np.uint8)
    mask_img = Image.fromarray(pred_mask).convert("L")
    mask_img = mask_img.resize((orig_w, orig_h), resample=Image.BILINEAR)
    
    result_img = image.copy()
    result_img.putalpha(mask_img)
    
    return result_img, mask_img

# ---------------------------------------------------------
# 🖥️ 웹 UI 구성
# ---------------------------------------------------------
st.set_page_config(page_title="AI 누끼 마스터", layout="wide")

st.title("🐰 AI 배경 제거기 (Masterpiece Ver.)")
st.markdown("사용자님의 Fine-tuned 모델을 사용하여 **털끝 하나까지 살리는** 고성능 배경 제거를 수행합니다.")

# 사이드바 설정
with st.sidebar:
    st.header("⚙️ 고급 설정")
    st.info(f"현재 모델: {os.path.basename(MODEL_PATH)}")
    
    # 사용자가 직접 조절 가능하게 UI 연결
    new_cutoff = st.slider("배경 제거 강도 (Noise Cutoff)", 0.0, 0.1, NOISE_CUTOFF, 0.01)
    new_gamma = st.slider("피사체 선명도 (Gamma)", 0.1, 1.0, GAMMA, 0.1)
    
    # 전역 변수 업데이트
    NOISE_CUTOFF = new_cutoff
    GAMMA = new_gamma

# 모델 로드
model = load_model()

if model is None:
    st.error(f"❌ 모델 파일을 찾을 수 없습니다! 경로를 확인해주세요: {MODEL_PATH}")
else:
    # 파일 업로드
    uploaded_file = st.file_uploader("이미지를 업로드하세요 (JPG, PNG)", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="원본 이미지", use_column_width=True)

        if st.button("🚀 배경 제거 시작!", type="primary"):
            with st.spinner("AI가 배경을 지우는 중입니다... (TTA + Guided Filter 적용 중)"):
                try:
                    result_img, mask_img = run_inference(model, image)
                    
                    with col2:
                        st.image(result_img, caption="결과 이미지", use_column_width=True)
                    
                    # 다운로드 버튼
                    buf = io.BytesIO()
                    result_img.save(buf, format="PNG")
                    byte_im = buf.getvalue()
                    
                    st.success("작업 완료!")
                    st.download_button(
                        label="📥 결과 이미지 다운로드 (PNG)",
                        data=byte_im,
                        file_name="remove_bg_result.png",
                        mime="image/png",
                    )
                    
                    # 마스크 확인용 (아코디언)
                    with st.expander("🔍 마스크(Mask) 자세히 보기"):
                        st.image(mask_img, caption="생성된 마스크", width=300)
                        
                except Exception as e:
                    st.error(f"에러가 발생했습니다: {e}")
