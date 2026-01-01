import streamlit as st
import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import io

# ---------------------------------------------------------
# 🎨 [UI 설정] 페이지 기본 설정 (가장 먼저 실행되어야 함)
# ---------------------------------------------------------
st.set_page_config(
    page_title="NuGgy Master - AI 배경 제거",
    page_icon="🐰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# 🎨 [커스텀 CSS] 웹페이지를 예쁘게 꾸미기 위한 스타일
# ---------------------------------------------------------
st.markdown("""
    <style>
    /* 메인 타이틀 폰트 및 정렬 */
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        color: #2E86C1;
        text-align: center;
        margin-bottom: 10px;
    }
    .sub-title {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 30px;
    }
    /* 버튼 스타일 커스텀 */
    div.stButton > button:first-child {
        background-color: #2E86C1;
        color: white;
        font-size: 18px;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 20px;
        border: none;
        width: 100%;
    }
    div.stButton > button:first-child:hover {
        background-color: #1B4F72;
        color: white;
    }
    /* 파일 업로더 박스 스타일 */
    .stFileUploader {
        border: 2px dashed #2E86C1;
        border-radius: 10px;
        padding: 20px;
    }
    /* 결과 이미지 컨테이너 */
    .result-container {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# 🛠️ [설정] 모델 및 파라미터
# ---------------------------------------------------------
MODEL_PATH = "./checkpoints/best_finetuned_model.pth"
IMG_SIZE = 1024
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 기본값 설정
NOISE_CUTOFF = 0.2
MIN_AREA_RATIO = 0.001
GAMMA = 0.5
GUIDED_R = 4
GUIDED_EPS = 1e-4
USE_TTA = True

# ---------------------------------------------------------
# 🧩 함수 정의 (모델 로드 & 알고리즘)
# ---------------------------------------------------------
@st.cache_resource
def load_model():
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

def run_inference(model, image, cutoff, gamma, guided_r, guided_eps):
    orig_w, orig_h = image.size
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    
    # TTA Inference
    with torch.no_grad():
        img_tensor = transform(image).unsqueeze(0).to(DEVICE)
        preds_1 = model(img_tensor)
        while isinstance(preds_1, (list, tuple)): preds_1 = preds_1[0]
        final_mask = torch.sigmoid(preds_1)

        if USE_TTA:
            img_flip = image.transpose(Image.FLIP_LEFT_RIGHT)
            img_flip_tensor = transform(img_flip).unsqueeze(0).to(DEVICE)
            preds_2 = model(img_flip_tensor)
            while isinstance(preds_2, (list, tuple)): preds_2 = preds_2[0]
            mask_2 = torch.sigmoid(preds_2)
            mask_2 = torch.flip(mask_2, dims=[3]) 
            final_mask = (final_mask + mask_2) / 2.0

    pred_mask = final_mask.squeeze().cpu().numpy()
    if pred_mask.max() != pred_mask.min():
        pred_mask = (pred_mask - pred_mask.min()) / (pred_mask.max() - pred_mask.min())

    # Guided Filter
    src_img_pil = image.resize((IMG_SIZE, IMG_SIZE)).convert("L")
    src_img = np.array(src_img_pil).astype(np.float32) / 255.0
    guidance_mask = pred_mask.astype(np.float32)
    refined_mask = guided_filter(I=src_img, p=guidance_mask, r=guided_r, eps=guided_eps)
    pred_mask = refined_mask

    # Island Removal
    pred_mask[pred_mask < cutoff] = 0.0
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

    # Gamma & Finalize
    pred_mask = np.power(pred_mask, gamma)
    pred_mask[pred_mask > 0.95] = 1.0
    
    pred_mask = (pred_mask * 255).astype(np.uint8)
    mask_img = Image.fromarray(pred_mask).convert("L")
    mask_img = mask_img.resize((orig_w, orig_h), resample=Image.BILINEAR)
    
    result_img = image.copy()
    result_img.putalpha(mask_img)
    
    return result_img, mask_img

# ---------------------------------------------------------
# 🖥️ 메인 UI 레이아웃
# ---------------------------------------------------------

# 1. 헤더 섹션
st.markdown('<div class="main-title">🐰 NuGgy Master</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI Powered Background Removal Tool | Fine-tuned IS-Net</div>', unsafe_allow_html=True)

# 2. 사이드바 (옵션 설정)
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4472/4472515.png", width=80)
    st.title("Settings")
    st.markdown("---")
    
    st.subheader("🎨 배경 합성 옵션")
    bg_color = st.color_picker("배경색 선택 (합성용)", "#FFFFFF")
    
    st.markdown("---")
    st.subheader("⚙️ 고급 설정 (Tuning)")
    
    with st.expander("전문가 옵션 펼치기"):
        st.info("결과가 마음에 안 들면 조절하세요.")
        val_cutoff = st.slider("노이즈 제거 강도", 0.0, 0.1, NOISE_CUTOFF, 0.01, help="값이 클수록 배경이 깨끗해지지만, 털 끝이 잘릴 수 있습니다.")
        val_gamma = st.slider("선명도 보정", 0.1, 1.0, GAMMA, 0.1, help="값이 작을수록 피사체가 두꺼워집니다.")
        val_guided_r = st.slider("털 디테일 반경", 1, 10, GUIDED_R, 1, help="털이 뭉개지면 이 값을 줄이세요.")

# 3. 모델 로드
model = load_model()

if model is None:
    st.error("🚨 모델 파일을 찾을 수 없습니다! `checkpoints` 폴더에 모델이 있는지 확인해주세요.")
else:
    # 4. 파일 업로드 섹션
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"], help="여기에 이미지를 드래그 앤 드롭하세요.")

    if uploaded_file is not None:
        # 이미지 로드
        image = Image.open(uploaded_file).convert("RGB")
        
        # 2단 컬럼 레이아웃
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 📸 원본 이미지")
            st.image(image, use_column_width=True, caption="Original Image")

        with col2:
            st.markdown("#### ✨ 결과 미리보기")
            
            # [버튼] 실행 트리거
            if st.button("배경 제거 실행 (Start Process)", type="primary"):
                with st.spinner("AI가 이미지를 분석하고 있습니다... 🐰"):
                    try:
                        res_img, mask_img = run_inference(
                            model, image, val_cutoff, val_gamma, val_guided_r, GUIDED_EPS
                        )
                        
                        # 세션 상태에 저장 (새로고침 방지)
                        st.session_state['res_img'] = res_img
                        st.session_state['mask_img'] = mask_img
                        
                    except Exception as e:
                        st.error(f"오류가 발생했습니다: {e}")

            # 결과가 있으면 보여주기
            if 'res_img' in st.session_state:
                final_res = st.session_state['res_img']
                final_mask = st.session_state['mask_img']
                
                # 탭 레이아웃 (여기가 핵심!)
                tab1, tab2, tab3 = st.tabs(["⬜ 투명 배경", "🎨 컬러 배경 합성", "⚫️ 마스크(Mask)"])
                
                with tab1:
                    st.image(final_res, use_column_width=True, caption="Transparent Background")
                    # 다운로드 버튼
                    buf = io.BytesIO()
                    final_res.save(buf, format="PNG")
                    st.download_button("📥 투명 PNG 다운로드", buf.getvalue(), "nuggy_transparent.png", "image/png")

                with tab2:
                    # 배경 합성 로직
                    bg_layer = Image.new("RGB", final_res.size, bg_color)
                    comp_img = Image.alpha_composite(bg_layer.convert("RGBA"), final_res)
                    st.image(comp_img, use_column_width=True, caption=f"Background Color: {bg_color}")
                    
                    # 합성 다운로드
                    buf_c = io.BytesIO()
                    comp_img.convert("RGB").save(buf_c, format="JPEG")
                    st.download_button("📥 합성된 JPG 다운로드", buf_c.getvalue(), "nuggy_color.jpg", "image/jpeg")

                with tab3:
                    st.image(final_mask, use_column_width=True, caption="Segmentation Mask")

    else:
        # 파일 없을 때 안내 문구
        st.info("☝️ 위 박스에 이미지를 업로드하면 배경 제거가 시작됩니다.")
        
        # 데모용 갤러리 (빈 공간 채우기)
        st.markdown("---")
        st.markdown("#### 👀 예시 결과")
        c1, c2, c3 = st.columns(3)
        c1.markdown("🐇 **동물의 미세한 털**")
        c2.markdown("🏸 **얇은 라켓 줄**")
        c3.markdown("🕸 **복잡한 거미줄**")
