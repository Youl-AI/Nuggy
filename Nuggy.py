import streamlit as st
import os
import torch
import sys
import io
from PIL import Image

# 🔥 분리한 로직 파일 임포트
import inference_utils 

# ---------------------------------------------------------
# 🎨 [UI 설정] 페이지 기본 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="NuGgy Master - AI 배경 제거",
    page_icon="🐰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일 적용
st.markdown("""
    <style>
    .main-title { font-size: 3rem; font-weight: 700; color: #2E86C1; text-align: center; margin-bottom: 10px; }
    .sub-title { font-size: 1.2rem; color: #555; text-align: center; margin-bottom: 30px; }
    div.stButton > button:first-child { background-color: #2E86C1; color: white; border-radius: 10px; border: none; width: 100%; padding: 10px 20px; font-weight: bold;}
    div.stButton > button:first-child:hover { background-color: #1B4F72; color: white; }
    .stFileUploader { border: 2px dashed #2E86C1; border-radius: 10px; padding: 20px; }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# 🛠️ 함수 정의 (모델 로드 & 상태 초기화)
# ---------------------------------------------------------
MODEL_PATH = "./checkpoints/best_finetuned_model.pth"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

@st.cache_resource
def load_model():
    # models 폴더 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_path = os.path.join(current_dir, 'DIS')
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

# 👇 [새로운 기능] 파일이 바뀌면 기존 결과를 삭제하는 함수
def reset_results():
    if 'res_img' in st.session_state:
        del st.session_state['res_img']
    if 'mask_img' in st.session_state:
        del st.session_state['mask_img']

# ---------------------------------------------------------
# 🖥️ 메인 UI 레이아웃
# ---------------------------------------------------------
st.markdown('<div class="main-title">🐰 NuGgy Master</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Fine-tuned IS-Net for High-Fidelity Matting</div>', unsafe_allow_html=True)

# 사이드바
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4472/4472515.png", width=80)
    st.title("Settings")
    st.markdown("---")
    
    st.subheader("🎨 배경 합성")
    bg_color = st.color_picker("배경색 선택", "#FFFFFF")
    
    st.markdown("---")
    st.subheader("⚙️ 튜닝 옵션")
    with st.expander("전문가 설정", expanded=True):
        
        # 👇 [수정됨] 최대값 0.9로 확장 완료!
        NOISE_CUTOFF = st.slider("노이즈 제거 (Cutoff)", 
                                 min_value=0.0, 
                                 max_value=0.9,  # <-- 0.5에서 0.9로 변경
                                 value=0.2, 
                                 step=0.01,
                                 help="값이 클수록 배경이 깨끗해지지만, 너무 높으면 피사체 일부가 지워질 수 있습니다.")
        
        GAMMA = st.slider("선명도 (Gamma)", 
                          min_value=0.1, 
                          max_value=0.9, 
                          value=0.5, 
                          step=0.1,
                          help="값이 작을수록(0.1) 피사체가 두꺼워지고, 클수록(0.9) 날씬해집니다.")
        
        # 털 디테일
        GUIDED_R = st.slider("털 디테일 (Radius)", 1, 10, 4, 1)
        
        # 고정값
        MIN_AREA = 0.001
        GUIDED_EPS = 1e-4

# 모델 로드
model = load_model()

if model is None:
    st.error(f"🚨 모델을 찾을 수 없습니다: {MODEL_PATH}")
else:
    # 👇 [수정됨] on_change=reset_results 추가 (파일 바뀌면 결과 초기화)
    uploaded_file = st.file_uploader(
        "", 
        type=["jpg", "jpeg", "png", "jfif", "webp", "bmp", "tiff"],
        help="이미지 파일을 여기에 드래그하세요.",
        on_change=reset_results  # <-- 이 부분이 핵심입니다!
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("#### 📸 원본")
            st.image(image, use_column_width=True)

        with col2:
            st.markdown("#### ✨ 결과")
            
            # 버튼 클릭 시 실행
            if st.button("배경 제거 실행", type="primary"):
                with st.spinner("AI 분석 중..."):
                    try:
                        res_img, mask_img = inference_utils.run_inference(
                            model=model,
                            image=image,
                            cutoff=NOISE_CUTOFF,
                            gamma=GAMMA,
                            guided_r=GUIDED_R,
                            guided_eps=GUIDED_EPS,
                            min_area_ratio=MIN_AREA,
                            use_tta=True
                        )
                        st.session_state['res_img'] = res_img
                        st.session_state['mask_img'] = mask_img
                    except Exception as e:
                        st.error(f"에러: {e}")

            # 결과 표시 (세션 스테이트 활용)
            if 'res_img' in st.session_state:
                final_res = st.session_state['res_img']
                final_mask = st.session_state['mask_img']
                
                tab1, tab2, tab3 = st.tabs(["투명 배경", "컬러 합성", "마스크"])
                
                with tab1:
                    st.image(final_res, use_column_width=True)
                    buf = io.BytesIO()
                    final_res.save(buf, format="PNG")
                    st.download_button("PNG 다운로드", buf.getvalue(), "result.png", "image/png")

                with tab2:
                    bg_layer = Image.new("RGB", final_res.size, bg_color)
                    comp_img = Image.alpha_composite(bg_layer.convert("RGBA"), final_res)
                    st.image(comp_img, use_column_width=True)
                    buf_comp = io.BytesIO()
                    comp_img.convert("RGB").save(buf_comp, format="JPEG")
                    st.download_button("JPG 다운로드", buf_comp.getvalue(), "result_color.jpg", "image/jpeg")
                
                with tab3:
                    st.image(final_mask, use_column_width=True)

    else:
        # 파일이 없을 때 (초기 화면 & 갤러리)
        st.info("☝️ 위 박스에 이미지를 업로드하면 배경 제거가 시작됩니다.")
        
        st.markdown("---")
        st.markdown("#### 👀 예시 결과 (Best Samples)")
        
        c1, c2, c3 = st.columns(3)
        
        # 1번: 동물 털
        with c1:
            st.markdown("##### 🐇 동물의 미세한 털")
            if os.path.exists("./assets/example_fur.png"):
                st.image("./assets/example_fur.png", caption="Fine-tuned Result", use_column_width=True)
            else:
                st.warning("이미지 준비중")

        # 2번: 라켓 줄
        with c2:
            st.markdown("##### 🏸 얇은 라켓 줄")
            if os.path.exists("./assets/example_racket.png"):
                st.image("./assets/example_racket.png", caption="Fine-tuned Result", use_column_width=True)
            else:
                st.warning("이미지 준비중")

        # 3번: 거미줄
        with c3:
            st.markdown("##### 🕸 복잡한 거미줄")
            if os.path.exists("./assets/example_web.png"):
                st.image("./assets/example_web.png", caption="Fine-tuned Result", use_column_width=True)
            else:
                st.warning("이미지 준비중")
