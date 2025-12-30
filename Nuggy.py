import streamlit as st
from PIL import Image
import io
import time
import sys
import os
from my_remover import CustomBackgroundRemover

# 이제 커스텀 모듈 임포트
try:
    from my_remover import CustomBackgroundRemover
except ImportError:
    st.error("❌ 'my_remover.py'를 찾을 수 없습니다. 파일 위치를 확인해주세요.")
    st.stop()

# ---------------------------------------------------------
# 페이지 설정
# ---------------------------------------------------------
st.set_page_config(page_title="나만의 AI 배경 제거기", page_icon="✂️")
st.title("✂️ Custom AI 배경 제거기")
st.caption("🚀 내가 직접 Fine-Tuning한 모델 사용 중")

# ---------------------------------------------------------
# 1. 모델 로드 함수 (Streamlit 캐싱 적용 ⭐️)
# 이 함수를 쓰면 버튼을 누를 때마다 모델을 다시 로드하지 않아서 빠릅니다.
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    # 학습된 모델 경로 (.pth) - 경로가 맞는지 꼭 확인하세요!
    # 팁: 절대 경로를 쓰거나, 현재 폴더 기준으로 상대 경로를 정확히 맞춰주세요.
    MODEL_PATH = "./checkpoints/best_finetuned_model.pth" 
    
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ 모델 파일이 없습니다: {MODEL_PATH}")
        st.stop()
    
    return CustomBackgroundRemover(MODEL_PATH)

# 사이드바: 옵션 (커스텀 모델은 Alpha Matting 옵션을 코드 내부에서 처리하거나 뺍니다)
st.sidebar.info("현재 Fine-Tuned ISNet 모델이 구동 중입니다.")

# ---------------------------------------------------------
# 메인 기능
# ---------------------------------------------------------
uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # 이미지 열기
    image = Image.open(uploaded_file).convert("RGB")
    
    # 화면 분할
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("원본")
        st.image(image, use_container_width=True)

    if st.button("배경 제거 실행"):
        with st.spinner("내 AI 모델이 분석 중입니다... 🧠"):
            try:
                # 1. 모델 가져오기 (캐시된 것 사용)
                remover = load_model()
                
                start_time = time.time()
                
                # 2. [핵심 변경] rembg.remove 대신 remover.process 사용
                # 커스텀 클래스는 바이트 변환 필요 없이 PIL 이미지를 바로 받습니다.
                result_image, _ = remover.process(image)
                
                # 소요 시간 계산
                end_time = time.time()
                process_time = end_time - start_time
                
                with col2:
                    st.subheader("결과")
                    st.image(result_image, use_container_width=True)
                    st.success(f"완료! ({process_time:.2f}초 소요)")
                    
                    # 다운로드 버튼
                    buf = io.BytesIO()
                    result_image.save(buf, format="PNG")
                    byte_im = buf.getvalue()
                    
                    st.download_button(
                        label="결과 다운로드 (PNG)",
                        data=byte_im,
                        file_name="custom_ai_result.png",
                        mime="image/png"
                    )
            
            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")
                # 에러 디버깅을 위해 자세한 정보 출력
                st.write(e)
