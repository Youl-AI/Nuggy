import streamlit as st
from rembg import remove, new_session
from PIL import Image
import io
import time

# 페이지 설정
st.set_page_config(page_title="배경 제거기 (Pro)", page_icon="✂️")

st.title("✂️ 고화질 배경 제거기 (무제한)")
st.caption("🚀 Powered by ISNet Model (100% 무료/무제한)")

# 💡 핵심: 모델을 미리 로딩해서 캐싱 (속도 향상)
# isnet-general-use: 일반적인 사진에서 u2net보다 디테일이 훨씬 좋습니다.
@st.cache_resource
def get_model():
    # 처음 실행 때만 모델을 다운로드합니다 (약 1~2분 소요)
    return new_session("isnet-general-use")

# 사이드바: 옵션
st.sidebar.header("옵션")
alpha_matting = st.sidebar.checkbox("경계선 부드럽게 (Alpha Matting)", value=False, help="머리카락 같은 세밀한 부분을 살리려면 체크하세요. (속도는 조금 느려짐)")

# 메인 기능
uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # 화면 분할 (왼쪽: 원본, 오른쪽: 결과)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("원본")
        st.image(image, use_container_width=True)

    if st.button("배경 제거 실행"):
        # 모델 로딩 (캐시 사용)
        session = get_model()
        
        with st.spinner("AI가 열심히 지우는 중입니다... (잠시만 기다려주세요)"):
            try:
                start_time = time.time()
                
                # 이미지를 바이트로 변환
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format=image.format)
                img_byte_arr = img_byte_arr.getvalue()

                # 배경 제거 수행
                output = remove(
                    img_byte_arr, 
                    session=session,
                    alpha_matting=alpha_matting, # 옵션 적용
                    alpha_matting_foreground_threshold=240,
                    alpha_matting_background_threshold=10,
                    alpha_matting_erode_size=10
                )
                
                # 결과 변환
                result_image = Image.open(io.BytesIO(output))
                
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
                        label="이미지 다운로드 (PNG)",
                        data=byte_im,
                        file_name="isnet_result.png",
                        mime="image/png"
                    )
            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")
