import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(page_title="배경 제거기 2.0", page_icon="✂️")

st.title("🚀 배경 제거기 (BRIA RMBG-2.0)")
st.caption("Powered by BRIA RMBG-2.0 via Hugging Face API")
st.info("⚠️ 이 모델은 Hugging Face 사이트에서 [RMBG-2.0 라이선스 동의](https://huggingface.co/briaai/RMBG-2.0)를 해야 작동합니다.")

# ✅ BRIA RMBG-2.0 모델 주소
API_URL = "https://api-inference.huggingface.co/models/briaai/RMBG-2.0"

# 토큰 가져오기
try:
    hf_token = st.secrets["HF_TOKEN"]
except FileNotFoundError:
    st.error("비밀 키 설정이 되어있지 않습니다. secrets.toml 파일을 확인해주세요.")
    st.stop()

headers = {"Authorization": f"Bearer {hf_token}"}

def query(image_bytes):
    response = requests.post(API_URL, headers=headers, data=image_bytes)
    
    # 에러 처리
    if response.status_code != 200:
        # 403 에러는 사용 동의를 안 했을 때 발생
        if response.status_code == 403:
             raise Exception("권한 오류(403): Hugging Face 홈페이지에서 'briaai/RMBG-2.0' 모델 사용 동의 버튼을 눌러주세요.")
        # 503 에러는 모델 로딩 중
        elif response.status_code == 503:
             raise Exception("모델을 로딩 중입니다. 잠시 후 다시 시도해주세요.")
        
        raise Exception(f"API Error: {response.status_code} - {response.text}")
        
    return response.content

# 메인 화면 구성
uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="원본 이미지", use_container_width=True)

    if st.button("배경 제거 실행"):
        with st.spinner("RMBG-2.0 모델로 분석 중..."):
            try:
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format=image.format)
                img_byte_arr = img_byte_arr.getvalue()

                image_bytes = query(img_byte_arr)
                
                result_image = Image.open(io.BytesIO(image_bytes))
                
                st.success("완료!")
                st.image(result_image, caption="결과 이미지", use_container_width=True)
                
                buf = io.BytesIO()
                result_image.save(buf, format="PNG")
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="이미지 다운로드 (PNG)",
                    data=byte_im,
                    file_name="rmbg_2.0_result.png",
                    mime="image/png"
                )
            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")
