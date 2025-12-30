import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import sys

# [중요] 모델 경로 설정 (학습 때와 동일하게)
# 사용자의 폴더 구조에 맞춰 경로를 잡아줍니다.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root_path = os.path.join(current_dir, 'DIS')
if project_root_path not in sys.path:
    sys.path.append(project_root_path)

try:
    from models.isnet import ISNetDIS
except ModuleNotFoundError:
    print("❌ 에러: 'models' 폴더를 찾을 수 없습니다. 경로를 확인해주세요.")

class CustomBackgroundRemover:
    def __init__(self, model_path, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"⏳ 커스텀 모델 로딩 중... ({self.device})")
        
        # 모델 구조 불러오기
        self.model = ISNetDIS().to(self.device)
        
        # 학습된 가중치(.pth) 로드
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval() # 평가 모드 전환
            print("✅ 모델 로드 완료!")
        else:
            raise FileNotFoundError(f"모델 파일이 없습니다: {model_path}")

        # 이미지 전처리 설정 (학습 때와 동일한 1024 사이즈)
        self.transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
        ])

    def process(self, original_image):
        """
        original_image: PIL Image 객체
        return: 배경이 제거된 PIL Image (RGBA)
        """
        # 1. 원본 크기 저장
        w, h = original_image.size
        
        # 2. 전처리 (Resize & Normalize)
        image_tensor = self.transform(original_image.convert("RGB")).unsqueeze(0).to(self.device)
        
        # 3. 추론 (Inference)
        with torch.no_grad():
            preds = self.model(image_tensor)
            if isinstance(preds, tuple): preds = preds[0]
            pred_mask = preds[0][0] # 배치 차원 제거

        # 4. 마스크 후처리
        # (1) 0~1 사이로 정규화 (혹시 모를 오차 방지)
        pred_mask = torch.sigmoid(pred_mask) 
        
        # (2) 원본 크기로 다시 복구 (Bilinear Interpolation)
        # 텐서 형태: (1, 1, H, W)가 되어야 interpolate 가능
        pred_mask = F.interpolate(pred_mask.unsqueeze(0).unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False)
        pred_mask = pred_mask.squeeze().cpu().numpy()

        # 5. 배경 제거 합성 (RGBA 변환)
        # 마스크를 PIL 이미지로 변환
        mask_pil = Image.fromarray((pred_mask * 255).astype(np.uint8)).convert("L")
        
        # 원본 이미지에 알파 채널(마스크) 적용
        result_image = original_image.convert("RGBA")
        result_image.putalpha(mask_pil)
        
        return result_image
