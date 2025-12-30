import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import sys

# ---------------------------------------------------------
# 경로 설정 (models 폴더 위치 찾기)
# ---------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from models.isnet import ISNetDIS
except ModuleNotFoundError:
    sys.path.append(os.path.join(current_dir, 'models'))
    from isnet import ISNetDIS

class CustomBackgroundRemover:
    def __init__(self, model_path, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"⚡ 커스텀 모델 로딩... Device: {self.device}")
        
        self.model = ISNetDIS().to(self.device)
        
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print("✅ 모델 로드 완료!")
        else:
            print(f"❌ 모델 파일 없음: {model_path}")
            # 웹 환경에서는 멈추는게 나을 수 있음
            # sys.exit() 

        # 학습 때와 동일한 전처리 (1024 사이즈)
        self.transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
        ])

    def process(self, original_image):
        w, h = original_image.size
        
        # 1. 전처리
        image_tensor = self.transform(original_image.convert("RGB")).unsqueeze(0).to(self.device)
        
        # 2. 추론
        with torch.no_grad():
            preds = self.model(image_tensor)
            if isinstance(preds, tuple): preds = preds[0]
            # 4D 텐서 유지 (1, 1, 1024, 1024)
            pred_mask_tensor = preds[0] 

        # 3. 후처리 시작
        # (1) Sigmoid로 0~1 확률값 변환
        pred_mask_tensor = torch.sigmoid(pred_mask_tensor)

        # (2) 원본 크기 복원 (Interpolate)
        pred_mask = F.interpolate(pred_mask_tensor, size=(h, w), mode='bilinear', align_corners=False)
        pred_mask = pred_mask.squeeze().cpu().numpy() # (H, W) 2D 배열로 변환

        # =========================================================
        # 💡 [핵심 수정] 유령 현상 해결! (확실하게 자르기)
        # =========================================================
        # 0.5를 기준으로 흰색(1)과 검은색(0)으로 딱 나눕니다.
        # 이 부분이 없으면 가장자리가 흐릿해집니다.
        pred_mask[pred_mask < 0.5] = 0 
        pred_mask[pred_mask >= 0.5] = 1
        # =========================================================

        # 4. 이미지 합성
        # 마스크를 이미지로 변환 (0/1 -> 0/255)
        mask_pil = Image.fromarray((pred_mask * 255).astype(np.uint8)).convert("L")
        
        result_image = original_image.convert("RGBA")
        result_image.putalpha(mask_pil)
        
        return result_image

# (테스트 코드는 웹 실행 시 필요 없으므로 제거했습니다)
