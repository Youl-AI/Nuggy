import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import sys

# ---------------------------------------------------------
# 경로 설정
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
        print(f"⚡ Device: {self.device}")
        
        self.model = ISNetDIS().to(self.device)
        
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
        else:
            print(f"❌ 모델 파일 없음: {model_path}")

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
            pred_mask_tensor = preds[0] 

        # 3. 후처리
        # (1) Sigmoid (확률값 변환)
        pred_mask_tensor = torch.sigmoid(pred_mask_tensor)

        # (2) 크기 복원
        pred_mask = F.interpolate(pred_mask_tensor, size=(h, w), mode='bilinear', align_corners=False)
        pred_mask = pred_mask.squeeze().cpu().numpy()

        # =========================================================
        # 💡 [핵심 추가] 흐릿한 회색을 선명하게 만들기 (Min-Max Normalization)
        # =========================================================
        # 현재 값의 범위 확인 (예: 0.1 ~ 0.4 라고 가정)
        min_val = pred_mask.min()
        max_val = pred_mask.max()
        
        # 만약 모델이 너무 소심해서 최대값이 0에 가까우면 그냥 둡니다.
        if max_val - min_val > 0.1: 
            # 0.1 ~ 0.4 범위를 -> 0.0 ~ 1.0 으로 강제로 쫙 펴줍니다.
            pred_mask = (pred_mask - min_val) / (max_val - min_val)
        
        # (3) 이제 0.5 기준으로 확실하게 자릅니다.
        pred_mask[pred_mask < 0.5] = 0 
        pred_mask[pred_mask >= 0.5] = 1
        # =========================================================

        # 4. 이미지 합성
        mask_pil = Image.fromarray((pred_mask * 255).astype(np.uint8)).convert("L")
        
        result_image = original_image.convert("RGBA")
        result_image.putalpha(mask_pil)
        
        # 웹 디버깅을 위해 마스크도 같이 반환
        return result_image, mask_pil
