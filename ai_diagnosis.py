# ai_diagnosis.py

import torch
from PIL import Image
from torchvision import transforms
import models_vit as models  # 你的模型结构定义模块

# 类别标签
class_table = ('AMD-CFP', 'CSC-CFP', '其他', 'RP-CFP', 'RVO-CFP', 'normal-CFP')


# 图像预处理
transform = transforms.Compose([
    transforms.Resize(240),
    transforms.CenterCrop(240),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])



def load_retfound_model(model_path: str, model_name='RETFound_mae', input_size=384, nb_classes=6, drop_path=0.2):


    """加载训练好的 RETFound 模型"""
    if model_name == 'RETFound_mae':
        model = models.__dict__[model_name](
            num_classes=6,
            drop_path_rate=0.2,
            global_pool=True
        )


    else:
        model = models.__dict__[model_name](
            num_classes=nb_classes,
            drop_path_rate=drop_path,
            args=None,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)


    model.load_state_dict(checkpoint['model'], strict=True)
    model.cuda()
    model.eval()
    return model


def run_ai_diagnosis(img: Image.Image,
                     model: torch.nn.Module,
                     nb_classes: int = 8,
                     confidence_threshold: float = 0.7) -> dict:
    """对单张图像进行AI诊断，返回分类结果与概率
    img: PIL.Image.Image 对象，不再需要路径
    """
    # img 已经是 Image 对象，直接转换为 tensor
    img_tensor = transform(img).unsqueeze(0).cuda().float()

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        topk_probs, topk_indices = torch.topk(probs, k=min(5, nb_classes))

        top1_prob = topk_probs[0].item()
        top1_idx = topk_indices[0].item()

        if top1_prob < confidence_threshold:
            return {
                "conclusion": "未知/低置信度",
                "confidence": top1_prob,
                "class_index": -1,
                "class_name": None,
                "details": {}
            }

        return {
            "conclusion": class_table[top1_idx],
            "confidence": top1_prob,
            "class_index": top1_idx,
            "class_name": class_table[top1_idx],
            "details": {
                "top5_classes": [class_table[i] for i in topk_indices.tolist()],
                "top5_probs": [round(p.item(), 4) for p in topk_probs]
            }
        }
def run_ai_diagnosis_debug(img: Image.Image,
                           model: torch.nn.Module,
                           nb_classes: int = 8,
                           confidence_threshold: float = 0.7) -> dict:
    """
    对单张图像进行AI诊断，返回分类结果与概率，同时打印每个类别的概率用于调试。
    img: PIL.Image.Image 对象，不再需要路径
    """
    # 图像转换为 tensor 并送入 GPU
    img_tensor = transform(img).unsqueeze(0).cuda().float()

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)

        # 打印每个类别概率
        print("📊 类别概率分布：")
        for idx, p in enumerate(probs):
            print(f"  {class_table[idx]:<15}: {p.item():.4f}")

        topk_probs, topk_indices = torch.topk(probs, k=min(5, nb_classes))

        top1_prob = topk_probs[0].item()
        top1_idx = topk_indices[0].item()

        if top1_prob < confidence_threshold:
            print(f"⚠️ 低置信度: {top1_prob:.4f}")
            return {
                "conclusion": "未知/低置信度",
                "confidence": top1_prob,
                "class_index": -1,
                "class_name": None,
                "details": {
                    "top5_classes": [class_table[i] for i in topk_indices.tolist()],
                    "top5_probs": [round(p.item(), 4) for p in topk_probs]
                }
            }

        print(f"✅ 预测结果: {class_table[top1_idx]}, 置信度: {top1_prob:.4f}")
        return {
            "conclusion": class_table[top1_idx],
            "confidence": top1_prob,
            "class_index": top1_idx,
            "class_name": class_table[top1_idx],
            "details": {
                "top5_classes": [class_table[i] for i in topk_indices.tolist()],
                "top5_probs": [round(p.item(), 4) for p in topk_probs]
            }
        }
