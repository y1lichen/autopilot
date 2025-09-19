import torch
from model.model import parsingNet

# 建立模型
BACKBONE = "18"
GRIDING_NUM = 100
CLS_NUM_PER_LANE = 56
USE_AUX = False
cls_dim = (GRIDING_NUM+1, CLS_NUM_PER_LANE, 4)
net = parsingNet(size=(288,800), pretrained=False, backbone=BACKBONE, cls_dim=cls_dim, use_aux=USE_AUX)
ckpt = torch.load("model/tusimple_18.pth", map_location="cpu")
state_dict = ckpt['model'] if 'model' in ckpt else ckpt
net.load_state_dict(state_dict, strict=False)
net.eval()

# dummy input
dummy_input = torch.randn(1, 3, 288, 800)

# 導出
torch.onnx.export(
    net,
    dummy_input,
    "ufld_tusimple_18.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=11,
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)
print("ONNX model exported to ufld_tusimple_18.onnx")
