import torch
from torch import nn
import yaml
from models.hyda import build_model


class Wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

        # Disable auxiliary losses when exporting
        if hasattr(self.model, "aux_loss"):
            self.model.aux_loss = False

    def forward(self, frame):
        outputs = self.model(frame)
        return outputs["pred_logits"], outputs["pred_boxes"], outputs["pred_masks"]


class ExportONNX():

    def __init__(
        self,
        model_yml,
        common_yml,
        state_dict,
        onnx_path,
        opset=16,          # nên dùng 16 cho TRT 8, nếu bạn muốn giữ 17 cũng được
    ):
        self.model_cfg = self._load_cfg(model_yml)
        self.common_cfg = self._load_cfg(common_yml)
        self.state_dict = state_dict
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ✅ DÙNG SIZE CỐ ĐỊNH 640x640 CHO EXPORT
        self.height = 640
        self.width = 640

        self.onnx_path = onnx_path
        self.opset = opset

    def _load_cfg(self, yml_file):
        with open(yml_file, "r") as f:
            cfg = yaml.safe_load(f)
        return cfg

    def _create_model(self):
        cfg = self.model_cfg["model"]
        model, _ = build_model(
            cfg["hidden_dim"],
            cfg["backbone"]["position_embedding"],
            cfg["backbone"]["lr_backbone"],
            cfg["backbone"]["based"],
            cfg["backbone"]["dilation"],
            cfg["backbone"]["return_interm_layers"],
            cfg["transformer"]["dropout"],
            cfg["transformer"]["nheads"],
            cfg["transformer"]["dim_feedforward"],
            cfg["transformer"]["enc_layers"],
            cfg["transformer"]["dec_layers"],
            cfg["transformer"]["pre_norm"],
            cfg["num_queries"],
            cfg["num_drive_queries"],
            False,
            cfg["num_classes"],
            False,
        )

        state_dict = torch.load(self.state_dict, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict["model"])
        return model

    def run(self):
        # 1) build model
        model = self._create_model()
        model.to(self.device)
        model.eval()

        wrapper = Wrapper(model).to(self.device)
        wrapper.eval()

        # 2) dummy input: cố định 1x3x640x640
        dummy = torch.randn(1, 3, self.height, self.width, device=self.device)

        # 3) export ONNX
        torch.onnx.export(
            wrapper,
            dummy,
            self.onnx_path,
            input_names=["images"],
            output_names=["pred_logits", "pred_boxes", "pred_masks"],
            opset_version=self.opset,
            do_constant_folding=True,
            # ✅ CHỈ CHO PHÉP DYNAMIC BATCH, H/W CỐ ĐỊNH
            dynamic_axes={
                "images": {0: "batch"},
                "pred_logits": {0: "batch"},
                "pred_boxes": {0: "batch"},
                "pred_masks": {0: "batch"},
            },
        )
        print(f"Export ONNX to: {self.onnx_path}")


if __name__ == "__main__":

    model_yml = "configs/model_cfg.yaml"
    common_yml = "configs/common_cfg.yaml"
    state_dict = "checkpoint/checkpoint0093.pth"
    onnx_path = "checkpoint/hyda_r50_e93.onnx"
    opset = 16  # nếu muốn giữ 17 thì đổi lại = 17

    export = ExportONNX(
        model_yml,
        common_yml,
        state_dict,
        onnx_path,
        opset,
    )

    export.run()
