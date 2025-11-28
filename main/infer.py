from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
# import ipywidgets as widgets
from IPython.display import display, clear_output
import yaml
from models.hyda import build_model
import torch
import torchvision.transforms as T
torch.set_grad_enabled(False)

class InferHyDA():
    def __init__(self,model_yml,
                #  common_yml,criterion_yml, 
                 state_dict):
        super().__init__()
        
        self.classes = ['N/A', 'person', "car", "rider", "bus", "truck", "bike", "motor", "traffic light", "traffic sign"]
        self.color = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
                      [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
        self.transform = T.Compose([T.Resize([640,640]),
                                    T.ToTensor(),
                                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
        self.model_cfg =  self._load_cfg(model_yml)
        # self.common_cfg = self._load_cfg(common_yml)
        # self.criterion_cfg = self._load_cfg(criterion_yml)
        self.state_dict = state_dict
    
    def _load_cfg(self, yml_file):
        with open(yml_file, 'r') as f:
            cfg = yaml.safe_load(f)
        return cfg
    
    @staticmethod
    def box_cxcywh_to_xyxy(x):
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
            (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)
    
    @staticmethod
    def rescale_bboxes(out_bbox, size):
        img_w, img_h = size
        b = InferHyDA.box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b 
    
    def plot_results_with_heatmap(self,
        pil_img,
        prob,                 # Tensor [N, num_classes] (đã softmax hoặc logits đều OK)
        boxes,                # Tensor [N, 4] ở pixel xyxy
        heatmap=None,         # Tensor logits hoặc prob của mask: [Hh,Wh] hoặc [1,1,Hh,Wh]
        thr=0.5,              # ngưỡng binary cho mask
        overlay_alpha=0.4,    # độ trong suốt heatmap/mask
        cmap='jet',
        show_binary=False     # nếu True: overlay thêm binary mask (thr)
    ):
        """
        - Vẽ bounding boxes + nhãn (giống plot_results cũ).
        - Nếu có heatmap: overlay heatmap lên ảnh.
        heatmap có thể là logits (chưa sigmoid) hoặc prob (0..1), hàm tự nhận dạng.
        - Option show_binary=True để overlay binary mask (thr).
        """
        # Chuẩn bị ảnh và axes
        fig = plt.figure(figsize=(16, 10))
        plt.imshow(pil_img)
        ax = plt.gca()

        # 1) Vẽ heatmap nếu có
        if heatmap is not None:
            hm = heatmap
            if isinstance(hm, torch.Tensor):
                hm = hm.detach().cpu()
                # chấp nhận [H,W] hoặc [1,1,H,W]
                if hm.dim() == 4:
                    hm = hm.squeeze(0).squeeze(0)  # -> [H,W]
                # nếu là logits: đưa về prob
                if hm.max() > 1.0 or hm.min() < 0.0:
                    hm = hm.sigmoid()
                hm_np = hm.numpy()  # [H,W] 0..1
            else:
                hm_np = hm  # numpy 2D

            # resize heatmap về kích thước ảnh PIL
            H, W = pil_img.size[1], pil_img.size[0]  # PIL: (W,H)
            hm_t = torch.from_numpy(hm_np).float().unsqueeze(0).unsqueeze(0)  # [1,1,Hh,Wh]
            hm_up = F.interpolate(hm_t, size=(H, W), mode='bilinear', align_corners=False).squeeze().numpy()

            # overlay heatmap
            plt.imshow(hm_up, cmap=cmap, alpha=overlay_alpha, vmin=0.0, vmax=1.0)

            # overlay binary mask (tuỳ chọn)
            if show_binary:
                mask_bin = (hm_up > thr).astype(np.float32)
                # dùng một lớp overlay xanh lá mờ
                green = np.zeros((H, W, 3), dtype=np.float32)
                green[..., 1] = 1.0
                plt.imshow(green, alpha=overlay_alpha * mask_bin, interpolation='nearest')

        # 2) Vẽ boxes + nhãn
        colors = self.color * 100
        prob_soft = prob
        if isinstance(prob_soft, torch.Tensor):
            # nếu là logits: softmax
            if prob_soft.max() > 1.0 or prob_soft.min() < 0.0:
                prob_soft = prob_soft.softmax(-1)
            prob_soft = prob_soft.detach().cpu()

        for p, (xmin, ymin, xmax, ymax), c in zip(prob_soft, boxes.tolist(), colors):
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                    fill=False, color=c, linewidth=3))
            # background là lớp cuối → bỏ lớp bg
            p_no_bg = p[:-1] if p.numel() == len(self.classes) + 1 else p
            cl = int(p_no_bg.argmax())
            score = float(p_no_bg[cl])
            name = self.classes[cl] if 0 <= cl < len(self.classes) else f'cls{cl}'
            text = f'{name}: {score:0.2f}'
            ax.text(xmin, ymin, text, fontsize=15,
                    bbox=dict(facecolor='yellow', alpha=0.5))

        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def _create_model(self):
        cfg = self.model_cfg['model']
        model, _ = build_model(
            cfg['hidden_dim'],
            cfg['backbone']['position_embedding'],
            cfg['backbone']['lr_backbone'],
            cfg['backbone']['based'],
            cfg['backbone']['dilation'],
            cfg['backbone']['return_interm_layers'],
            cfg['transformer']['dropout'],
            cfg['transformer']['nheads'],
            cfg['transformer']['dim_feedforward'],
            cfg['transformer']['enc_layers'],
            cfg['transformer']['dec_layers'],
            cfg['transformer']['pre_norm'],
            cfg['num_queries'],
            cfg['num_drive_queries'],
            False,
            cfg['num_classes'],
            False
        )

        state_dict = torch.load(self.state_dict, map_location='cpu', weights_only=False)
        model.load_state_dict(state_dict['model'])
        return model
    
    def run(self, image_file):
        model = self._create_model()
        model.eval()

        im = Image.open(image_file).convert('RGB')
        img = self.transform(im).unsqueeze(0)

        outputs = model(img)
        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > 0.8
        logits = outputs["pred_masks"][0, 0]         # [160,160], logits
        bboxes_scaled = self.rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
        scores = probas[keep]

        self.plot_results_with_heatmap(
            pil_img=im,
            prob=scores,         # tensor [N,C]
            boxes=bboxes_scaled,   # tensor [N,4]
            heatmap=outputs["pred_masks"],  # [Hh,Wh] hoặc [1,1,Hh,Wh]
            thr=0.5,
            overlay_alpha=0.45,
            cmap='jet',
            show_binary=False
        )
        pass
    
if __name__ == '__main__':
    model_yml = "configs/model_cfg.yaml"
    # common_yml = "common_cfg.yaml"
    # criterion_yml = "criterion_cfg.yaml"
    state_dict = "checkpoints/checkpoint0093.pth"

    image_file = 'data/bdd100k/bdd100k_images_100k/val/ca6412a2-3db85e24.jpg'

    infer = InferHyDA(
        model_yml, 
        # common_yml, criterion_yml, 
        state_dict
    )

    infer.run(image_file)

