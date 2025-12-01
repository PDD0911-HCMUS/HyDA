import argparse
import onnxruntime as ort
import numpy as np
import cv2
import torch
from PIL import Image

class InferHyDAONNX():
    def __init__(self, onnx_path):
        self.size = (640, 640)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.classes = ['N/A', 'person', "car", "rider", "bus", "truck", "bike", "motor", "traffic light", "traffic sign"]
        self.mask_threshold = 0.5
        self.det_threshold = 0.8
        self.onnx_path = onnx_path
        
        # Create ONNXRuntime session
        self.sess = ort.InferenceSession(
            self.onnx_path,
            providers=["CUDAExecutionProvider"]  
        )
        self.input_name = self.sess.get_inputs()[0].name
        
    def rescale_boxes(self, boxes, orig_size):
        """
        boxes: (N,4) in cxcywh normalized (0..1)
        Return xyxy in original resolution
        """
        orig_h, orig_w = orig_size
        cx, cy, w, h = boxes.T

        x1 = (cx - w/2) * orig_w
        y1 = (cy - h/2) * orig_h
        x2 = (cx + w/2) * orig_w
        y2 = (cy + h/2) * orig_h

        b = np.stack([x1, y1, x2, y2], axis=1)
        return b
    
    def postprocess_mask(self, mask, orig_size):
        """
        mask: (1,1,Hm,Wm) logits
        return: (H_orig, W_orig) uint8 binary mask
        """
        mask = torch.from_numpy(mask)
        mask = mask.sigmoid()

        # Resize to original size
        mask_up = torch.nn.functional.interpolate(
            mask,
            size=orig_size,
            mode="bilinear",
            align_corners=False
        ).squeeze()

        mask_bin = (mask_up > self.mask_threshold).cpu().numpy().astype(np.uint8)
        return mask_bin
        
    def preprocess_frame(self, frame):

        h0, w0 = frame.shape[:2]
        orig_size = (h0, w0)

        # Resize to model input
        frame_resized = cv2.resize(frame, self.size[::-1])  # (W,H)

        # Normalize
        frame_norm = frame_resized.astype(np.float32) / 255.0
        frame_norm = (frame_norm - self.mean) / self.std

        # To NCHW
        frame_tensor = frame_norm.transpose(2, 0, 1)[None, :]  # (1,3,H,W)
        return frame_tensor.astype(np.float32), orig_size, frame_resized

    def visualize(self, img_path, boxes, labels, scores, mask):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Draw boxes
        for (x1, y1, x2, y2), sc, lb in zip(boxes, scores, labels):
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img, f"{self.classes[lb]}:{sc:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Overlay mask
        mask_colored = np.zeros_like(img)
        mask_colored[:, :, 1] = mask * 255  # green

        overlay = cv2.addWeighted(img, 0.7, mask_colored, 0.3, 0)

        cv2.imshow("Result", overlay)
        cv2.waitKey(0)
    
    def visualize_frame(self, frame_bgr, boxes_xyxy, labels, scores, final_mask):
        """
        frame_bgr: np.ndarray [H,W,3] (BGR)
        boxes_xyxy: Tensor or np.ndarray [N,4]
        labels: Tensor [N]
        scores: Tensor [N]
        final_mask: np.ndarray [H,W] hoặc [1,H,W] (0/1 hoặc 0~1)
        """
        if isinstance(boxes_xyxy, torch.Tensor):
            boxes_xyxy = boxes_xyxy.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()

        h, w, _ = frame_bgr.shape

        # Vẽ mask mờ lên frame
        if final_mask is not None:
            if isinstance(final_mask, torch.Tensor):
                final_mask = final_mask.cpu().numpy()
            if final_mask.ndim == 3:
                final_mask = final_mask[0]
            mask_resized = cv2.resize(final_mask.astype(np.float32), (w, h))
            mask_color = np.zeros_like(frame_bgr, dtype=np.uint8)
            mask_color[:, :, 1] = (mask_resized * 255).astype(np.uint8)  # kênh G
            alpha = 0.4
            frame_bgr = cv2.addWeighted(frame_bgr, 1.0, mask_color, alpha, 0)

        # Vẽ bbox + label
        for box, label, score in zip(boxes_xyxy, labels, scores):
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 255), 2)
            text = f"{int(label)}:{score:.2f}"
            cv2.putText(frame_bgr, text, (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        return frame_bgr
    
    def run(self, frame_path):
        
        # Preprocess
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor, orig_size, frame_resized = self.preprocess_frame(frame)
        
        # Run
        pred_logits, pred_boxes, pred_masks = self.sess.run(
            None,
            {self.input_name: frame_tensor}
        )
        # Shape info
        # print("pred_logits:", pred_logits.shape)  # (1,100,num_classes)
        # print("pred_boxes :", pred_boxes.shape)   # (1,100,4)
        # print("pred_masks :", pred_masks.shape)   # (1,1,Hm,Wm)
        
        scores = torch.from_numpy(pred_logits[0]).softmax(-1)[:, :-1]  # remove background
        max_scores, labels = scores.max(-1)

        keep = max_scores > self.det_threshold
        boxes = pred_boxes[0][keep]
        labels = labels[keep]
        scores = max_scores[keep]
        
        boxes_xyxy = self.rescale_boxes(boxes, orig_size)
        final_mask = self.postprocess_mask(pred_masks, orig_size)
        
        self.visualize(
            frame_path,
            boxes_xyxy, labels, scores, final_mask
        )

    def run_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video: {video_path}")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        frame_idx = 0
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            # BGR -> RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            # Preprocess (giữ nguyên logic của bạn)
            frame_tensor, orig_size, frame_resized = self.preprocess_frame(frame_rgb)

            # ONNX inference
            pred_logits, pred_boxes, pred_masks = self.sess.run(
                None,
                {self.input_name: frame_tensor}
            )

            # --- Post-process detection ---
            scores = torch.from_numpy(pred_logits[0]).softmax(-1)[:, :-1]
            max_scores, labels = scores.max(-1)

            keep = max_scores > self.det_threshold
            boxes = pred_boxes[0][keep]
            labels = labels[keep]
            scores_kept = max_scores[keep]

            boxes_xyxy = self.rescale_boxes(boxes, orig_size)

            final_mask = self.postprocess_mask(pred_masks, orig_size)  # [H, W] or [1,H,W]

            vis_frame = self.visualize_frame(
                frame_bgr.copy(), 
                boxes_xyxy, labels, scores_kept, final_mask
            )
            
            cv2.imshow("HyDA Inference", vis_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            if frame_idx % 50 == 0:
                print(f"[INFO] Processed {frame_idx} frames")
                
        cap.release()
        cv2.destroyAllWindows()
        
def get_args_parser():
    parser = argparse.ArgumentParser('Inference with ONNX', add_help=False)
    parser.add_argument('--onnx', default=None, type=str, help="Path to the ONNX model")
    parser.add_argument('--frame', default=None, type=str, help="Path to the image test")
    parser.add_argument('--video', default=None, type=str, help="Path to the video test")
    

if __name__ == "__main__":
    onnx_path = "checkpoint/hyda_r50_e93.onnx"
    img_path = "/home/map4/ThisPC/PhD_Journey/Datasets/bdd100k/bdd100k_images_100k/val/b1c9c847-3bda4659.jpg"

    infer = InferHyDAONNX(onnx_path)
    infer.run(img_path)