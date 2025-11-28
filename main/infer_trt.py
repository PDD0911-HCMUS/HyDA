import argparse
import numpy as np
import cv2
import torch
import ctypes
import os

import tensorrt as trt
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda

class InferHyDATRT:
    def __init__(self, engine_path):
        # ==== Cấu hình model ====
        self.size = (640, 640)  # (W,H)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.classes = [
            "N/A", "person", "car", "rider", "bus",
            "truck", "bike", "motor", "traffic light", "traffic sign"
        ]
        self.mask_threshold = 0.5
        self.det_threshold = 0.8

        self.engine_path = engine_path

        # ==== Khởi tạo TensorRT + PLUGIN ====
        self.trt_logger = trt.Logger(trt.Logger.WARNING)

        # (1) Load thư viện plugin nếu cần (thường là libnvinfer_plugin.so hoặc .so.8)
        try:
            # Thường lib sẽ nằm trong /usr/lib/x86_64-linux-gnu
            ctypes.CDLL("libnvinfer_plugin.so.8", mode=ctypes.RTLD_GLOBAL)
        except OSError:
            try:
                ctypes.CDLL("libnvinfer_plugin.so", mode=ctypes.RTLD_GLOBAL)
            except OSError:
                print("[WARN] Could not explicitly load libnvinfer_plugin.so, "
                      "assuming it is already in LD_LIBRARY_PATH")

        # (2) Đăng ký tất cả built-in plugins (bao gồm InstanceNormalization_TRT)
        trt.init_libnvinfer_plugins(self.trt_logger, "")

        # (3) Runtime + deserialize engine
        self.runtime = trt.Runtime(self.trt_logger)

        with open(self.engine_path, "rb") as f:
            engine_bytes = f.read()
        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine from {self.engine_path}")

        self.context = self.engine.create_execution_context()

        # Giả định tên binding như khi export ONNX
        self.input_name = "images"
        self.output_names = ["pred_logits", "pred_boxes", "pred_masks"]

        self._setup_bindings()

    def _setup_bindings(self):
        """Chuẩn bị bindings, allocate buffer trên GPU cho input/output."""
        self.bindings = [None] * self.engine.num_bindings

        # Input
        self.input_binding_idx = self.engine.get_binding_index(self.input_name)
        if self.input_binding_idx < 0:
            raise RuntimeError(f"Input binding {self.input_name} not found in engine.")

        # Nếu shape là dynamic, set về (1,3,640,640)
        input_shape = self.engine.get_binding_shape(self.input_binding_idx)
        if -1 in input_shape:
            self.context.set_binding_shape(
                self.input_binding_idx, (1, 3, self.size[1], self.size[0])
            )
            input_shape = self.context.get_binding_shape(self.input_binding_idx)

        self.input_shape = tuple(input_shape)
        self.input_dtype = trt.nptype(self.engine.get_binding_dtype(self.input_binding_idx))
        self.input_nbytes = (
            int(np.prod(self.input_shape)) * np.dtype(self.input_dtype).itemsize
        )

        # GPU buffer cho input
        self.d_input = cuda.mem_alloc(self.input_nbytes)
        self.bindings[self.input_binding_idx] = int(self.d_input)

        # Outputs
        self.output_info = {}  # name -> (host_array, device_ptr, binding_index)
        for name in self.output_names:
            idx = self.engine.get_binding_index(name)
            if idx < 0:
                raise RuntimeError(f"Output binding {name} not found in engine.")
            # Với dynamic shape thì sau khi set input shape, output shape sẽ cụ thể
            out_shape = tuple(self.context.get_binding_shape(idx))
            out_dtype = trt.nptype(self.engine.get_binding_dtype(idx))
            host_arr = np.empty(out_shape, dtype=out_dtype)
            d_out = cuda.mem_alloc(host_arr.nbytes)
            self.bindings[idx] = int(d_out)
            self.output_info[name] = (host_arr, d_out, idx)

        print("[INFO] TensorRT bindings set up:")
        print(f"  Input  {self.input_name}: shape={self.input_shape}, dtype={self.input_dtype}")
        for name, (h, _, _) in self.output_info.items():
            print(f"  Output {name}: shape={h.shape}, dtype={h.dtype}")

    def _infer_trt(self, input_np):
        """
        input_np: np.ndarray (1,3,640,640), float32
        return: (pred_logits, pred_boxes, pred_masks) as numpy arrays
        """
        if input_np.shape != self.input_shape:
            raise ValueError(
                f"Input shape mismatch. Expected {self.input_shape}, got {input_np.shape}"
            )

        # copy input lên GPU
        cuda.memcpy_htod(self.d_input, input_np.ravel())

        # execute
        self.context.execute_v2(self.bindings)

        # copy output về host
        outputs = {}
        for name, (h_arr, d_ptr, _) in self.output_info.items():
            cuda.memcpy_dtoh(h_arr, d_ptr)
            outputs[name] = h_arr

        return outputs["pred_logits"], outputs["pred_boxes"], outputs["pred_masks"]

    def rescale_boxes(self, boxes, orig_size):
        """
        boxes: (N,4) in cxcywh normalized (0..1)
        Return xyxy in original resolution
        """
        orig_h, orig_w = orig_size
        cx, cy, w, h = boxes.T

        x1 = (cx - w / 2) * orig_w
        y1 = (cy - h / 2) * orig_h
        x2 = (cx + w / 2) * orig_w
        y2 = (cy + h / 2) * orig_h

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
            align_corners=False,
        ).squeeze()

        mask_bin = (mask_up > self.mask_threshold).cpu().numpy().astype(np.uint8)
        return mask_bin

    def preprocess_frame(self, frame_rgb):
        """
        frame_rgb: np.ndarray [H,W,3] (RGB)
        trả về:
            frame_tensor: (1,3,640,640) float32
            orig_size   : (H_orig, W_orig)
        """
        h0, w0 = frame_rgb.shape[:2]
        orig_size = (h0, w0)

        # Resize to model input
        frame_resized = cv2.resize(frame_rgb, self.size[::-1])  # (W,H)

        # Normalize
        frame_norm = frame_resized.astype(np.float32) / 255.0
        frame_norm = (frame_norm - self.mean) / self.std

        # To NCHW
        frame_tensor = frame_norm.transpose(2, 0, 1)[None, :]  # (1,3,H,W)
        return frame_tensor.astype(np.float32), orig_size

    def visualize(self, img_path, boxes, labels, scores, mask):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Draw boxes
        for (x1, y1, x2, y2), sc, lb in zip(boxes, scores, labels):
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cls_name = self.classes[int(lb)] if int(lb) < len(self.classes) else str(int(lb))
            cv2.putText(
                img,
                f"{cls_name}:{sc:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2,
            )

        # Overlay mask
        mask_colored = np.zeros_like(img)
        mask_colored[:, :, 1] = mask * 255  # green

        overlay = cv2.addWeighted(img, 0.7, mask_colored, 0.3, 0)

        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imshow("Result", overlay_bgr)
        cv2.waitKey(0)

    def visualize_frame(self, frame_bgr, boxes_xyxy, labels, scores, final_mask):
        """
        frame_bgr: np.ndarray [H,W,3] (BGR)
        boxes_xyxy: [N,4]
        labels: [N]
        scores: [N]
        final_mask: [H,W] hoặc [1,H,W]
        """
        if isinstance(boxes_xyxy, torch.Tensor):
            boxes_xyxy = boxes_xyxy.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()

        h, w, _ = frame_bgr.shape

        # Vẽ mask
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
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 255), 1)
            # cls_name = self.classes[int(label)] if int(label) < len(self.classes) else str(int(label))
            # text = f"{cls_name}:{score:.2f}"
            # cv2.putText(
            #     frame_bgr,
            #     text,
            #     (x1, max(0, y1 - 5)),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.5,
            #     (0, 255, 255),
            #     1,
            #     cv2.LINE_AA,
            # )

        return frame_bgr

    def run_image(self, frame_path):
        # Đọc ảnh
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"[ERROR] Cannot read image: {frame_path}")
            return
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Preprocess
        frame_tensor, orig_size = self.preprocess_frame(frame_rgb)

        # TensorRT inference
        pred_logits, pred_boxes, pred_masks = self._infer_trt(frame_tensor)

        # Post-process detection
        scores = torch.from_numpy(pred_logits[0]).softmax(-1)[:, :-1]  # remove background
        max_scores, labels = scores.max(-1)

        keep = max_scores > self.det_threshold
        boxes = pred_boxes[0][keep]
        labels = labels[keep]
        scores_kept = max_scores[keep]

        boxes_xyxy = self.rescale_boxes(boxes, orig_size)
        final_mask = self.postprocess_mask(pred_masks, orig_size)

        self.visualize(frame_path, boxes_xyxy, labels, scores_kept, final_mask)

    def run_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video: {video_path}")
            return

        frame_idx = 0
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                cap.release()
                cv2.destroyAllWindows()
                break   

            frame_idx += 1
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_tensor, orig_size = self.preprocess_frame(frame_rgb)

            pred_logits, pred_boxes, pred_masks = self._infer_trt(frame_tensor)

            scores = torch.from_numpy(pred_logits[0]).softmax(-1)[:, :-1]
            max_scores, labels = scores.max(-1)

            keep = max_scores > self.det_threshold
            boxes = pred_boxes[0][keep]
            labels = labels[keep]
            scores_kept = max_scores[keep]

            boxes_xyxy = self.rescale_boxes(boxes, orig_size)
            final_mask = self.postprocess_mask(pred_masks, orig_size)

            vis_frame = self.visualize_frame(
                frame_bgr.copy(), boxes_xyxy, labels, scores_kept, final_mask
            )

            cv2.imshow("HyDA TensorRT Inference", vis_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if frame_idx % 50 == 0:
                print(f"[INFO] Processed {frame_idx} frames")

        cap.release()
        cv2.destroyAllWindows()


def get_args_parser():
    parser = argparse.ArgumentParser("Inference with TensorRT engine", add_help=True)
    parser.add_argument(
        "--engine",
        default="checkpoint/hyda_r50_e93_fp32.trt",
        type=str,
        help="Path to TensorRT engine (.trt)",
    )
    parser.add_argument("--frame", default=None, type=str, help="Path to test image")
    parser.add_argument("--video", default=None, type=str, help="Path to test video")
    return parser


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    infer = InferHyDATRT(args.engine)

    if args.frame is not None:
        infer.run_image(args.frame)
    elif args.video is not None:
       infer.run_video(args.video)
    else:
        print("[ERROR] Please specify --frame or --video")
