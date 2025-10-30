import argparse
import os
import cv2
import numpy as np
import torch
from tqdm import tqdm

def letterbox(img, new_size=640, color=(114, 114, 114)):
    """Resize with unchanged aspect ratio and pad to square."""
    h, w = img.shape[:2]
    scale = min(new_size / h, new_size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    img_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (new_size - nh) // 2
    bottom = new_size - nh - top
    left = (new_size - nw) // 2
    right = new_size - nw - left
    img_padded = cv2.copyMakeBorder(
        img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    meta = {
        "scale": scale,
        "pad": (left, top),
        "resized_wh": (nw, nh),    # size after aspect-ratio resize (before padding)
        "orig_hw": (h, w),         # original frame size
        "box_size": new_size       # padded square size (e.g., 640)
    }
    return img_padded, meta

def main():
    ap = argparse.ArgumentParser(description="YOLOP lane overlay on video")
    ap.add_argument("--source", required=True, help="Path to input .mp4/.mov")
    ap.add_argument("--save", default="runs/yolop_lane_out.mp4", help="Path to save output video")
    ap.add_argument("--show", action="store_true", help="Show live preview window")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--alpha", type=float, default=0.45, help="Overlay opacity (0-1)")
    args = ap.parse_args()

    # Ensure output folder exists
    save_dir = os.path.dirname(args.save)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Load YOLOP from PyTorch Hub
    model = torch.hub.load(
        'hustvl/yolop', 'yolop',
        pretrained=True, verbose=False, trust_repo=True
    ).eval().to(args.device)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.source}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    out_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(args.save, fourcc, fps, (out_w, out_h), True)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    pbar = tqdm(total=total_frames, unit="f", desc="Processing") if total_frames > 0 else None

    with torch.no_grad():
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            base = frame_bgr.copy()

            # letterbox to 640x640 (BGR->RGB)
            img_lb, meta = letterbox(frame_bgr, 640)
            img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            tensor = tensor.to(args.device)

            # forward
            det_out, da_seg_out, ll_seg_out = model(tensor)

            # lane seg -> binary mask (0/1)
            # ll_seg_out is [B, 2, H', W'] (background, lane)
            lanes = torch.argmax(ll_seg_out, dim=1).squeeze(0).to("cpu").numpy().astype(np.uint8)

            # ---- ALIGNMENT FIX ----
            # A) Upsample mask to the letterboxed square (e.g., 640x640)
            box = meta["box_size"]  # 640
            lanes = cv2.resize(lanes, (box, box), interpolation=cv2.INTER_NEAREST)

            # B) Remove padding using true pads and resized dims
            pad_left, pad_top = meta["pad"]
            nw, nh = meta["resized_wh"]
            pad_right  = box - nw - pad_left
            pad_bottom = box - nh - pad_top
            lanes = lanes[pad_top: box - pad_bottom, pad_left: box - pad_right]

            # C) Resize back to original frame size
            lanes = cv2.resize(lanes, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

            # make a colored overlay for lanes
            overlay = base.copy()
            lane_color = (0, 255, 255)  # yellow
            overlay[lanes == 1] = lane_color

            blended = cv2.addWeighted(overlay, args.alpha, base, 1 - args.alpha, 0)

            writer.write(blended)
            if args.show:
                cv2.imshow("YOLOP Lane Detection", blended)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                    break

            if pbar:
                pbar.update(1)

    if pbar:
        pbar.close()
    cap.release()
    writer.release()
    if args.show:
        cv2.destroyAllWindows()
    print(f"Saved: {args.save}")

if __name__ == "__main__":
    main()
