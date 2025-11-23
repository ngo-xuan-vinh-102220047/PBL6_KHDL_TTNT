import streamlit as st
import cv2
import numpy as np
import requests
import tempfile
import os
from pathlib import Path
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

st.set_page_config(page_title="Video Upload / Model Inference", layout="wide")


def write_bytes_to_temp_file(content: bytes, suffix: str = ".mp4") -> Path:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(content)
        tmp.flush()
    finally:
        tmp.close()
    return Path(tmp.name)


def download_video_to_tempfile(url: str) -> Path:
    r = requests.get(url, stream=True, timeout=20)
    r.raise_for_status()
    suffix = Path(url).suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                tmp.write(chunk)
        tmp.flush()
    finally:
        tmp.close()
    return Path(tmp.name)


def extract_sample_frames(video_path: Path, max_frames: int = 6) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    frames: List[np.ndarray] = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, (total // max_frames) if total > 0 else 1)
    idx = 0
    grabbed = 0
    while grabbed < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            grabbed += 1
        idx += 1
    cap.release()
    return frames


def preprocess_frame(frame: np.ndarray, img_size=(224, 224)) -> np.ndarray:
    im = cv2.resize(frame, img_size)
    im = im.astype(np.float32) / 255.0
    im = np.transpose(im, (2, 0, 1))
    return im


def try_load_best_model(path: Path, device: torch.device):
    try:
        scripted = torch.jit.load(str(path), map_location=device)
        return scripted
    except Exception:
        pass

    ckpt = torch.load(str(path), map_location=device)
    if isinstance(ckpt, nn.Module):
        return ckpt

    if isinstance(ckpt, dict):
        if 'state_dict' in ckpt:
            state = ckpt['state_dict']
        elif 'model_state_dict' in ckpt:
            state = ckpt['model_state_dict']
        else:
            state = ckpt

        if isinstance(state, dict):
            class Fallback(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv = nn.Sequential(
                        nn.Conv2d(3, 32, 3, stride=2, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(32, 64, 3, stride=2, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(64, 3, 3, padding=1),
                        nn.Sigmoid(),
                    )

                def forward(self, x):
                    return self.conv(x)

            model = Fallback()
            try:
                model.load_state_dict(state)
                model.to(device)
                model.eval()
                return model
            except Exception as e:
                raise RuntimeError("Couldn't load state_dict into fallback model: " + str(e))

    raise RuntimeError("Unsupported checkpoint format. Provide a scripted model or a compatible state_dict.")


def run_single_frame_inference(model: nn.Module, frames: List[np.ndarray], device: torch.device):
    if len(frames) == 0:
        raise ValueError("No frames for inference")
    mid = frames[len(frames) // 2]
    x = preprocess_frame(mid, img_size=(224, 224))
    tx = torch.from_numpy(x).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tx)
    return out


def run_model_on_video(model: nn.Module, video_path: Path, device: torch.device,
                       frame_stride: int = 5, max_frames: int = 500, batch_size: int = 8):
    """
    Run model over frames of a video file.
    Processes every `frame_stride` frame up to `max_frames`, in batches.
    Returns a dict with timestamps (frame indices), scores (anomaly score per processed frame),
    and a small list of frames (RGB) corresponding to the processed frames.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Cannot open video for processing")

    scores = []
    timestamps = []
    processed_frames = []

    batch_inputs = []
    batch_frame_idxs = []
    idx = 0
    processed = 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_stride != 0:
            idx += 1
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        batch_inputs.append(preprocess_frame(rgb, img_size=(224, 224)))
        batch_frame_idxs.append(idx)

        if len(batch_inputs) >= batch_size or (processed + len(batch_inputs)) >= max_frames:
            try:
                tx = torch.from_numpy(np.stack(batch_inputs)).to(device)
                with torch.no_grad():
                    out = model(tx)
            except Exception as e:
                # inference error: break and return what we have
                cap.release()
                raise RuntimeError(f"Model inference failed: {e}")

            # interpret outputs per-sample
            def interpret(out_tensor, inputs_tensor):
                # returns list of scalar scores (higher -> more anomalous)
                if isinstance(out_tensor, torch.Tensor):
                    o = out_tensor.detach().cpu()
                    inp = inputs_tensor.detach().cpu()
                    if o.dim() == 4 and o.shape[1] in (1, 3):
                        # predicted images: compute MSE per sample vs input
                        # ensure shapes: (B,C,H,W)
                        mse = ((o - inp) ** 2).mean(dim=(1, 2, 3)).numpy().tolist()
                        return mse
                    elif o.dim() == 2:
                        # class logits: score = 1 - max_prob
                        probs = F.softmax(o, dim=1)
                        maxp, _ = probs.max(dim=1)
                        scores_list = (1.0 - maxp).cpu().numpy().tolist()
                        return scores_list
                    elif o.dim() == 1 or o.numel() == o.shape[0]:
                        return o.squeeze().cpu().numpy().tolist()
                    else:
                        # fallback: use L2 norm per-sample
                        norms = o.reshape(o.shape[0], -1).norm(p=2, dim=1).cpu().numpy().tolist()
                        return norms
                else:
                    return [0.0] * inputs_tensor.shape[0]

            scores_batch = interpret(out, tx)
            # store results and sample frames
            for i, frame_idx in enumerate(batch_frame_idxs):
                timestamps.append(frame_idx)
                scores.append(float(scores_batch[i]))
                # store small RGB preview
                img = (np.clip(np.transpose(batch_inputs[i], (1, 2, 0)) * 255.0, 0, 255)).astype(np.uint8)
                processed_frames.append(img)

            processed += len(batch_inputs)
            batch_inputs = []
            batch_frame_idxs = []

        if processed >= max_frames:
            break

        idx += 1

    cap.release()
    return {"timestamps": timestamps, "scores": scores, "frames": processed_frames, "total_frames": total_frames}


def main():
    st.title("Video Upload / URL + Model Inference")

    with st.sidebar:
        st.header("Input")
        mode = st.radio("Input method", ["Upload file", "Video URL"])
        uploaded_file = None
        video_url = ""
        if mode == "Upload file":
            uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi", "mkv"]) 
        # else:
        #     video_url = st.text_input("Direct video URL (http(s)://)")

        # Auto-detection settings (default ON)
        auto_detect = st.checkbox("Auto-detect with model (process video frames)", value=True)
        process_stride = st.slider("Process every Nth frame", 1, 10, 5)
        max_process = st.slider("Max frames to process", 10, 1000, 200)

        # Optional: also show sampled frames for visualization
        show_sample_frames = st.checkbox("Also show sampled frames (for debugging)", value=False)
        sample_frames = st.slider("Sample frames to display", 1, 12, 6)

    video_path: Optional[Path] = None
    video_bytes: Optional[bytes] = None

    if uploaded_file is not None:
        try:
            video_bytes = uploaded_file.read()
        except Exception as e:
            st.error(f"Failed reading uploaded file: {e}")
    elif video_url:
        try:
            st.info("Downloading video from URL...")
            video_path = download_video_to_tempfile(video_url)
            st.success("Downloaded temporary video for processing")
        except Exception as e:
            st.error(f"Download failed: {e}")

    if (video_bytes is not None) or (video_path is not None):
        st.subheader("Preview")
        try:
            if video_bytes is not None:
                st.video(video_bytes)
            elif video_path is not None:
                st.video(str(video_path))
        except Exception:
            st.warning("Preview not available in this environment")

        # Automatic model-based detection
        temp_to_delete: Optional[Path] = None
        try:
            p_best = Path("best_model_pytorch.pth")
            if not p_best.exists():
                st.info("Place `best_model_pytorch.pth` in the app folder to enable inference.")
            else:
                run_best = st.checkbox("Run `best_model_pytorch.pth`", value=False)
                if run_best:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    try:
                        model = try_load_best_model(p_best, device)
                    except Exception as e:
                        st.error(f"Model load failed: {e}")
                        model = None

                    if model is not None:
                        st.success("Model loaded")

                        # if uploaded bytes, write to temp for processing
                        if video_bytes is not None:
                            temp = write_bytes_to_temp_file(video_bytes, suffix=Path(getattr(uploaded_file, 'name', '') or '.mp4').suffix)
                            temp_to_delete = temp
                            vpath = temp
                        else:
                            vpath = video_path

                        if vpath is None:
                            st.error("No video available for processing")
                        else:
                            if auto_detect:
                                st.subheader("Auto-detection results")
                                progress = st.progress(0)
                                try:
                                    results = run_model_on_video(model, vpath, device,
                                                                 frame_stride=process_stride,
                                                                 max_frames=max_process,
                                                                 batch_size=8)
                                except Exception as e:
                                    st.error(f"Detection failed: {e}")
                                    results = None

                                if results is not None:
                                    # show simple chart of scores
                                    if len(results["timestamps"]) > 0:
                                        import pandas as pd
                                        df = pd.DataFrame({"frame_idx": results["timestamps"], "score": results["scores"]})
                                        df = df.sort_values("frame_idx")
                                        st.line_chart(df.set_index("frame_idx")["score"])

                                        # show top anomaly frames
                                        top_k = min(6, len(results["scores"]))
                                        order = np.argsort(results["scores"])[::-1][:top_k]
                                        cols = st.columns(min(6, top_k))
                                        for i, idx_top in enumerate(order):
                                            with cols[i % len(cols)]:
                                                st.image(results["frames"][int(idx_top)], caption=f"Frame {results['timestamps'][int(idx_top)]}\nscore={results['scores'][int(idx_top)]:.6f}")
                                    else:
                                        st.info("No frames were processed by the model.")

                            # optionally show sampled frames for visualization
                            if show_sample_frames:
                                st.subheader("Sample frames (visualization)")
                                frames = extract_sample_frames(vpath, max_frames=sample_frames)
                                if len(frames) == 0:
                                    st.warning("No frames extracted for display")
                                else:
                                    cols = st.columns(min(6, len(frames)))
                                    for i, f in enumerate(frames):
                                        with cols[i % len(cols)]:
                                            st.image(f, caption=f"Frame {i+1}")
        finally:
            if temp_to_delete is not None:
                try:
                    os.remove(str(temp_to_delete))
                except Exception:
                    pass
            if video_path is not None:
                try:
                    if video_path.exists():
                        os.remove(str(video_path))
                except Exception:
                    pass

        st.markdown("---")
        st.info("Videos are not saved persistently. Temporary files used for processing are removed.")
    else:
        st.info("No video selected yet. Upload a file or enter a video URL in the sidebar.")


if __name__ == '__main__':
    main()
