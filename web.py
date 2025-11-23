import streamlit as st
import cv2
import numpy as np
import requests
import tempfile
import os
from pathlib import Path
from typing import List


st.set_page_config(page_title="Video Upload / URL Import", layout="wide")


def write_bytes_to_temp_file(content: bytes, suffix: str = ".mp4") -> Path:
	tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
	try:
		tmp.write(content)
		tmp.flush()
	finally:
		tmp.close()
	return Path(tmp.name)


def download_video_to_tempfile(url: str) -> Path:
	# download to a temporary file and return the path (caller must delete)
	r = requests.get(url, stream=True, timeout=15)
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
	frames = []
	total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
	step = max(1, max(1, total // max_frames)) if total > 0 else 1
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


st.title("Video Upload or Import")
st.markdown(
	"Upload a local video file or provide a direct video URL (mp4, mov, avi). The app previews the video and shows sample extracted frames."
)

with st.sidebar:
	st.header("Input")
	mode = st.radio("Select input method:", ["Upload file", "Video URL"])
	uploaded_file = None
	video_url = ""
	if mode == "Upload file":
		uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi", "mkv"], accept_multiple_files=False)
	elif mode == "Video URL":
		video_url = st.text_input("Direct video URL (http(s)://)")
	show_frames = st.checkbox("Extract and show sample frames", value=True)
	max_frames = st.slider("Max sample frames", 1, 12, 6)

video_path = None
video_source = None

if uploaded_file is not None:
	try:
		# read into memory for preview; do not persist to uploads/
		file_bytes = uploaded_file.read()
		video_source = "uploaded"
		# we'll write to a temp file only if we need frame extraction
		video_bytes = file_bytes
	except Exception as e:
		st.error(f"Failed to read uploaded file: {e}")

elif video_url:
	try:
		st.info("Downloading video from URL (this may take a few seconds)...")
		# download to a temp file for processing; will be deleted after use
		downloaded = download_video_to_tempfile(video_url)
		st.success(f"Downloaded to temporary file `{downloaded}` (will be removed after processing)")
		video_path = downloaded
		video_source = video_url
	except Exception as e:
		st.error(f"Failed to download video: {e}")

if (video_source == "uploaded") or (video_path is not None):
	# Preview using st.video (accepts URL or raw bytes/file)
	st.subheader("Preview")
	try:
		# If original input was an uploaded file, preview from bytes (no save)
		if video_source == "uploaded":
			st.video(video_bytes)
		# If input was a URL, pass the URL where possible (st.video can handle URLs)
		elif isinstance(video_source, str) and video_source.startswith("http"):
			st.video(video_source)
		# Otherwise preview from the temporary path
		elif video_path is not None:
			st.video(str(video_path))
	except Exception:
		# fallback: stream bytes if available
		try:
			if video_source == "uploaded":
				st.video(video_bytes)
			elif video_path is not None:
				with open(video_path, "rb") as f:
					st.video(f.read())
		except Exception as e:
			st.error(f"Cannot preview video: {e}")

	if show_frames:
		st.subheader("Sample frames")
		temp_to_delete = None
		try:
			if video_source == "uploaded":
				# write uploaded bytes to a temporary file for OpenCV processing
				temp_path = write_bytes_to_temp_file(video_bytes, suffix=Path(uploaded_file.name).suffix or ".mp4")
				temp_to_delete = temp_path
				frames = extract_sample_frames(temp_path, max_frames=max_frames)
			else:
				# video_path already points to a temporary downloaded file
				frames = extract_sample_frames(video_path, max_frames=max_frames)

			if len(frames) == 0:
				st.warning("No frames could be extracted from this video.")
			else:
				cols = st.columns(min(6, len(frames)))
				for i, frm in enumerate(frames):
					with cols[i % len(cols)]:
						st.image(frm, caption=f"Frame {i+1}", use_column_width=True)
		finally:
			# cleanup any temporary file created for processing
			if 'temp_to_delete' in locals() and temp_to_delete is not None and temp_to_delete.exists():
				try:
					os.remove(str(temp_to_delete))
				except Exception:
					pass
			# also remove downloaded URL file if present
			if video_path is not None and video_source and isinstance(video_source, str) and video_source.startswith('http'):
				try:
					if video_path.exists():
						os.remove(str(video_path))
				except Exception:
					pass
				
	st.markdown("---")
	st.write("Videos are not saved persistently by this app. Temporary files created for processing are removed after frame extraction.")
	if video_path is not None:
		st.write(f"Temporary path used: `{video_path}` (may have been deleted after processing)")
	else:
		st.write("Uploaded video was processed in-memory and not written to disk.")
else:
	st.info("No video selected yet. Upload a file or enter a direct video URL in the sidebar.")
