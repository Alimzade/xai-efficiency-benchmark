"""
Media Components Module
Handles rendering of images, lightboxes, galleries, and base64 encoding/decoding for displaying images in Streamlit.
"""
import os
import base64
import requests
import streamlit as st
from PIL import Image
from io import BytesIO

from utils.helpers import rerun_app

def get_base64(img):
    buffered = BytesIO(); img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def get_image_thumbnail_base64(img_path, size=(24, 24)):
    if not img_path or not os.path.exists(img_path):
        return ""
    try:
        
        with Image.open(img_path) as img:
            img_rgb = img.convert("RGB")
            
            # Crop to center square to avoid aspect ratio distortion
            width, height = img_rgb.size
            min_dim = min(width, height)
            left = (width - min_dim) // 2
            top = (height - min_dim) // 2
            right = left + min_dim
            bottom = top + min_dim
            
            img_square = img_rgb.crop((left, top, right, bottom))
            img_square.thumbnail(size, Image.LANCZOS)
            
            buffer = BytesIO()
            img_square.save(buffer, format="JPEG", quality=85)
            encoded = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/jpeg;base64,{encoded}"
    except Exception:
        return ""

@st.dialog("Image Viewer", width="large")
def show_lightbox(img):
    st.markdown(f'<div style="display: flex; justify-content: center;"><img src="data:image/png;base64,{st.session_state.current_img_base64}" style="max-height: 80vh; max-width: 100%; object-fit: contain;"></div>', unsafe_allow_html=True)

def render_image_preview_gallery(img_sources):
    if img_sources:
        num_imgs = len(img_sources)
        if st.session_state.img_idx >= num_imgs:
            st.session_state.img_idx = 0
        current_src = img_sources[st.session_state.img_idx]
        try:
            if hasattr(current_src, 'name'):
                img_view = Image.open(current_src)
                src_name = f"Uploaded: {current_src.name}"
            elif isinstance(current_src, str) and (current_src.startswith("http://") or current_src.startswith("https://")):
                response = requests.get(current_src)
                img_view = Image.open(BytesIO(response.content))
                src_name = "URL Link"
            else:
                img_view = Image.open(current_src)
                src_name = f"Local: {os.path.basename(current_src)}"
            st.session_state.current_img_base64 = get_base64(img_view)
            
            html_content = f"""
            <div class="compact-preview">
                <div style='text-align: center; color: #94a3b8; font-size: 0.8em; margin-bottom: 2px;'>Resolution: {img_view.size[0]}x{img_view.size[1]} px | {src_name}</div>
                <div class="preview-image-frame"><img src="data:image/png;base64,{st.session_state.current_img_base64}" alt="Selected input preview"></div>
            </div>
            """
            st.markdown(html_content, unsafe_allow_html=True)
            st.markdown('<div style="margin-top: 8px;"></div>', unsafe_allow_html=True)
            if st.button("View", use_container_width=True):
                show_lightbox(img_view)
        except Exception:
            st.markdown("<div class='compact-preview'><div style='height: 330px; text-align: center; padding-top: 100px; color: #94a3b8;'>Preview unavailable</div></div>", unsafe_allow_html=True)
        n1, n2, n3 = st.columns([1, 0.8, 1])
        with n1:
            if st.button("⬅️ Prev", key="prev_btn", use_container_width=True):
                st.session_state.img_idx = (st.session_state.img_idx - 1) % num_imgs
                rerun_app()
        with n2:
            st.markdown(f"<div style='text-align: center; padding-top: 5px; font-weight: bold;'>{st.session_state.img_idx + 1}/{num_imgs}</div>", unsafe_allow_html=True)
        with n3:
            if st.button("Next ➡️", key="next_btn", use_container_width=True):
                st.session_state.img_idx = (st.session_state.img_idx + 1) % num_imgs
                rerun_app()
