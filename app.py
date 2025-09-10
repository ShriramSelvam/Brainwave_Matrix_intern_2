import os
import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import base64

# ----------------------------
# Streamlit Page Setup
# ----------------------------
st.set_page_config(page_title="Text-to-Image (HF Inference API)", layout="centered")
st.title("🚀 Text → Image (Hugging Face Inference API)")
st.markdown("Using **stabilityai/sd-turbo** for super-fast generation ⚡")

# ----------------------------
# User Input
# ----------------------------
prompt = st.text_input("Enter an image prompt:", "A futuristic city skyline at sunset")
guidance_scale = st.slider("Guidance Scale", 1.0, 20.0, 7.5)
steps = st.slider("Inference Steps", 5, 50, 25)

# ----------------------------
# Hugging Face API Setup
# ----------------------------
MODEL_ID = "stabilityai/sd-turbo"
HF_TOKEN = None

if "HF_TOKEN" in st.secrets:
    HF_TOKEN = st.secrets["HF_TOKEN"]
else:
    HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    st.error("❌ Hugging Face token not found. Add HF_TOKEN to Streamlit Secrets or set env var HF_TOKEN.")
    st.stop()

API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# ----------------------------
# API Call Function
# ----------------------------
def call_hf_api(prompt_text, steps=25, guidance=7.5):
    payload = {
        "inputs": prompt_text,
        "options": {"wait_for_model": True},
        "parameters": {
            "num_inference_steps": int(steps),
            "guidance_scale": float(guidance)
        }
    }
    resp = requests.post(API_URL, headers=HEADERS, json=payload, timeout=240)
    return resp

# ----------------------------
# Extract Image
# ----------------------------
def extract_image_from_response(resp):
    ct = resp.headers.get("content-type", "")
    if ct.startswith("image"):
        return Image.open(BytesIO(resp.content))

    try:
        data = resp.json()
    except Exception:
        st.error(f"Unexpected response content-type: {ct}\nStatus: {resp.status_code}")
        return None

    if isinstance(data, dict) and data.get("error"):
        st.error(f"Hugging Face API error: {data.get('error')}")
        return None

    if isinstance(data, dict) and "images" in data and len(data["images"]) > 0:
        img_b64 = data["images"][0]
        img_bytes = base64.b64decode(img_b64)
        return Image.open(BytesIO(img_bytes))

    st.error("❌ Could not decode image from HF API response.")
    return None

# ----------------------------
# Generate Button
# ----------------------------
if st.button("🎨 Generate Image"):
    with st.spinner("Calling Hugging Face Inference API... Please wait ⏳"):
        try:
            response = call_hf_api(prompt, steps=steps, guidance=guidance_scale)
            if response.status_code != 200:
                try:
                    err = response.json()
                    st.error(f"HF API returned {response.status_code}: {err}")
                except Exception:
                    st.error(f"HF API returned status {response.status_code}: {response.text}")
            else:
                image = extract_image_from_response(response)
                if image:
                    st.image(image, use_container_width=True, caption="Generated Image")
                    buf = BytesIO()
                    image.save(buf, format="PNG")
                    st.download_button("⬇️ Download PNG", buf.getvalue(), "generated.png", "image/png")
        except requests.exceptions.RequestException as e:
            st.error(f"Network/timeout error: {e}")
