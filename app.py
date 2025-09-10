import os
import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import base64

st.set_page_config(page_title="Text-to-Image (HF Inference API)", layout="centered")
st.title("🚀 Text → Image (Hugging Face Inference API)")

# Prompt + params
prompt = st.text_input("Enter an image prompt:", "A futuristic city skyline at sunset")
guidance_scale = st.slider("Guidance Scale", 1.0, 20.0, 7.5)
steps = st.slider("Inference Steps", 10, 50, 25)

# Choose model (server-side model hosted by HF)
# You can change to another HF model if you prefer.
MODEL_ID = "stabilityai/stable-diffusion-2"   # or "stabilityai/sd-xl-beta" or "stabilityai/sd-turbo"

# Get token from Streamlit secrets or environment
HF_TOKEN = None
if "HF_TOKEN" in st.secrets:
    HF_TOKEN = st.secrets["HF_TOKEN"]
else:
    HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    st.error("Hugging Face token not found. Add HF_TOKEN to Streamlit Secrets or set env var HF_TOKEN.")
    st.stop()

API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

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

def extract_image_from_response(resp):
    # If HF returns raw image bytes
    ct = resp.headers.get("content-type", "")
    if ct.startswith("image"):
        return Image.open(BytesIO(resp.content))

    # Otherwise try JSON (some endpoints return base64 in JSON)
    try:
        data = resp.json()
    except Exception:
        st.error(f"Unexpected response content-type: {ct}\nStatus: {resp.status_code}")
        return None

    if isinstance(data, dict) and data.get("error"):
        st.error(f"Hugging Face API error: {data.get('error')}")
        return None

    # Some HF endpoints return base64 list under "images" or similar
    if isinstance(data, dict) and "images" in data and len(data["images"]) > 0:
        img_b64 = data["images"][0]
        img_bytes = base64.b64decode(img_b64)
        return Image.open(BytesIO(img_bytes))

    # Try common keys containing base64
    for key in ("generated_images", "image", "images"):
        if key in data:
            val = data[key]
            if isinstance(val, list) and len(val) > 0:
                img_b64 = val[0]
                if isinstance(img_b64, str):
                    img_bytes = base64.b64decode(img_b64)
                    return Image.open(BytesIO(img_bytes))

    st.error("Could not decode image from HF API response.")
    return None

# Generate button
if st.button("Generate Image"):
    with st.spinner("Calling Hugging Face Inference API — this may take a few seconds..."):
        try:
            response = call_hf_api(prompt, steps=steps, guidance=guidance_scale)
            if response.status_code != 200:
                # Show helpful error messages
                try:
                    err = response.json()
                    st.error(f"HF API returned {response.status_code}: {err}")
                except Exception:
                    st.error(f"HF API returned status {response.status_code}: {response.text}")
            else:
                image = extract_image_from_response(response)
                if image:
                    st.image(image, use_container_width=True, caption="Generated Image")
                    # Download
                    buf = BytesIO()
                    image.save(buf, format="PNG")
                    st.download_button("Download PNG", buf.getvalue(), "generated.png", "image/png")
        except requests.exceptions.RequestException as e:
            st.error(f"Network/timeout error: {e}")
