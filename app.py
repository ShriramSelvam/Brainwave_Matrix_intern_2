import os
import streamlit as st
import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from PIL import Image

# 🔧 Convert PIL image to bytes for download
def image_to_bytes(img):
    from io import BytesIO
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# 🧠 Fix known issues on some systems
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

# 📄 Page configuration
st.set_page_config(page_title="Text-to-Image Generator", layout="centered")
st.title("🖼️ AI Text-to-Image Generator")
st.markdown("Generate images from text prompts using Stable Diffusion.")

@st.cache_resource(show_spinner=True)
def load_pipeline():
    try:
        # ✅ Detect Streamlit Cloud (limited RAM)
        is_cloud = os.environ.get("STREAMLIT_RUNTIME") is not None

        if is_cloud:
            model_id = "stabilityai/sd-turbo"   # lightweight model
        else:
            model_id = "runwayml/stable-diffusion-v1-5"  # full model

        scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")

        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            scheduler=scheduler,
            torch_dtype=torch.float32
        )

        pipe.to("cpu")  # CPU mode for compatibility
        return pipe

    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None

pipe = load_pipeline()

# ✏️ Prompt input
prompt = st.text_input("Enter your image prompt:", "A futuristic city skyline at sunset")

# ⚙️ Parameters
guidance_scale = st.slider("Guidance Scale", 1.0, 15.0, 7.5)
steps = st.slider("Inference Steps", 5, 30, 15)

# 🖼️ Generate image
if st.button("Generate Image") and pipe:
    with st.spinner("Generating image... Please wait..."):
        try:
            result = pipe(prompt, guidance_scale=guidance_scale, num_inference_steps=steps)
            image: Image.Image = result.images[0]
            st.image(image, caption="🎨 Generated Image", use_container_width=True)

            st.download_button(
                label="Download Image",
                data=image_to_bytes(image),
                file_name="generated_image.png",
                mime="image/png"
            )
        except Exception as e:
            st.error(f"❌ Error during image generation: {e}")
