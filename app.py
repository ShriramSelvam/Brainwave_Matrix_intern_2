import os
import streamlit as st
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

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

# 📄 Page config
st.set_page_config(page_title="Text-to-Image Generator", layout="centered")
st.title("🖼️ AI Text-to-Image Generator")
st.markdown("Generate images from text prompts using Stable Diffusion (demo-friendly).")

# ✏️ Prompt input
prompt = st.text_input("Enter your image prompt:", "A fantasy landscape with castles and dragons")

# ⚙️ Parameters
guidance_scale = st.slider("Guidance Scale", 1.0, 15.0, 7.5)
steps = st.slider("Inference Steps", 10, 50, 25)

# Lazy load model (only when needed)
@st.cache_resource(show_spinner=True)
def load_pipeline():
    model_id = "runwayml/stable-diffusion-v1-5"

    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        scheduler=scheduler,
        torch_dtype=torch.float32
    )

    # Streamlit Cloud has no GPU → force CPU
    pipe.to("cpu")
    return pipe

# 🖼️ Generate
if st.button("Generate Image"):
    with st.spinner("Loading model & generating image (may take 1–2 mins on first run)..."):
        try:
            pipe = load_pipeline()
            result = pipe(prompt, guidance_scale=guidance_scale, num_inference_steps=steps)
            image: Image.Image = result.images[0]

            # Show result
            st.image(image, caption="🎨 Generated Image", use_container_width=True)

            # Download option
            st.download_button(
                label="Download Image",
                data=image_to_bytes(image),
                file_name="generated_image.png",
                mime="image/png"
            )
        except Exception as e:
            st.error(f"❌ Error: {e}")
