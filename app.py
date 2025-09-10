import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
import streamlit as st
import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from PIL import Image

# 🔧 Convert PIL image to bytes for download
def image_to_bytes(img):
    from io import BytesIO
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im
    
# 🧠 Fix known issues on some systems (e.g., PyTorch on Windows)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 📄 Page configuration
st.set_page_config(page_title="Text-to-Image Generator", layout="centered")
st.title("🖼️ AI Text-to-Image Generator")
st.markdown("Generate high-quality images from text prompts using Stable Diffusion.")

@st.cache_resource(show_spinner=True)
def load_pipeline():
    try:
        model_id = "runwayml/stable-diffusion-v1-5"

        # Use an efficient, compatible scheduler
        scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")

        # Use float32 for CPU compatibility
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            scheduler=scheduler,
            torch_dtype=torch.float32
        )

        pipe.to("cpu")  # force CPU mode for compatibility
        return pipe

    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None

pipe = load_pipeline()

# ✏️ Prompt input
prompt = st.text_input("Enter your image prompt:", "A fantasy landscape with castles and dragons")

# ⚙️ Parameters
guidance_scale = st.slider("Guidance Scale (Creativity vs Accuracy)", 1.0, 15.0, 7.5)
steps = st.slider("Inference Steps", 10, 50, 25)

# 🖼️ Generate image
if st.button("Generate Image") and pipe:
    with st.spinner("Generating image... Please wait (CPU may take 30–60s)"):
        try:
            result = pipe(prompt, guidance_scale=guidance_scale, num_inference_steps=steps)
            image: Image.Image = result.images[0]
            st.image(image, caption="🎨 Generated Image", use_container_width=True)

            # Add download button
            st.download_button(
                label="Download Image",
                data=image_to_bytes(image),
                file_name="generated_image.png",
                mime="image/png"
            )
        except Exception as e:
            st.error(f"❌ Error during image generation: {e}")
