import streamlit as st
import torch
from diffusers import DiffusionPipeline
from PIL import Image

st.set_page_config(page_title="Text to Image Generator", layout="centered")
st.title("🖼️ Text to Image Generator")

@st.cache_resource
def load_model():
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        use_safetensors=True
    )
    if torch.cuda.is_available():
        pipe.to("cuda")
    return pipe

pipe = load_model()

prompt = st.text_input("Enter your prompt:", "A serene landscape with a mountain lake")
if st.button("Generate"):
    with st.spinner("Generating image..."):
        image = pipe(prompt).images[0]
        st.image(image, caption="Generated Image", use_column_width=True)
