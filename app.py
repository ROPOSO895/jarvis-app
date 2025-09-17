import streamlit as st
from transformers import pipeline
from gtts import gTTS
import tempfile
import torch
from diffusers import StableDiffusionPipeline

st.set_page_config(page_title="🤖 Jarvis 2.0", layout="wide")

# 1. Chatbot (Blenderbot - free HF model)
chatbot = pipeline("conversational", model="facebook/blenderbot-400M-distill")

def chat_response(message):
    response = chatbot(message)
    return response[0]['generated_text']

def chat_with_voice(message):
    text = chat_response(message)
    tts = gTTS(text)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        audio_path = fp.name
    return text, audio_path

# 2. Image Generator (Stable Diffusion)
@st.cache_resource
def load_sd():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sd_model = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base", torch_dtype=torch.float16
    )
    return sd_model.to(device)

sd_model = load_sd()

def generate_image(prompt):
    image = sd_model(prompt).images[0]
    return image

# ------------------- UI -------------------
st.title("🤖 Jarvis 2.0 - Multi AI Assistant")

tab1, tab2, tab3 = st.tabs(["💬 Chatbot", "🎙️ Voice Jarvis", "🎨 Image Generator"])

with tab1:
    st.subheader("💬 Chat with Jarvis")
    user_input = st.text_input("Type your message")
    if st.button("Send", key="chat"):
        if user_input:
            reply = chat_response(user_input)
            st.success(reply)

with tab2:
    st.subheader("🎙️ Jarvis with Voice")
    user_input = st.text_input("Say something to Jarvis")
    if st.button("Speak", key="voice"):
        if user_input:
            text, audio = chat_with_voice(user_input)
            st.success(text)
            st.audio(audio)

with tab3:
    st.subheader("🎨 Image Generator")
    prompt = st.text_input("Enter prompt for image")
    if st.button("Generate Image", key="img"):
        if prompt:
            img = generate_image(prompt)
            st.image(img, caption=prompt)
