import streamlit as st
from streamlit_image_select import image_select

from transformers import Blip2Processor, Blip2ForConditionalGeneration, BitsAndBytesConfig
from diffusers import AudioLDM2Pipeline
import torch

from PIL import Image

from peft import PeftModel

import accelerate

import torch

import scipy

from pathlib import Path
import io
   
device = torch.device("cuda" if torch.cuda.is_available else "cpu")

@st.cache_resource
def get_blip_model():
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )
    model = Blip2ForConditionalGeneration.from_pretrained(
        "ybelkada/blip2-opt-2.7b-fp16-sharded",
        device_map = "auto",
        quantization_config = quant_config
    )
    training_model = PeftModel.from_pretrained(model, "blip-lora")
    training_model.eval()

    return training_model

@st.cache_resource
def get_processor():
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    return processor

@st.cache_resource
def get_audio_pipe():
    pipe = AudioLDM2Pipeline.from_pretrained(
        "cvssp/audioldm2-music",
        torch_dtype=torch.float16
    )
    pipe = pipe.to(device)

    return pipe

@st.cache_data
def get_sample_music(image_path):
    _, wav_buffer = image_to_music(Image.open(image_path), num_steps=200, status=False)
    return wav_buffer

def image_to_music(image_input, num_steps=10, status=True):
    with st.status("Image to Music pipeline", expanded=status, state = "running" if status else "complete") as status:
        st.write("Captioning image...")
        processor = get_processor()
        inputs = processor(images=image_input, return_tensors="pt").to(device, torch.float32)
        pixel_values = inputs.pixel_values

        st.write("Translating description...")
        blip_model = get_blip_model()
        generated_ids = blip_model.generate(pixel_values=pixel_values, max_length=100)
        # generated text
        generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        st.write("Generating music...")
        pipe = get_audio_pipe()

        prompt = generated_caption
        audio = pipe(prompt, num_inference_steps=num_steps, audio_length_in_s=10.24).audios[0]

        status.update(expanded=False, state = "complete")
    # generated music
    wav_buffer = io.BytesIO()
    scipy.io.wavfile.write(wav_buffer, rate=16000, data=audio)

    return generated_caption, wav_buffer

st.title('Image to Music generator')

image_input = st.file_uploader('Upload an image', type=['png', 'jpg'])
if image_input is not None:
    st.image(image_input)

if image_input is None:
    sample_image_path = Path("sample_images")
    sample_image_input = image_select(
        label = "Select a sample image",
        images = [image_path for image_path in sample_image_path.iterdir()],
    )
    sample_music = get_sample_music(str(sample_image_input))
    st.audio(sample_music, format="audio/wav")

if st.button('Generate Music'):
    image = Image.open(image_input if image_input is not None else sample_image_input)
    caption, wav_buffer = image_to_music(image)
    st.caption(caption)
    st.audio(wav_buffer, format="audio/wav")
