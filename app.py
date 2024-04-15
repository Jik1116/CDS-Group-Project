import streamlit as st
from transformers import Blip2Processor, Blip2ForConditionalGeneration, BitsAndBytesConfig
from diffusers import AudioLDM2Pipeline

import torch

from PIL import Image

from peft import LoraConfig, get_peft_model

import accelerate

import torch

import scipy
   
device = torch.device("cuda" if torch.cuda.is_available else "cpu")

@st.cache_resource
def get_blip_model():
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )
    model = Blip2ForConditionalGeneration.from_pretrained(
        "ybelkada/blip2-opt-2.7b-fp16-sharded",
        device_map = device,
        quantization_config = quant_config
    )
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj"]
    )

    training_model = get_peft_model(model, config)
    state_dict = torch.load("model1.pt")
    training_model.load_state_dict(state_dict)
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


def image_to_music(image_input):
    with st.status("Image to Music pipeline", expanded=True) as status:
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
        audio = pipe(prompt, num_inference_steps=10, audio_length_in_s=10.2).audios[0]

        status.update(expanded=False)
    # generated music
    scipy.io.wavfile.write(f"output1.wav", rate=16000, data=audio)

    return generated_caption

st.title('Images/Text to Music generator')

text_input = st.text_area('Enter text here')
image_input = st.file_uploader('Upload an image', type=['png', 'jpg'])

if st.button('Generate Music'):
    image = Image.open(image_input)
    caption = image_to_music(image)
    st.write(caption)
    st.audio("output1.wav", format="audio/wav")
