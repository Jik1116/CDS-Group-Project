import streamlit as st
import sys
from transformers import ViTImageProcessor, RobertaTokenizer, Blip2Processor, Blip2ForConditionalGeneration, InstructBlipProcessor, InstructBlipForConditionalGeneration

from datasets import list_metrics

from diffusers import AudioLDMPipeline

import torch
from torch.utils.data import Dataset, DataLoader

from PIL import Image

from peft import LoraConfig, get_peft_model

import accelerate

from diffusers import AudioLDMPipeline
import torch

import scipy
import os
import shutil

st.title('Images/Text to Music generator')

text_input = st.text_area('Enter text here')
image_input = st.file_uploader('Upload an image')

if st.button('Generate Music'):
    st.write('Generating music...')
    
    image_to_text_generated = "This is a generated text from image" #yq model output for image to text
    st.write(image_to_text_generated)

def load_model():
   
   device = "cuda" if torch.cuda.is_available else "cpu"
   
   model = Blip2ForConditionalGeneration.from_pretrained("ybelkada/blip2-opt-2.7b-fp16-sharded", device_map=device, load_in_8bit=True, torch_dtype=torch.float32)
   
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
   processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
   image = Image.open("sky.jpg")

   inputs = processor(images=image, return_tensors="pt").to(device, torch.float32)
   pixel_values = inputs.pixel_values
   
   generated_ids = training_model.generate(pixel_values=pixel_values, max_length=100)
   # generated text
   generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
   
   repo_id = "cvssp/audioldm-s-full-v2"
   pipe = AudioLDMPipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
   pipe = pipe.to("cuda")
   
   prompt = generated_caption
   audio = pipe(prompt, num_inference_steps=10, audio_length_in_s=5.0).audios[0]
   # generated image
   scipy.io.wavfile.write(f"sky.wav", rate=16000, data=audio)
