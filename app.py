import streamlit as st

st.title('Images/Text to Music generator')

text_input = st.text_area('Enter text here')
image_input = st.file_uploader('Upload an image')

if st.button('Generate Music'):
    st.write('Generating music...')
    
    image_to_text_generated = "This is a generated text from image" #yq model output for image to text
    st.write(image_to_text_generated)
