import cv2 as cv
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from PIL import Image
from kokoro import KPipeline
import soundfile as sf

st.set_page_config(page_title="Image Captioning AI", page_icon="📷")
st.markdown("""
    <style>
    .main { background-color: #f5f5f5; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #00acee; color: white; }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource 
def load_assets():
    model_cnn = tf.keras.models.load_model('./project_k/featur_extract.h5')
    model_rnn = tf.keras.models.load_model('./project_k/model_rnn.keras')
    with open('./project_k/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    pipeline = KPipeline(lang_code='a')
    return model_cnn, model_rnn, tokenizer,pipeline

try:
    model_cnn, model_rnn, tokenizer,tts_pipeline = load_assets()

    index_word = {v: k for k, v in tokenizer.word_index.items()}
    MAX_LEN = 40
except Exception as e:
    st.error(f"error: {e}")
    st.stop()

# --- واجهة المستخدم ---
st.title("🤖 AI Image Captioning")
st.write("upload image and will get descraption")

uploaded_file = st.file_uploader("select image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # عرض الصورة

    image = Image.open(uploaded_file)
    st.image(image, caption='caption images', use_container_width=True)
    
    if st.button('genration'):
        with st.spinner('waiting'):
            try:
                img_cv = np.array(image.convert('RGB'))
                img_cv = cv.resize(img_cv, (224, 224))
                img_input = preprocess_input(img_cv)
                img_input = np.expand_dims(img_input, axis=0)

                feature = model_cnn.predict(img_input, verbose=0)

             
                in_text = '<start>'
                for i in range(MAX_LEN):
                    sequence = tokenizer.texts_to_sequences([in_text])[0]
                    sequence = pad_sequences([sequence], maxlen=MAX_LEN, padding='post')
                    
                    yhat = model_rnn.predict([feature, sequence], verbose=0)
                    yhat = np.argmax(yhat)
                    word = index_word.get(yhat)
                    
                    if word is None or word == 'end':
                        break
                    
                    in_text += ' ' + word
                final_caption = in_text.replace('<start>', '').replace('end', '.').strip()
                
                st.success('done')
                st.markdown('captions')
                st.info(f"**{final_caption.capitalize()}**")

                generator = tts_pipeline(final_caption, voice='af_heart', speed=1)
                audio_chunks = []
                for _, _, audio in generator:
                    audio_chunks.append(audio)
                
                if audio_chunks:
                    final_audio = np.concatenate(audio_chunks)
                    st.audio(final_audio, sample_rate=24000)
                    
            except Exception as e:
                st.error(f"there is some thing wrong {e}")


st.markdown("---")
st.caption("Streamlit & TensorFlow")