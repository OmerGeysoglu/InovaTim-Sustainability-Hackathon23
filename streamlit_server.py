import streamlit as st
from PIL import Image
from predict_img import BeerDetector


st.title('Fill or Empty')

# Dosya yükleyici widget
uploaded_file = st.file_uploader("Upload a File", type=['png', 'jpg', 'jpeg'])
detectorI = BeerDetector("model/img/best.pt")
button = st.button("Detect the image")
col1, col2 = st.columns(2)


# Yüklenen dosyayı kontrol etme
if uploaded_file is not None:
    # Dosyanın içeriğini okuma
    if 'image' in uploaded_file.type:
        try:
            image = Image.open(uploaded_file)
            with col1: 
                st.image(image, caption='The Uploaded Photograph')
            
            if button:
                detected_image = detectorI(image)
                with col2:
                    st.image(detected_image, caption="The Detected Photograph", use_column_width="True")
        except Exception as e:
            st.error(f"Hata oluştu: {e}")

    st.write("Please upload a Photograph")
