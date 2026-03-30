import streamlit as st
from ultralytics import YOLO
from PIL import Image

# 1. Load your trained model (update this path to your best.pt file)
@st.cache_resource
def load_model():
    return YOLO("best.pt") 

model = load_model()

st.title("Object Detection")

# 2. Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Image", use_container_width=True)
    
    st.write("Detecting...")
    
    # 3. Run inference
    results = model(image)
    
    # 4. Extract the plotted image (with bounding boxes)
    res_plotted = results[0].plot()
    
    # Ultralytics returns BGR arrays, convert to RGB for Streamlit
    res_rgb = res_plotted[..., ::-1]
    
    st.image(res_rgb, caption="Detected Objects", use_container_width=True)