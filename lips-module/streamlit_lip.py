import streamlit as st
from utils import read_image, get_lip_landmark, change_lip_color
import numpy as np
import cv2

# Streamlit app
def main():
    st.title("Lip Color Changer")

    # File uploader
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Read uploaded image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        original_img = cv2.imdecode(file_bytes, 1)

        # Display original image
        st.image(original_img, caption="Original Image", use_column_width=True)

        # Select color option
        color_option = st.selectbox("Select a color option:",
                                    ["Punch", "Rose", "Brick red", "Dusty rose", "Nude", "Royal blue"])

        # Convert color option to corresponding number
        color_dict = {"Punch": "1", "Rose": "2", "Brick red": "3", "Dusty rose": "4", "Nude": "5", "Royal blue": "6"}
        color_num = color_dict[color_option]

        # Change lip color
        modified_img = change_lip_color(original_img, color_num)

        # Display modified image
        st.image(modified_img, caption="Modified Image", use_column_width=True)

if __name__ == "__main__":
    main()
