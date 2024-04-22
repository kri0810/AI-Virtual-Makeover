import streamlit as st
import cv2
import numpy as np
import dlib

# Function to read image
def read_image(file_path):
    '''Read and return the image from a given file path'''
    img = cv2.imread(file_path)
    return img

# Function to get lip landmarks
def get_lip_landmark(img):
    '''Finding lip landmark and return list of corresponded coordinations'''
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Perform landmark detection for whole face
    faces = detector(gray_img)
    for face in faces:
        landmarks = predictor(gray_img, face)
        lmPoints = []
        # Obtain landmark coordinations for lips only, since lips landmark ranging from point 48 to 68
        for n in range(48,68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            lmPoints.append([x, y])
    return lmPoints

# Function to change lip color
def change_lip_color(img, color):
    '''Change lip color based on given color option'''
    img_original = img.copy()
    lm_points = get_lip_landmark(img_original)

    # Color options
    colors = {
        "1": (83, 55, 220),   # Punch
        "2": (126, 109, 229), # Rose
        "3": (14, 22, 139),   # Brick red
        "4": (106, 105, 184), # Dusty rose
        "5": (158, 172, 204), # Nude
        "6": (86, 52, 49)     # Royal blue
    }

    # Use default color (Red) if option not found
    selected_color = colors.get(color, (0, 0, 255))

    # Obtain exact coordination of the lips
    poly1 = np.array(lm_points[:12], np.int32).reshape((-1, 1, 2))
    poly2 = np.array(lm_points[12:], np.int32).reshape((-1, 1, 2))

    # Create a blank image to draw the colored lips on
    colored = np.zeros_like(img_original)

    # Fill in the color based on the coordination by fillPoly
    colored = cv2.fillPoly(colored, [poly1, poly2], selected_color)

    # Smoothen the image by GaussianBlur
    colored = cv2.GaussianBlur(colored, (7, 7), 0)

    # Blend colored lips and the original picture together
    result = cv2.addWeighted(colored, 0.3, img_original, 0.7, 0)

    return result

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
