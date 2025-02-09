import streamlit as st
import tensorflow as tf
import numpy as np

def model_prediction(test_image):
    model = tf.keras.models.load_model('my_model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128,3))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    return np.argmax(predictions)


st.sidebar.title("Plant Disease Detector for Agriculture Field")
app_mode= st.sidebar.selectbox('select page',['Home','Disease Recognition'])

from PIL import Image

img = Image.open('Diseases.png')
st.image(img)

# Main Page
if(app_mode == 'HOME'):
    st.markdown("<h1 style='text-align : center;'> Plant Disease Detection System for Sustainable Agriculture", unsafe_allow_html=True )
# Prediction Page
elif(app_mode == 'Disease Recognition'):
    st.header("Plant Disease Detection System For Sustainable Agriculture")


# Recognition
test_image = st.file_uploader('Choose an image:')
if(st.button('Show Image')):
    st.image(test_image,width=4,use_container_width=True)

if(st.button('Predict')):
    st.snow()
    st.write('our prediction')
    result_index = model_prediction(test_image)

    # Class names
    class_name = ['Potato__Early_blight','Potato__Late_blight','Potato__healthy']    
    st.success('Model is Predicting its a {}'.format(class_name[result_index]))