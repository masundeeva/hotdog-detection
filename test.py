import streamlit as st
import tensorflow as tf
from PIL import Image
from skimage.transform import resize
import numpy as np
import time

loaded_model = tf.keras.models.load_model('model_tf.h5')  

# Prediction
def sigmoid(z):
    
    s = 1/(1+np.exp(-z))
    
    return s

def predict(w, b, X): # takes image and makes prediction
    
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    
    # probability of hotdog pic
    
    Y_prediction = sigmoid((np.dot(w.T, X)+ b))
         
    return Y_prediction

# Interface
st.title('Is this a hotdog ??')
st.write('\n')

# Default image
image = Image.open('preds.png')
show = st.image(image, use_column_width=True)

st.sidebar.title("Your pic here")

# User input image
uploaded_file = st.sidebar.file_uploader(" ",type=['png', 'jpg', 'jpeg'] )

if uploaded_file is not None:
    
    u_img = Image.open(uploaded_file) # PIL to open an image
    show.image(u_img, 'Uploaded Image', use_column_width=True) # displaying image
 
    image = np.asarray(u_img)/255 # conversion to array and normalizing pixel value
    
    my_image = resize(image, (128,128)).reshape((1, 128*128*3)).T # reshaping to our model
  
# newline
st.sidebar.write('\n')
    
if st.sidebar.button("Is my pic a hotdog??"):
    
    if uploaded_file is None:
        
        st.sidebar.write("Please upload an Image to Classify")
    
    else:
        
        with st.spinner('Classifying ...'):
            
            prediction = predict(loaded_model["w"], loaded_model["b"], my_image)
            time.sleep(2)
            st.success('Done!')
            
        st.sidebar.header("Algorithm Predicts: ")
        
        #Formatted probability value to 3 decimal places
        probability = "{:.3f}".format(float(prediction*100))
        
        # Classify cat being present in the picture if prediction > 0.5
        
        if prediction > 0.5:
            
            st.sidebar.write("It's a 'Hotdog", '\n' )
            
            st.sidebar.write('**Probability: **',probability,'%')
            
                             
        else:
            st.sidebar.write(" It's not a Hotdog!! ",'\n')
            
            st.sidebar.write('**Probability: **',probability,'%')
    