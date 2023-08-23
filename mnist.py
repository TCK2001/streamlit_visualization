# import cv2
# from tensorflow.keras.models import load_model
# from streamlit_drawable_canvas import st_canvas
# import numpy as np
# import time
import streamlit as st
from MNIST_CNN import MNIST

name = st.sidebar.selectbox('Model', ['MNIST', 'CIFAR-10', 'CIFAR-100', 'ImageNet'])

if name == 'MNIST':
    MNIST()
# elif name == "CIFAR-10":
#     MNIST()