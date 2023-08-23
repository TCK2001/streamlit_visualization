import cv2
import streamlit as st
from torchvision import transforms
import torch
import torch.nn.functional as F
from streamlit_drawable_canvas import st_canvas
import numpy as np
import time

from model import CNN, LeNet5, AlexNet

def MNIST():
    
    MNIST_option = st.selectbox(
        'What model do you want to try?',
        ('LeNet-5', 'AlexNet', 'CNN'))
    
    @st.cache_resource
    def load(MNIST_option):
        loaded_model = None
        
        if MNIST_option == 'CNN':
            loaded_model = CNN()
            loaded_model.load_state_dict(torch.load('./models/simplecnn.pth'))
            loaded_model.eval()
          
        elif MNIST_option == 'LeNet-5':
            checkpoint = torch.load('./models/lenet5.pth')
            loaded_model = LeNet5()
            loaded_model.load_state_dict(checkpoint['model_state_dict'])
            loaded_model.eval()
        
        else: 
            loaded_model = AlexNet()  
            loaded_model.load_state_dict(torch.load('./models/alex.pth')) 
            loaded_model.eval()
            
        st.success("Loaded model!", icon="✅")
        
        return loaded_model
    
    def predict(model, image):
        input_tensor = transforms.ToTensor()(image) # C H W
        input_batch = input_tensor.unsqueeze(0) # B C H W

        with torch.no_grad():
            output = model(input_batch)
            
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
        return probabilities
    
    
    model = load(MNIST_option)
    
    st.write('Now your model is :', MNIST_option)

    st.write('# 🔥MNIST Recognizer🔥' )

    CANVAS_SIZE = 192

    with st.chat_message("user"):
        st.write("Hello 👋 draw the digits between 0 ~ 9 !")
        col1, col2 = st.columns(2)
        with col1:
            canvas = st_canvas(
                fill_color='black',       
                stroke_width=20,              
                stroke_color='white',    
                background_color='black', 
                width=CANVAS_SIZE,          
                height=CANVAS_SIZE,         
                drawing_mode='freedraw',    
                key='canvas'
            )
    
    if canvas.image_data is not None:
        img = canvas.image_data.astype(np.uint8)
        
        img = cv2.resize(img, dsize=(28, 28))
            
        preview_img = cv2.resize(img, dsize=(CANVAS_SIZE+5, CANVAS_SIZE+5), interpolation=cv2.INTER_NEAREST)
        col2.image(preview_img)
        
        x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        st.title('Custom model demo')
        
        y = predict(model, x)
        
        with st.chat_message("assistant"):
            st.write("Let's see the result !👋")
            with st.spinner('Wait for it...'):
                time.sleep(2)

            st.write('## Result : %d' % np.argmax(y).item())
            st.success('Is it correct..?!')
            st.bar_chart(y)
                
    txt = '''More informations : \n 
    > https://medium.com/@hichengkang \n 
    > https://tck2001.github.io/ \n'''
    st.write('TCK :', txt)



    
    
