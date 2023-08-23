import cv2
import streamlit as st
from torchvision import transforms
import torch
import torch.nn.functional as F
from streamlit_drawable_canvas import st_canvas
import numpy as np
import time

class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = torch.nn.Linear(7 * 7 * 64, 10, bias=True)

        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)   
        out = self.fc(out)
        return out

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=16 * 4 * 4, out_features=120),
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        probs = F.softmax(x, dim=1)
        return probs

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1) 
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.relu2 = nn.ReLU()


        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu3 = nn.ReLU()

        self.fc6 = nn.Linear(256*3*3, 1024) 
        self.fc7 = nn.Linear(1024, 512)
        self.fc8 = nn.Linear(512, 10)  

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool3(x)
        x = self.relu3(x)
        x = x.view(-1, 256 * 3 * 3)
        x = self.fc6(x)
        x = F.relu(x)
        x = self.fc7(x)
        x = F.relu(x)
        x = self.fc8(x)  
        return x

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



    
    
