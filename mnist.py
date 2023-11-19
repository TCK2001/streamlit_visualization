import cv2
import streamlit as st
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from streamlit_drawable_canvas import st_canvas
import numpy as np
import time
from PIL import Image
import math

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

first_HL =8
class Spinalnet(nn.Module):
  def __init__(self):
      super(Spinalnet, self).__init__()
      self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
      self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
      self.conv2_drop = nn.Dropout()
      self.fc1 = nn.Linear(160, first_HL) #changed from 16 to 8
      self.fc1_1 = nn.Linear(160 + first_HL, first_HL) #added
      self.fc1_2 = nn.Linear(160 + first_HL, first_HL) #added
      self.fc1_3 = nn.Linear(160 + first_HL, first_HL) #added
      self.fc1_4 = nn.Linear(160 + first_HL, first_HL) #added
      self.fc1_5 = nn.Linear(160 + first_HL, first_HL) #added
      self.fc2 = nn.Linear(first_HL*6, 10) # changed first_HL from second_HL

  def forward(self, x):
      x = F.relu(F.max_pool2d(self.conv1(x), 2))
      x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
      x = x.view(-1, 320)

      x1 = x[:, 0:160]
      x1 = F.relu(self.fc1(x1))
      x2= torch.cat([ x[:,160:320], x1], dim=1)
      x2 = F.relu(self.fc1_1(x2))
      x3= torch.cat([ x[:,0:160], x2], dim=1)
      x3 = F.relu(self.fc1_2(x3))
      x4= torch.cat([ x[:,160:320], x3], dim=1)
      x4 = F.relu(self.fc1_3(x4))
      x5= torch.cat([ x[:,0:160], x4], dim=1)
      x5 = F.relu(self.fc1_4(x5))
      x6= torch.cat([ x[:,160:320], x5], dim=1)
      x6 = F.relu(self.fc1_5(x6))

      x = torch.cat([x1, x2], dim=1)
      x = torch.cat([x, x3], dim=1)
      x = torch.cat([x, x4], dim=1)
      x = torch.cat([x, x5], dim=1)
      x = torch.cat([x, x6], dim=1)

      x = self.fc2(x)
      return F.log_softmax(x)


class Bottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, in_channels, growth_rate):
        super(Bottleneck, self).__init__()
        zip_channels = self.expansion * growth_rate
        self.features = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, zip_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(zip_channels),
            nn.ReLU(True),
            nn.Conv2d(zip_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        )
        
    def forward(self, x):
        out = self.features(x)
        out = torch.cat([out, x], 1)
        return out        
      
class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.features = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(2)
        )
        
    def forward(self, x):
        out = self.features(x)
        return out
    
class DenseNet(nn.Module):
    def __init__(self, num_blocks, growth_rate=12, reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate
        self.reduction = reduction
        
        num_channels = 2 * growth_rate
        
        self.features = nn.Conv2d(3, num_channels, kernel_size=3, padding=1, bias=False)
        self.layer1, num_channels = self._make_dense_layer(num_channels, num_blocks[0])
        self.layer2, num_channels = self._make_dense_layer(num_channels, num_blocks[1])
        self.layer3, num_channels = self._make_dense_layer(num_channels, num_blocks[2])
        self.layer4, num_channels = self._make_dense_layer(num_channels, num_blocks[3], transition=False)
        self.avg_pool = nn.Sequential(
            nn.BatchNorm2d(num_channels),
            nn.ReLU(True),
            nn.AvgPool2d(4),
        )
        self.classifier = nn.Linear(num_channels, num_classes)
        
        self._initialize_weight()
        
    def _make_dense_layer(self, in_channels, nblock, transition=True):
        layers = []
        for i in range(nblock):
            layers += [Bottleneck(in_channels, self.growth_rate)]
            in_channels += self.growth_rate
        out_channels = in_channels
        if transition:
            out_channels = int(math.floor(in_channels * self.reduction))
            layers += [Transition(in_channels, out_channels)]
        return nn.Sequential(*layers), out_channels
    
    def _initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def forward(self, x):
        out = self.features(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
    
def densenet_cifar():
    return DenseNet([6,12,24,16], growth_rate=12)

class MobileNetV1(nn.Module):
    def __init__(self, ch_in, n_classes):
        super(MobileNetV1, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
                )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                # pw
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
                )

        self.model = nn.Sequential(
            conv_bn(ch_in, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(1024, n_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x

def dwise_conv(ch_in, stride=1):
    return (
        nn.Sequential(
            #depthwise
            nn.Conv2d(ch_in, ch_in, kernel_size=3, padding=1, stride=stride, groups=ch_in, bias=False),
            nn.BatchNorm2d(ch_in),
            nn.ReLU6(inplace=True),
        )
    )

def conv1x1(ch_in, ch_out):
    return (
        nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU6(inplace=True)
        )
    )

def conv3x3(ch_in, ch_out, stride):
    return (
        nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU6(inplace=True)
        )
    )

class InvertedBlock(nn.Module):
    def __init__(self, ch_in, ch_out, expand_ratio, stride):
        super(InvertedBlock, self).__init__()

        self.stride = stride
        assert stride in [1,2]

        hidden_dim = ch_in * expand_ratio

        self.use_res_connect = self.stride==1 and ch_in==ch_out

        layers = []
        if expand_ratio != 1:
            layers.append(conv1x1(ch_in, hidden_dim))
        layers.extend([
            #dw
            dwise_conv(hidden_dim, stride=stride),
            #pw
            conv1x1(hidden_dim, ch_out)
        ])

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.layers(x)
        else:
            return self.layers(x)

class MobileNetV2(nn.Module):
    def __init__(self, ch_in=3, n_classes=10):
        super(MobileNetV2, self).__init__()

        self.configs=[
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1]
        ]

        self.stem_conv = conv3x3(ch_in, 32, stride=2)

        layers = []
        input_channel = 32
        for t, c, n, s in self.configs:
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(InvertedBlock(ch_in=input_channel, ch_out=c, expand_ratio=t, stride=stride))
                input_channel = c

        self.layers = nn.Sequential(*layers)

        self.last_conv = conv1x1(input_channel, 1280)

        self.classifier = nn.Sequential(
            nn.Dropout2d(0.2),
            nn.Linear(1280, n_classes)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.stem_conv(x)
        x = self.layers(x)
        x = self.last_conv(x)
        x = self.avg_pool(x).view(-1, 1280)
        x = self.classifier(x)
        return x

def MNIST():
    global model
    # name = st.sidebar.selectbox('Model', ['MNIST', 'CIFAR-10', 'CIFAR-100', 'ImageNet'])
    MNIST_option = st.selectbox(
        'What model do you want to try?',
        ('LeNet-5', 'AlexNet', 'CNN', 'SpinalNet'))
    
    @st.cache_resource
    def load(MNIST_option):
        loaded_model = None
        
        if MNIST_option == 'CNN':
            loaded_model = CNN()
            loaded_model.load_state_dict(torch.load('simplecnn.pth', map_location='cpu'))  # Load on CPU
            loaded_model.eval()
          
        elif MNIST_option == 'LeNet-5':
            loaded_model = LeNet5()
            checkpoint = torch.load('lenet5.pth', map_location='cpu')  # Load on CPU
            loaded_model.load_state_dict(checkpoint['model_state_dict'])
            loaded_model.eval()
            
        elif MNIST_option == 'SpinalNet':
            loaded_model = Spinalnet()
            loaded_model.load_state_dict(torch.load('spinalnet.pth', map_location='cpu'))
            loaded_model.eval()
            
        else: 
            loaded_model = AlexNet()  
            loaded_model.load_state_dict(torch.load('alex.pth', map_location='cpu'))  # Load on CPU
            loaded_model.eval()
            
        st.success("Loaded model!", icon="✅")
        
        return loaded_model
    
    def predict(model, image):
        input_tensor = transforms.ToTensor()(image)  # C H W
        input_batch = input_tensor.unsqueeze(0)  # B C H W

        with torch.no_grad():
            output = model(input_batch.cpu())  # Move the input to CPU
    
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
    
def CIFAR10():
    global model
    CIFAR10_option = st.selectbox(
    'What model do you want to try?',
    ('Densenet', 'mobilnetv1', 'mobilnetv2'))
    
    @st.cache_resource
    def load(CIFAR10_option):
        loaded_model = None
        
        if CIFAR10_option == 'Densenet':
            loaded_model = densenet_cifar()
            loaded_model.load_state_dict(torch.load('densenet_cifar.pth', map_location='cpu'))  # Load on CPU
            loaded_model.eval()
          
        elif CIFAR10_option == 'mobilnetv1':
            loaded_model = MobileNetV1(ch_in=3, n_classes=10)
            loaded_model.load_state_dict(torch.load('mobilenet_cifar.pth', map_location='cpu'))
            loaded_model.eval()
            
        elif CIFAR10_option == 'mobilnetv2':
            loaded_model = MobileNetV2(ch_in=3, n_classes=10)
            loaded_model.load_state_dict(torch.load('mobilenetv2_cifar.pth', map_location='cpu'))
            loaded_model.eval()
            
            
        st.success("Loaded model!", icon="✅")
        
        return loaded_model
    
    def predict(model, image):
        input_tensor = transforms.ToTensor()(image)  # C H W
        input_batch = input_tensor.unsqueeze(0)  # B C H W

        with torch.no_grad():
            output = model(input_batch.cpu())  # Move the input to CPU
    
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
        return probabilities
    
    model = load(CIFAR10_option)

    st.write('Now your model is :', CIFAR10_option)

    st.write('# 🔥MNIST Recognizer🔥' )

    CANVAS_SIZE = 192

    with st.chat_message("user"):
        st.write("Hello 👋 upload the cifar10 image !")
        st.write("Labels : 'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'")
        

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Resize and display the image
        img = cv2.resize(img, dsize=(32, 32))
        preview_img = cv2.resize(img, dsize=(CANVAS_SIZE + 5, CANVAS_SIZE + 5), interpolation=cv2.INTER_NEAREST)
        preview_img = cv2.cvtColor(np.array(preview_img), cv2.COLOR_BGR2RGB)
        st.image(preview_img)

        st.title('Custom model demo')

        y = predict(model, img)
        predicted_label_index = np.argmax(y).item()
        cifar10_labels = [
            'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
        ]

        predicted_label = cifar10_labels[predicted_label_index]
        
        
        with st.chat_message("assistant"):
            st.write("Let's see the result !👋")
            with st.spinner('Wait for it...'):
                # Simulate some processing time
                time.sleep(2)

            st.write('## Result : %s' % predicted_label)
            st.success('Is it correct..?!')
            st.bar_chart(y)

    txt = '''More information: \n 
    > https://medium.com/@hichengkang \n 
    > https://tck2001.github.io/ \n'''
    st.write('TCK :', txt)

    
if __name__ == "__main__":
    name = st.sidebar.selectbox('Model', ['MNIST', 'CIFAR-10', 'CIFAR-100', 'ImageNet'])
    if name == "MNIST":
        MNIST()
    elif name == "CIFAR-10":
        CIFAR10()
    else:
        MNIST()
