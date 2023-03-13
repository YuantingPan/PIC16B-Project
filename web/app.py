from flask import Flask, render_template, request
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import os

app = Flask(__name__)

PATH = "./model/"
# Define the device to use for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# class and label convertion
class2label = {'angry': 0,
                'disgust': 1,
                'fear': 2,
                'happy': 3,
                'neutral': 4,
                'sad': 5,
                'surprise': 6}
label2class = {v: k for k, v in class2label.items()}

# resnet
class Resnet(nn.Module):
    def __init__(self, mode='finetune', augmented=False, pretrained=True):
        super().__init__()
        self.augmented = augmented
        self.resnet = models.resnet50(pretrained=True)
        if mode == 'linear':
            for name, param in self.resnet.named_parameters():
                if param.requires_grad and 'fc' not in name:
                    param.requires_grad = False
            num_features = self.resnet.fc.in_features
            self.resnet.fc = nn.Linear(num_features, 7)
        elif mode == 'finetune':
            num_features = self.resnet.fc.in_features
            self.resnet.fc = nn.Linear(num_features, 7)

    def forward(self, x):
        x = self.resnet(x)
        return x

    def to(self, device):
        return self.resnet.to(device=device)

model = Resnet(mode='finetune', augmented = False, pretrained=True)
model.load_state_dict(torch.load(PATH+"resnet_finetune5.pt",map_location='cpu'))

@app.route('/')
def upload_file():
   return render_template('upload.html')
	
@app.route('/uploader', methods = ['GET', 'POST'])
def get_upload_file():
   if request.method == 'POST':
      f = request.files['file']
      f.filename="./static/"+f.filename
      f.save(f.filename)

      img = Image.open(f.filename)
      img = img.convert('RGB')
      img = transforms.Grayscale(num_output_channels=3)(img)

      img = transforms.Resize((128, 128))(img)
      img = transforms.ToTensor()(img)
      img = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])(img)
      pred_fraction = None
      pred_label = None
      model.eval()
      with torch.no_grad():
         logit = model(torch.tensor(np.expand_dims(img,axis=0)).to(device)).detach().cpu().numpy()
         logit = np.maximum(logit, 0)    # relu
         pred_fraction = logit / np.sum(logit)  # normalize
         pred_label = np.argmax(logit)
      invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                              transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ]) ])
      image = invTrans(img)
      image = np.moveaxis(image.numpy(), 0, -1)
      frac = pred_fraction[0]
      sorted_id = np.flip(np.argsort(frac))
      valid_id = frac[sorted_id]!=0   # excluding zero fractions
      sorted_id = sorted_id[valid_id] # id of sorted fractions, big to small
      xlab = ""
      for id in sorted_id:
         xlab += list(class2label.keys())[id]
         xlab += ': '
         xlab += str(np.round(pred_fraction[0][id]*100,2))
         xlab += "%\n"

      return render_template('uploader.html', img=f.filename,result=xlab,predict=label2class[pred_label])

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
