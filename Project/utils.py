import os, cv2, tarfile, torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import confusion_matrix
import torchvision.transforms as transforms


def show_examples(train_dir):
    """
        Show 5 images of each facial expression from the training set.
    """
    target_var = os.listdir(train_dir)
    fig, axes = plt.subplots(7, 5, figsize=(16, 24))

    for i in range(len(target_var)):
        for j in range(5):
            
            image = cv2.imread(os.path.join(train_dir, target_var[i], os.listdir(os.path.join(train_dir, target_var[i]))[j]))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            axes[i][j].imshow(image)
            axes[i][j].set_title(target_var[i] + "-" + str(j+1))
            axes[i][j].axis('off')
                
    plt.axis('off')
    plt.show()
        

def show_distribution(train_dir, test_dir):
    """
        Show the fraction of emotion labels in train and test sets.
    """
    target_var = os.listdir(train_dir)
    x_train = np.array([ len(os.listdir(os.path.join(train_dir, i))) for i in target_var ])
    x_test = np.array([ len(os.listdir(os.path.join(test_dir, i))) for i in target_var ])
    label = target_var
    
    fig, axes = plt.subplots(1, 2, figsize=(8,4))
    axes[0].pie(x_train, labels=label, autopct='%1.1f%%',shadow=True, startangle=90)
    axes[1].pie(x_test, labels=label, autopct='%1.1f%%',shadow=True, startangle=90)
    axes[0].set_title('Train')
    axes[1].set_title('Test')
    plt.show()

    for i in target_var:
        print('Emotion : ' + i )
        print('\tTraining : ' + str(len(os.listdir(os.path.join(train_dir, i)))) +
              '\n\t Testing : ' + str(len(os.listdir(os.path.join(test_dir, i)))))
        

def test_evaluate(model, test_loader, device):
    try:
        model.eval().cuda()
    except:
        model.eval()

    labels = []
    
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs[0].to(device)
            logits = model(inputs)
            _, predictions = torch.max(logits, dim=1)

            for prediction in predictions:
                labels += [prediction.item()]

    return labels


def get_sample_inputs(loader, seed=0):
  """
    Get the input and labels for each of the seven facial expressions
    seed: changes the set of images. < 64
  """
  true_labels = np.arange(7)
  all_inputs = []
  cnt = 0
  for inputs, labels in loader:
    labels = labels.numpy()
    if labels[seed] == true_labels[cnt]:
        all_inputs.append(inputs[seed])
        cnt += 1
        if cnt == 7:
          return all_inputs, true_labels


def eval_sample_inputs(model, loader, device, seed=0):
  """
    Evaluate the inputs, relu and normalize output, return the fraction of each facial expression
  """
  pred_fractions = []
  pred_labels = []
  inputs, true_labels = get_sample_inputs(loader, seed)
  model.eval()

  with torch.no_grad():
    for input in inputs:
      logit = model(torch.tensor(np.expand_dims(input,axis=0)).to(device)).detach().cpu().numpy()
      pred = np.argmax(logit)
      logit = np.maximum(logit, 0)    # relu
      pred_fractions.append(logit / np.sum(logit))   # normalize
      pred_labels.append(pred)
  return inputs, true_labels, pred_fractions, pred_labels


def show_samples(inputs, true_labels, pred_fractions, pred_labels, label2class, class2label):
  """
    Show sample images of the 7 facial expressions,
    with true and predicted labels, and predicted fractions for each expression
  """
  # inverse transform back to normal images
  invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                  transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ]) ])
  plt.figure(figsize=(20,20))
  for i in range(7):
    plt.subplot(3,3,i+1)
    image = invTrans(inputs[i])

    # dimension (3, 128, 128) to (128, 128, 3)
    image = np.moveaxis(image.numpy(), 0, -1)
    plt.imshow(image)
    plt.title(f"True Label: {label2class[true_labels[i]]}")

    frac = pred_fractions[i][0]
    sorted_id = np.flip(np.argsort(frac))
    valid_id = frac[sorted_id]!=0   # excluding zero fractions
    sorted_id = sorted_id[valid_id] # id of sorted fractions, big to small

    # xlabel text
    xlab = f"Predicted Label: {label2class[pred_labels[i]]}\n\n"
    for id in sorted_id:
      xlab += list(class2label.keys())[id]
      xlab += ': '
      xlab += str(np.round(pred_fractions[i][0][id]*100,2))
      xlab += "%\n"
    plt.xlabel(xlab)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.tight_layout()
  plt.show()
