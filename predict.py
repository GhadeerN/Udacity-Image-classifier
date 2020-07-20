import tensorflow as tf
import tensorflow_hub as hub
import argparse
from PIL import Image
import json
import numpy as np

#arguments
parser = argparse.ArgumentParser(description="Flowers Image Cassifier") #object 
parser.add_argument('-img','--img_path', type = str,help = 'The path of an image to predict', default= './test_images/orange_dahlia.jpg')
parser.add_argument('--model', type = str, help = 'The model path', default= 'test_model.h5')
parser.add_argument('--top_k', type = int,help = 'Top_k results of must likely classses', default= 5)
parser.add_argument('--classes', help = 'Class names or Category names ', default= 'label_map.json')

arg = parser.parse_args()
img = arg.img_path
model = arg.model
topk = arg.top_k
classes = arg.classes  

#read json file :)
with open('label_map.json', 'r') as n:
    class_names = json.load(n)

new_class_names = dict()
for i in class_names:
    new_class_names[str(int(i)-1)] = class_names[i]
    
    
load_model = tf.keras.models.load_model(model,custom_objects={'KerasLayer':hub.KerasLayer}) #load the model


#Create the process_image function
#the image size = 224
def process_image(img):
    tensor_img = tf.convert_to_tensor(img)
    tensor_img = tf.image.resize(tensor_img, (224,224))
    normalize_img = tensor_img/255
    return normalize_img.numpy()

#Create the predict function
def predict(path, model, top_k): #path => image path
    
    img = Image.open(path)
    test_img = np.asarray(img)
    trans_img = process_image(test_img)
    r_img = np.expand_dims(trans_img, axis=0)
    pred = load_model.predict(r_img)
    
    pred = pred.tolist()
   
    values, indices = tf.math.top_k(pred, k = top_k)
    probs = values.numpy().tolist()[0]
    classes = indices.numpy().tolist()[0]
    
    return probs, classes

    
def sanity_check(img):
        image = Image.open(img)
        test_img = np.asarray(image)
        process_img = process_image(test_img)
        
        probs, labels = predict(img, load_model, 5)
        print('The probabilities: ',probs)
        print("The labels: ",classes)
        
        item_name = [new_class_names[str(idd)] for idd in labels]
        print('Flower name: ',item_name)

    
  
if __name__ == '__main__':
    probs, classes = predict(img, model, topk)
    sanity_check(img)
    ##predict(img, load_model, topk)
