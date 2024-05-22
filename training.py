from PIL import Image
import numpy as np
from knn import KNN
import os
import cv2

def get_image_data(image: str | Image.Image) -> dict:

    if isinstance(image, str):
        image = Image.open(image)
    else:
        image = image

    img = image
    img = img.resize((8, 8))
    img = img.convert('L')
    img = list(img.getdata())
    return {str(i): img[i] for i in range(len(img))}


def training_1(name: str = "knnset.json", training_path: str = 'training_data'):
    
    knn = KNN(5)

    for label in os.listdir(training_path):
        print('Label:', label)
        for image in os.listdir(training_path + '/' + label):
        
            data = get_image_data(training_path + '/' + label + '/' + image)
            data['label'] = label
            
            knn.add_element(element=data)


    knn.save_json(name)
    

def training_2(name: str = "knnset.json", training_path: str = 'training_data'):

    knn = KNN(5)

    for label in os.listdir(training_path):
        print('Label:', label)
        for image in os.listdir(training_path + '/' + label):
        
            data = get_image_data(training_path + '/' + label + '/' + image)
            data['label'] = label
            
            knn.add_element(element=data)
            
            image = Image.open(training_path + '/' + label + '/' + image)
            image_grayscale = image.convert("L")
            image_np = np.array(image_grayscale)

            _, thresh = cv2.threshold(
                image_np, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            caraters = []

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                character_image = thresh[y:y+h, x:x+w]


                img = Image.fromarray(character_image)
                img = img.point(lambda x: 255-x)
                
                data = get_image_data(img)
                data['label'] = label
                
                knn.add_element(element=data)
        
    # knn.save_json(name)


if __name__ == "__main__":
    
    training_1("letters.json", r'C:\Users\Eleve\Downloads\archive\data\testing_data')

    # test_img = r"C:\Users\Eleve\Downloads\archive\data\training_data\0\468.png"
    # data = get_image_data(test_img)

    # knn = KNN(5)
    # knn.load_json('numbers.json')
    # print(knn.elements)
    # print(knn.predict(data))

        
        