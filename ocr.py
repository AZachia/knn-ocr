import numpy as np
from knn import KNN
from PIL import Image
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


class OCR(KNN):
    def __init__(self, model: str | None = 'numbers', k: int = 5):
        if model is not None and not os.path.exists(model +'.json') and not os.path.exists(model):
            raise FileNotFoundError('Model not found')
        
        self.model = model
        super().__init__(k)
        if model is not None:
            if os.path.exists(model + '.json'):
                self.load_json(model + '.json')
            else:
                self.load_json(model)
    
    def predict_image(self, image):
        return self.predict(get_image_data(image))
    
    def ocr_image(self, image: str | Image.Image) -> str:
        
        images = self.split_image(image)
        caraters = []
        
        for img, (x, y, w, h) in images:
            caraters.append((self.predict_image(img), x))
        
        caraters.sort(key=lambda x: x[1])
        
        text = ''.join([c[0] for c in caraters])
        
        return text
    
    def split_image(self, image: str | Image.Image) -> list:
        
        if isinstance(image, str):
            image = Image.open(image)
        else:
            image = image
            
        image_grayscale = image.convert("L")
        image_np = np.array(image_grayscale)

        _, thresh = cv2.threshold(image_np, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        images = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            character_image = thresh[y:y+h, x:x+w]
            
            img = Image.fromarray(character_image)
            img = img.point(lambda x: 255-x)
            
            images.append((img, (x, y, w, h)))
        
        return images
    
    def train_image(self, image: str | Image.Image, value: str, model: str = None):
        result = self.ocr_image(image)
        
        if result.strip() != value.strip():
            images = self.split_image(image)
            positions = [x for _, (x, _, _, _) in images]
            # sort the images by x position
            images = sorted(zip(images, positions), key=lambda x: x[1])

            if len(value) == len(result) and len(images) == len(result):
                for i, image in enumerate(images):
                    if result[i] != value[i]:
                        data = get_image_data(image[0][0])
                        data['label'] = value[i]

                        self.add_element(data)
                        
                        if model is None:
                            model = self.model
                        if model:
                            self.save_json(model + '.json')
        

if __name__ == "__main__":
    ocr = OCR("letters")
    print(ocr.ocr_image("test.PNG"))
    print(ocr.ocr_image("test copy.PNG"))
    while ocr.ocr_image("test copy.PNG") != "BONJOURJESUISEN4EMEETJESUISNEEN1983LE7":
        ocr.train_image("test copy.PNG", "BONJOURJESUISEN4EMEETJESUISNEEN1983LE7")
        print(ocr.ocr_image("test copy.PNG"))
