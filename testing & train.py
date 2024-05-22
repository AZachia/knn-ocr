import random
from ocr import OCR
from PIL import Image, ImageDraw, ImageFont
import string


def test_ocr(nb_test: int = 20, train: bool = False, chars: str = string.digits , ocr: OCR = None):
    
    if ocr is None:
        ocr = OCR()
    fonts = ['arial.ttf', 'arialbd.ttf', 'arialbi.ttf', 'ariali.ttf', 'times.ttf', 'timesbd.ttf', 'timesbi.ttf', 'timesi.ttf', 'verdana.ttf', 'verdanab.ttf', 'verdanai.ttf', 'verdanaz.ttf', 'tahoma.ttf', 'tahomabd.ttf']
    results = []
    for i in range(nb_test):
        nb = "".join([str(random.choice(chars)) for _ in range(random.randint(1, 5))])
        
        # write the number on an image
        img = Image.new('RGB', (100, 100), color = (255, 255, 255))
        img_draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(random.choice(fonts), 30)
        
        
        img_draw.text((10, 10), nb, fill=(0, 0, 0), font=font)
        
                
        nb2 = ocr.ocr_image(img)
        print(f"Expected: {nb}, got: {nb2}. Result: {nb.strip() == nb2.strip()}")
        results.append(nb.strip() == nb2.strip())
        
        if train and nb.strip() != nb2.strip():
            ocr.train_image(img, nb)
            
    return sum(results) / len(results)

if __name__ == "__main__":
    ocr = OCR("letters")
    res = 0
    while res < 50:
        res = test_ocr(50, ocr=ocr, train=True, chars=string.ascii_uppercase + string.digits)*100
        print(res)
        
    print(res)
        
