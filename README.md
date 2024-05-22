# knn-ocr
A Ocr (Optical Character Recognition) tool that uses Knn algorithm to recognize numbers from a given image.


## Files

- `knn.py` : Contains the implementation of Knn algorithm.
- `ocr.py` : Contains the implementation of Ocr tool.
- `ui.py` : Contains the implementation of the user interface for drawing the numbers.
- `train.py` : Contains the code to train a model with a training dataset.
- `test.py` : Contains the code to test the model by generating images with random numbers.


## Usage

### Training

There is an already trained model (`numbers.json`) in the repository. If you want to train a new model, you can run the following code:

```python
from training import train
train("trained.json", r'C:\Users\Eleve\Downloads\archive\data\testing_data')
```

and to use it:
```python

from ocr import OCR

ocr = OCR()
ocr.load_json("trained.json")
print(ocr.predict("image.png"))
```

### Use a pre-trained model

```python
from ocr import OCR

ocr = OCR("numbers")
print(ocr.predict("image.png"))
```

