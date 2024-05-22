import os
import tkinter as tk
from PIL import Image, ImageGrab
from ocr import OCR, get_image_data

root = tk.Tk()

pen_size = 5

canvas = tk.Canvas(root, width=400, height=400, bg='white')
canvas.pack()

def paint(event):
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    canvas.create_oval(x1, y1, x2, y2, fill='black', width=pen_size)
    return

canvas.bind('<B1-Motion>', paint)

def clear():
    canvas.delete('all')
    value_entry.delete(0, 'end')
    
def recognise():
    # transform the canvas into a PIL image
    
    result = ""
    
    x = root.winfo_rootx() + canvas.winfo_x()
    y = root.winfo_rooty() + canvas.winfo_y()
    xx = x + canvas.winfo_width()
    yy = y + canvas.winfo_height()
    ImageGrab.grab(bbox=(x + 120, y + 100, xx, yy)).save('tmp.png')
    
    
    img = Image.open('tmp.png')
    # img.show()    
    
    ocr = OCR()
    # ocr = OCR("letters")
    result = ocr.ocr_image(img)
    print("Result:", result)
    
    result_label.config(text='Result: ' + result)
    
    os.remove('tmp.png')
    
    if train and value_entry.get() != "":
        ocr.train_image(img, value_entry.get())
    

train = True

result_label = tk.Label(root, text='Result: ')
result_label.pack()
    
    

recognise_btn = tk.Button(root, text='Recognise', command=recognise)
recognise_btn.pack()

cls_btn = tk.Button(root, text='Clear', command=clear)
cls_btn.pack()

value_entry = tk.Entry(root)
value_entry.pack()

root.mainloop()