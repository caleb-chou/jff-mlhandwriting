from tkinter import *
from PIL import Image, ImageDraw
import os.path
from csp_ml_main import Recognizer

wid = 280
hei = 280
center = hei/2
white = (255,255,255)
black = (0,0,0)

image = Image.new("RGB",(wid,hei), black )
draw = ImageDraw.Draw(image)

canvas_width = 280
canvas_height = 280

def paint( event ):
   color = "#FFFFFF"
   x1, y1 = ( event.x ), ( event.y )
   x2, y2 = ( event.x + 25 ), ( event.y )
   w.create_line( x1, y1, x2, y2, fill = color, width = 25)
   draw.line([x1, y1, x2, y2], fill = white,width = 25)
def clear():
    w.delete('all')
    global draw
    draw.rectangle((0,0,280,280), fill =(0,0,0,0))
def save():
    w.update()
    #os.remove(os.path.abspath(os.path.pardir) + '/resources/temp.png')
    image.convert('LA').resize((28,28),Image.ANTIALIAS).save(os.path.abspath(os.path.pardir) + '/resources/temp.png')
model = Recognizer("defaultmodel.pkl")
model.load()
def predict():
    global model
    prediction = model.predict_image_from_path()
    print(prediction)
def make_prediction():
    save()
    predict()
master = Tk()
master.title( "Draw a Number" )
w = Canvas(master, 
           width=canvas_width, 
           height=canvas_height)
w.pack(expand = YES, fill = BOTH)
w.bind( "<B1-Motion>", paint )
w.bind( '<Button-3>', clear)
save_button = Button(master, text = "Predict", command = make_prediction)
save_button.pack (side = RIGHT)
clear_button = Button(master, text = "Clear", command = clear)
clear_button.pack(side = LEFT)
#message = Label( master, text = "Press and Drag the mouse to draw" )
#message.pack( side = BOTTOM )
    
mainloop()