from tkinter import *
from PIL import ImageGrab
import numpy as np
import datetime
from threading import Thread
import time

image = False
file_save = False
network_structure = True

if image:
    canvas_width = 600
    canvas_height = 450


    def paint(event):
        color = 'red'
        x1, y1 = (event.x-1), (event.y-1)
        x2, y2 = (event.x+1), (event.y+1)
        c.create_oval(x1, y1, x2, y2, fill=color, outline=color)


    def save(event):
        print("yes")
        x = c.winfo_rootx() + c.winfo_x()
        y = c.winfo_rooty() + c.winfo_y()
        x1 = x + c.winfo_width()
        y1 = y + c.winfo_height()
        ImageGrab.grab().crop((x, y, x1, y1)).save("./canvas.jpg")


    master = Tk()
    master.title('Paint App')

    c = Canvas(master=master, width=canvas_width, height=canvas_height, bg='white')
    c.pack(expand=YES, fill=BOTH)
    c.bind('<B1-Motion>', paint)
    c.bind('<Button-3>', save)

    message = Label(master=master, text="Press and drag to draw")
    message.pack(fill=BOTH)
    master.mainloop()

    imageHolder = Label(master=master)
    imageHolder.pack()

if file_save:

    data_structure = [["a", 1, 3, True],
                      [False, 3, 2, "b"],
                      ["c", "d", False, 1],
                      ["e", 2, True, "f", [8, 6, [0, [7]]]]]

    def save():
        print("Saving the network...")
        start_time = time.time()
        thread = Thread(target=save_thread, args=(start_time,))
        thread.start()
        thread.join()

    def save_thread(start_time):
        network_name = "network_" + datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
        np.save(network_name, data_structure, allow_pickle=True)
        print("Finished saving in {:.5f} seconds".format(time.time() - start_time))

    def load():
        data_structure = np.load()

    save()

if network_structure:
    network = np.load("./nets/nets_2020_11_17__00_43_30.npy", allow_pickle=True)
    print()
