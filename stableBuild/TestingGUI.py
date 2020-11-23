import datetime
import os
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

from PIL import ImageTk, Image

# Using PlaidML to speed up the training/testing process, since Tensorflow is not an option for me because
# it requires an NVIDIA card, which I don't own. PlaidML on the other side is not based on CUDA but on OpenGL and thus
# (almost) any graphics card can be used with PlaidML (in my case I have a Radeon RX 480 graphics card which speeds up
# the training process by about 100% as if I were to use my CPU).

# Installing plaidml as backend before importing keras to ensure that correct backend is used with keras
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array


class Application(tk.Frame):
    """
    GUI for testing a CNN Character model (multi-class images)

    @author David Jerman
    """

    def __init__(self, master=None):
        """
        Initialization of widgets and variables
        
        :param master: The root tkinter window
        """

        super().__init__(master)
        self.master = master

        # Variables
        self.default_img = "./default_img.png"
        self.model = None
        self.isValidModel = False
        self.img = 0
        self.isValidImg = False
        self.model_path = "  N/A  "
        self.img_path = "  N/A  "
        # Starting time
        self.start_time = datetime.datetime.now().strftime('%Y/%m/%d - %H:%M:%S')
        self.character_labels_dictionary = {'4a': 19, '4b': 20, '4c': 21, '4d': 22, '4e': 23, '4f': 24,
                                            '5a': 35,
                                            '6a': 45, '6b': 46, '6c': 47, '6d': 48, '6e': 49, '6f': 50,
                                            '7a': 61,
                                            '30': 0, '31': 1, '32': 2, '33': 3, '34': 4, '35': 5, '36': 6,
                                            '37': 7, '38': 8, '39': 9,
                                            '41': 10, '42': 11, '43': 12, '44': 13, '45': 14, '46': 15, '47': 16,
                                            '48': 17, '49': 18,
                                            '50': 25, '51': 26, '52': 27, '53': 28, '54': 29, '55': 30, '56': 31,
                                            '57': 32, '58': 33, '59': 34,
                                            '61': 36, '62': 37, '63': 38, '64': 39, '65': 40, '66': 41, '67': 42,
                                            '68': 43, '69': 44,
                                            '70': 51, '71': 52, '72': 53, '73': 54, '74': 55, '75': 56, '76': 57,
                                            '77': 58, '78': 59, '79': 60
                                            }
        # Input variables
        self.image_size = [0, 0]
        self.color_depth = 0

        # Menu bar
        self.help_menu = tk.Menu(master=self.master)
        self.help_menu.add_command(label="Exit", command=self.master.destroy)
        self.help_menu.add_command(label="About", command=self.about)
        self.help_menu.add_command(label="Help", command=self.help)
        self.master.config(menu=self.help_menu)

        # Model selection buttons
        self.select_model_button = tk.Button(master=self.master)
        self.select_model_label = tk.Label(master=self.master)

        # Image selection buttons
        self.select_img_button = tk.Button(master=self.master)
        self.select_img_label = tk.Label(master=self.master)

        # Output frame with information about the testing of the model
        self.output_frame = tk.Frame(master=root)
        self.output_text = tk.Text(master=self.output_frame)
        self.output_scrollbar = tk.Scrollbar(master=self.output_frame)

        # Image window for displaying the prediction image
        self.img_window = tk.Canvas(master=self.master)

        # Exit button to exit the program
        self.clear_output_button = tk.Button(master=self.master)

        # Prediction button to start the model prediction
        self.predict_button = tk.Button(master=self.master)

        # Initial output console setup
        self.clear_output()

        self.create_widgets()

    def create_widgets(self):
        """
        Creates widgets and puts them in a grid
        """
        # Font size
        font_size = 9

        # Padding
        padx = 10
        pady = 8

        # Configuring rows
        self.master.rowconfigure(2, weight=20)
        self.master.rowconfigure(3, weight=1)

        # Configuring columns
        self.master.columnconfigure(0, weight=50)
        self.master.columnconfigure(1, weight=1)
        self.master.columnconfigure(2, weight=100)
        self.master.columnconfigure(3, weight=100)

        # Selection pane for model
        self.select_model_button.config(text="Select model", width=30, command=self.get_model, bg='yellow',
                                        font=("Courier", font_size))
        self.select_model_button.grid(column=0, row=0, padx=(20, padx), pady=pady, sticky='w')
        self.select_model_label.config(text=self.model_path, relief='sunken', width=110, font=("Courier", 9))
        self.select_model_label.grid(column=1, row=0, columnspan=3, sticky='w', padx=padx, pady=pady)

        # Selection pane for image
        self.select_img_button.config(text="Select image", width=30, command=self.get_image, bg='yellow',
                                      font=("Courier", font_size))
        self.select_img_button.grid(row=1, column=0, sticky='w', pady=pady, padx=(20, padx))
        self.select_img_label.config(text=self.img_path, relief='sunken', width=110, font=("Courier", font_size))
        self.select_img_label.grid(column=1, row=1, columnspan=3, sticky='w', padx=padx, pady=pady)

        # Output
        # Output frame
        self.output_frame.config(height=20, width=60)
        self.output_frame.grid(row=2, column=0, columnspan=2, rowspan=2, padx=padx, pady=pady, sticky='w')

        # Output text
        self.output_text.config(relief='sunken', height=23, width=56, state=tk.DISABLED, font=("Courier", font_size))
        self.output_text.grid(row=0, column=1, padx=0, pady=0, sticky='e')

        # Output scrollbar
        self.output_scrollbar.config(orient='vertical', command=self.output_text.yview)
        self.output_text.config(yscrollcommand=self.output_scrollbar.set)
        self.output_scrollbar.grid(row=0, column=0, padx=0, pady=0, sticky='nsw')

        # Image canvas
        self.set_image()
        self.img_window.config(width=20, height=25)
        self.img_window.grid(column=2, row=2, columnspan=2, padx=(80, 100), pady=0, sticky='nsew')

        # Clear output button
        self.clear_output_button.config(command=self.clear_output, text="Clear output", width=30, bg='white',
                                        font=("Courier", font_size))
        self.clear_output_button.grid(column=3, row=3, padx=(padx, 20), sticky='we')

        # Predict button
        self.predict_button.config(command=self.predict, text="Predict", width=30, bg='white',
                                   font=("Courier", font_size))
        self.predict_button.grid(column=2, row=3, padx=padx, sticky='ew')

        # Max/min window size - fixed window size
        self.master.maxsize(910, 450)
        self.master.minsize(910, 450)

    def get_model(self):
        """
        Sets the model
        """
        try:
            file_name = filedialog.askopenfilename(parent=self.master, filetypes=(("model files", "*.h5"),))
            # Information about getting the model
            self.write_output("Getting the model...")
            self.master.update()
            # Loading the model
            self.model = load_model(file_name)
            self.model_path = file_name
            self.isValidModel = True
            self.clear_output()
            # Get information about the model and save it to the object
            self.get_model_info()
            self.select_model_label.config(text="  " + self.model_path + "  ")
            self.select_model_button.config(bg='green')
            # If everything succeeds, proceed to start the model
            self.start_model()
        except:
            # An error has occurred
            self.model_path = "Invalid - Please try a different model/ try again"
            self.write_output("An error occurred")
            self.select_model_label.config(text="  " + self.model_path + "  ")
            self.select_model_button.config(bg='red')
            self.isValidModel = False

    def get_image(self):
        """
        Sets the image
        """
        try:
            file_name = filedialog.askopenfilename(parent=self.master, filetypes=(("png files", "*.png"),
                                                                                  ("jpg files", "*.jpg")))
            self.img = load_img(file_name)
            self.img_path = file_name
            self.select_img_label.config(text="  " + self.img_path + "  ")
            self.select_img_button.config(bg='green')
            self.isValidImg = True
            self.set_image(self.img_path)
        except:
            self.img_path = "Invalid - Please try a different image/ try again"
            self.select_img_label.config(text="  " + self.img_path + "  ")
            self.select_img_button.config(bg='red')
            self.isValidImg = False

    def clear_output(self):
        """
        Clears output console
        """
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete(1.0, tk.END)
        # Writing out the start time - when the application was started
        self.write_output("Start time: " + self.start_time)
        self.output_text.config(state=tk.DISABLED)

    def predict(self):
        """
        Makes a model prediction
        """
        if self.isValidModel and self.isValidImg:
            try:
                img = self.load_image()
                digit = self.model.predict_classes(img)
                self.write_output("Character: " + bytes.fromhex(str(self.get_key(digit[0]))).decode('utf-8'))
            except:
                self.write_output("Unknown error has occured")
        elif not self.isValidImg and not self.isValidModel:
            self.write_output("Missing model and image")
        elif not self.isValidModel:
            self.write_output("Missing model")
        elif not self.isValidImg:
            self.write_output("Missing image")

    def set_image(self, img_path=None):
        """
        Sets and image to a canvas

        :param img_path: Path to the image
        """
        if img_path is None:
            img_path = self.default_img
        img = Image.open(img_path)
        img = img.resize((int(300), int(300)), Image.ANTIALIAS)
        # We need to keep a reference to this image or it will not appear
        self.img = ImageTk.PhotoImage(img)
        self.img_window.create_image(20, 20, anchor=tk.NW, image=self.img)

    def load_image(self):
        """
        Loads and pre-processes the image for prediction
        """
        # Load the image
        color_mode = 0
        if self.color_depth == 1:
            color_mode = "grayscale"
        elif self.color_depth == 3:
            color_mode = "rgb"
        img = load_img(self.img_path, target_size=(self.image_size[0], self.image_size[1]), color_mode=color_mode)
        # Convert to array
        img = img_to_array(img)
        # Reshape into a single sample with 1 channel
        img = img.reshape(1, self.image_size[0], self.image_size[1], self.color_depth)
        # Prepare pixel data
        img = img.astype('float32')
        # Normalization
        img = img / 255.0
        return img

    def get_key(self, val: str) -> int:
        """
        Gets a key based on a value out of a dictionary

        :param val: Value corresponding to the key
        """
        for key, value in self.character_labels_dictionary.items():
            if val == value:
                return key
        return -1

    def write_output(self, text):
        """
        Writes text to the output window

        :param text: Text to write
        """
        self.output_text.config(state=tk.NORMAL)
        self.output_text.insert(tk.INSERT, " > " + text + "\n")
        self.output_text.config(state=tk.DISABLED)

    def get_model_info(self):
        """
        Obtains information about the model and adjusts the input parameters based on the model
        """
        try:
            if self.isValidModel:
                self.image_size[0] = self.model.get_input_shape_at(0)[1]
                self.image_size[1] = self.model.get_input_shape_at(0)[1]
                # White-black images
                if self.model.get_input_shape_at(0)[3] == 1:
                    self.color_depth = 1
                # Color images
                elif self.model.get_input_shape_at(0)[3] == 3:
                    self.color_depth = 3
                # Model not compatible with this program
                else:
                    self.model = 0
                    self.isValidModel = False
                    raise Exception("Invalid model")
                # Printing out the model shape
                self.write_output("Model input shape: (" + str(self.image_size[0]) + ", " + str(self.image_size[1]) + ", "
                                  + str(self.color_depth) + ")")
        except:
            self.write_output("Failed to obtain model info")

    def start_model(self):
        """
        Makes a prediction just to start the model
        """
        self.img_path = self.default_img
        self.model.predict(self.load_image())
        self.set_image()
        self.select_img_button.config(bg='yellow')
        self.select_img_label.config(text="  N/A  ")
        self.write_output("Model working")

    def about(self):
        """
        About pop-up window
        """
        # About message
        messagebox.showinfo(title="About", message='CNN Testing GUI.\n'
                                                   '\nThe application provides user with the ability to test'
                                                   ' a pre-trained CNN Character model. The application was'
                                                   ' created as a part of a school project.\n'
                                                   '\nBackend: PlaidML (https://github.com/plaidml/plaidml)\n'
                                                   '\nAuthor: David Jerman'
                                                   '\nVersion: 2020.03.28', master=self.master)

    def help(self):
        """
        Help pop-up window
        """
        # Help message
        messagebox.showinfo(title="Help", message='How to use the program:\n'
                                                  '\nFirst select the model (the model needs to be trained on a 62'
                                                  ' character dataset or it won\'t work) --> The model takes some time'
                                                  ' to load.\n'
                                                  '\nThen select an image (the model mostly works with black-white, in'
                                                  ' paint created images).\n'
                                                  '\nMake a prediction by clicking on predict button.\n'
                                                  '\nThe console can be cleared by clicking on the Clear Output button.'
                                                  '\n\nLabels in the upper part of the GUI represent the model and the'
                                                  ' image path', master=self.master)


# Main tkinter loop
root = tk.Tk()
root.title("CNN Testing GUI")
app = Application(master=root)
app.mainloop()
