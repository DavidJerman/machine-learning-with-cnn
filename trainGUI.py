import datetime
import os
import time
import tkinter as tk
from threading import Thread
from tkinter import filedialog
from tkinter import messagebox
import psutil as sm
from threading import Timer
import numpy as np

# Using PlaidML to speed up the training/testing process, since Tensorflow is not an option for me because
# it requires an NVIDIA card, which I don't own. PlaidML on the other side is not based on CUDA but on OpenGL and thus
# (almost) any graphics card can be used with PlaidML (in my case I have a Radeon RX 480 graphics card which speeds up
# the training process by about 100% as if I were to use my CPU).

# Installing plaidml as backend before importing keras to ensure that correct backend is used with keras
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import keras


class Application(tk.Frame):
    """
    GUI for training a CNN Model (multi/single-class images)

    @author David Jerman

    Date of modification: 2020.04.28
    """

    def __init__(self, master=None, callbacks=None):
        """
        Initialization of widgets and variables

        :param master:    The root tkinter window
        :param callbacks: Custom callbacks
        """

        super().__init__(master)
        self.master = master
        self.callbacks = callbacks

        # Handling the tkinter exit button
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Start time
        self.start_time = datetime.datetime.now().strftime('%Y/%m/%d - %H:%M:%S')

        # Variables
        self.train_size_var = tk.IntVar()
        self.test_size_var = tk.IntVar()
        self.num_classes_var = tk.IntVar()
        self.img_size_var = tk.IntVar()
        self.batch_size_var = tk.IntVar()
        self.epochs_var = tk.IntVar()
        self.optimizer_var = tk.StringVar()
        self.lr_var = tk.DoubleVar()
        self.loss_function_var = tk.StringVar()
        self.metrics_var = tk.StringVar()
        self.train_folder_var = tk.StringVar()
        self.test_folder_var = tk.StringVar()
        self.save_folder_var = tk.StringVar()
        self.output_shape_var = tk.StringVar()
        self.train_job_epoch_var = tk.StringVar()
        self.train_job_batch_var = tk.StringVar()
        self.train_job_num_var = tk.IntVar()
        self.train_job_id_var = tk.StringVar()
        self.train_job_acc_var = tk.DoubleVar()
        self.train_job_loss_var = tk.DoubleVar()
        self.ram_usage_var = tk.StringVar()
        self.cpu_usage_var = tk.StringVar()
        self.time_var = tk.StringVar()
        self.train_len_var = tk.IntVar()
        self.stopped_training_var = tk.BooleanVar()
        self.network = [[[] for x in range(8)] for y in range(50)]
        self.model = None
        self.train_job_thread = None

        self.network_position_index = 0
        self.first_layer = True

        # Start timer
        self.system_resource_monitor = SystemResourceMonitor(ram_usage_var=self.ram_usage_var,
                                                             cpu_usage_var=self.cpu_usage_var,
                                                             time_var=self.time_var)

        # Menu bar
        self.help_menu = tk.Menu(master=self.master)
        self.help_menu.add_command(label="Exit", command=self.exit_app)
        self.help_menu.add_command(label="About", command=self.about)
        self.help_menu.add_command(label="Help", command=self.help)
        self.master.config(menu=self.help_menu)

        # Train section
        self.train_folder_button = tk.Button(master=self.master)
        self.train_folder_label = tk.Label(master=self.master)

        # Validation section
        self.test_folder_button = tk.Button(master=self.master)
        self.test_folder_label = tk.Label(master=self.master)

        # Save model section
        self.save_folder_button = tk.Button(master=self.master)
        self.save_folder_label = tk.Label(master=self.master)

        # Number of images per dir - train
        self.train_num_button = tk.Button(master=self.master)
        self.train_num_entry = tk.Entry(master=self.master)

        # Number of images per dir - test
        self.test_num_button = tk.Button(master=self.master)
        self.test_num_entry = tk.Entry(master=self.master)

        # Num of classes
        self.num_classes_button = tk.Button(master=self.master)
        self.num_classes_entry = tk.Entry(master=self.master)

        # Image size
        self.image_size_button = tk.Button(master=self.master)
        self.image_size_entry = tk.Entry(master=self.master)

        # Batch size
        self.batch_size_button = tk.Button(master=self.master)
        self.batch_size_entry = tk.Entry(master=self.master)

        # Epochs
        self.epochs_button = tk.Button(master=self.master)
        self.epochs_entry = tk.Entry(master=self.master)

        # Optimizer
        self.optimizer_label = tk.Label(master=self.master)
        self.optimizer_listbox = tk.Listbox(master=self.master)

        # Learning rate
        self.lr_button = tk.Button(master=self.master)
        self.lr_entry = tk.Entry(master=self.master)

        # Loss function
        self.loss_function_label = tk.Label(master=self.master)
        self.loss_function_listbox = tk.Listbox(master=self.master)

        # Metrics
        self.metrics_label = tk.Label(master=self.master)
        self.metrics_listbox = tk.Listbox(master=self.master)

        # Add Layer
        self.layer_add_listbox = tk.Listbox(master=self.master)
        self.layer_add_button = tk.Button(master=self.master)

        # Compile
        self.compile_button = tk.Button(master=self.master)

        # Network frame
        self.network_frame = tk.Frame(master=self.master)
        self.canvas = tk.Canvas(master=self.network_frame)
        self.scrollbar = tk.Scrollbar(master=self.network_frame, orient='vertical', command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(master=self.canvas)

        # Training frame
        self.train_frame = tk.Frame(master=self.master)

        # Output frame
        self.output_frame = tk.Frame(master=self.train_frame)

        # Training info frame
        self.train_info_frame = tk.Frame(master=self.train_frame)

        # Training output
        self.output_text = tk.Text(master=self.output_frame)
        self.output_scrollbar = tk.Scrollbar(master=self.output_frame, orient='vertical',
                                             command=self.output_text.yview)

        # Model info
        # Optimizer info
        self.optimizer_info_type_label = tk.Label(master=self.train_info_frame)
        self.optimizer_info_label = tk.Label(master=self.train_info_frame)

        # Loss function info
        self.loss_function_info_type_label = tk.Label(master=self.train_info_frame)
        self.loss_function_info_label = tk.Label(master=self.train_info_frame)

        # Metrics info
        self.metrics_info_type_label = tk.Label(master=self.train_info_frame)
        self.metrics_info_label = tk.Label(master=self.train_info_frame)

        # Output shape info
        self.output_shape_name_label = tk.Label(master=self.train_info_frame)
        self.output_shape_info_label = tk.Label(master=self.train_info_frame)
        self.output_shape_get_button = tk.Button(master=self.train_info_frame)

        # Clear output
        self.clear_output_button = tk.Button(master=self.train_info_frame)

        # Current training job
        self.train_job_name_label = tk.Label(master=self.train_info_frame)
        self.train_job_epoch_name_label = tk.Label(master=self.train_info_frame)
        self.train_job_epoch_info_label = tk.Label(master=self.train_info_frame)
        self.train_job_batch_name_label = tk.Label(master=self.train_info_frame)
        self.train_job_batch_info_label = tk.Label(master=self.train_info_frame)
        self.train_job_acc_name_label = tk.Label(master=self.train_info_frame)
        self.train_job_acc_info_label = tk.Label(master=self.train_info_frame)
        self.train_job_loss_name_label = tk.Label(master=self.train_info_frame)
        self.train_job_loss_info_label = tk.Label(master=self.train_info_frame)
        self.train_job_cancel_button = tk.Button(master=self.train_info_frame)

        # Save/load model
        self.save_network_button = tk.Button(master=self.train_info_frame, command=self.save_network)
        self.load_network_button = tk.Button(master=self.train_info_frame, command=self.load_network)

        # Resource monitor
        self.ram_usage_label = tk.Label(master=self.train_info_frame)
        self.cpu_usage_label = tk.Label(master=self.train_info_frame)
        self.time_label = tk.Label(master=self.train_info_frame)

        # Create the widgets
        self.create_widgets()

    def on_closing(self):
        """
        Handles the tkinter exit button
        """
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.exit_app()

    def exit_app(self):
        """
        Exits the application
        """
        self.system_resource_monitor.stop_timer()
        self.master.destroy()
        self.close_thread()
        exit(1)

    def create_widgets(self):
        """
        Creates widgets and configures them
        """
        # Root
        self.master.config(bg="beige")

        # Column configure
        self.master.columnconfigure(0, weight=1)
        self.master.columnconfigure(1, weight=2)
        self.master.columnconfigure(2, weight=1)
        self.master.columnconfigure(3, weight=2)
        self.master.columnconfigure(4, weight=1)
        self.master.columnconfigure(5, weight=2)
        self.master.columnconfigure(6, weight=1)
        self.master.columnconfigure(7, weight=2)

        # Variable initialization
        self.train_job_num_var.set(0)
        self.stopped_training_var.set(False)

        # Train section
        self.train_folder_button.config(text=" Train folder ", command=self.set_train_folder, width=15,
                                        font=("Courier", 9))
        self.train_folder_label.config(textvariable=self.train_folder_var, relief="sunken", width=80,
                                       font=("Courier", 8))
        self.train_folder_var.set("N/A")
        self.train_folder_button.grid(row=0, column=0, columnspan=1, rowspan=1, sticky='ew', padx=5, pady=10)
        self.train_folder_label.grid(column=1, row=0, columnspan=7, rowspan=1, sticky='ew', padx=5, pady=10)

        # Validation section
        self.test_folder_button.config(text="Test folder", command=self.set_test_folder, width=15,
                                       font=("Courier", 9))
        self.test_folder_label.config(textvariable=self.test_folder_var, relief="sunken", width=80, font=("Courier", 8))
        self.test_folder_var.set("N/A")
        self.test_folder_button.grid(row=1, column=0, columnspan=1, rowspan=1, sticky='ew', padx=5, pady=10)
        self.test_folder_label.grid(row=1, column=1, columnspan=7, rowspan=1, sticky='ew', padx=5, pady=10)

        # Save model section
        self.save_folder_button.config(text="Save folder", command=self.set_save_location, width=15,
                                       font=("Courier", 9))
        self.save_folder_label.config(textvariable=self.save_folder_var, relief="sunken", width=80, font=("Courier", 8))
        self.save_folder_var.set("N/A")
        self.save_folder_button.grid(row=2, column=0, columnspan=1, rowspan=1, sticky='ew', padx=5, pady=10)
        self.save_folder_label.grid(row=2, column=1, columnspan=7, rowspan=1, sticky='ew', padx=5, pady=10)

        # Number of images per dir - train
        self.train_num_button.config(text="Train Size", command=self.set_train_num_per_dir, width=15,
                                     font=("Courier", 9))
        self.train_num_entry.config(relief="sunken", font=("Courier", 9), textvariable=self.train_size_var)
        self.train_size_var.set(0)
        self.train_num_button.grid(row=3, column=0, columnspan=1, rowspan=1, sticky='ew', padx=5, pady=10)
        self.train_num_entry.grid(row=3, column=1, columnspan=1, rowspan=1, sticky='ew', padx=5, pady=10)

        # Number of images per dir - test
        self.test_num_button.config(text="Test Size", command=self.set_test_num_per_dir, width=15,
                                    font=("Courier", 9))
        self.test_num_entry.config(relief="sunken", font=("Courier", 9), textvariable=self.test_size_var)
        self.test_size_var.set(0)
        self.test_num_button.grid(row=3, column=2, columnspan=1, rowspan=1, sticky='ew', padx=5, pady=10)
        self.test_num_entry.grid(row=3, column=3, columnspan=1, rowspan=1, sticky='ew', padx=5, pady=10)

        # Num of classes
        self.num_classes_button.config(text="Classes", command=self.set_num_classes, width=15, font=("Courier", 9))
        self.num_classes_entry.config(relief="sunken", font=("Courier", 9), textvariable=self.num_classes_var)
        self.num_classes_var.set(0)
        self.num_classes_button.grid(row=3, column=4, columnspan=1, rowspan=1, sticky='ew', padx=5, pady=10)
        self.num_classes_entry.grid(row=3, column=5, columnspan=1, rowspan=1, sticky='ew', padx=5, pady=10)

        # Image size
        self.image_size_button.config(text="Image Size", command=self.set_image_size, width=15,
                                      font=("Courier", 9))
        self.image_size_entry.config(relief="sunken", font=("Courier", 9), textvariable=self.img_size_var)
        self.img_size_var.set(0)
        self.image_size_button.grid(row=3, column=6, columnspan=1, rowspan=1, sticky='ew', padx=5, pady=10)
        self.image_size_entry.grid(row=3, column=7, columnspan=1, rowspan=1, sticky='ew', padx=5, pady=10)

        # Batch size
        self.batch_size_button.config(text="Batch Size", command=self.set_batch_size, width=15, font=("Courier", 9))
        self.batch_size_entry.config(relief="sunken", font=("Courier", 9), textvariable=self.batch_size_var)
        self.batch_size_var.set(0)
        self.batch_size_button.grid(row=4, column=7, columnspan=1, rowspan=1, sticky='ew', padx=5, pady=5)
        self.batch_size_entry.grid(row=5, column=7, columnspan=1, rowspan=1, sticky='ew', padx=5, pady=5)

        # Epochs
        self.epochs_button.config(text="Epochs", command=self.set_epochs, width=15, font=("Courier", 9))
        self.epochs_entry.configure(relief="sunken", font=("Courier", 9), textvariable=self.epochs_var)
        self.epochs_var.set(0)
        self.epochs_button.grid(row=6, column=7, columnspan=1, rowspan=1, sticky='ew', padx=5, pady=5)
        self.epochs_entry.grid(row=7, column=7, columnspan=1, rowspan=1, sticky='ew', padx=5, pady=5)

        # Optimizer
        self.optimizer_label.config(text="Optimizer", font=("Courier", 8), bg="beige")
        self.optimizer_listbox.config(font=("Courier", 9), height=2)
        self.optimizer_listbox.insert(1, "Adam")
        self.optimizer_listbox.insert(2, "SGD")
        # Event handler
        self.optimizer_listbox.bind("<<ListboxSelect>>", self.onselect_optimizer)
        self.optimizer_label.grid(row=8, column=7, columnspan=1, rowspan=1, sticky='ew', padx=5, pady=(5, 1))
        self.optimizer_listbox.grid(row=9, column=7, columnspan=1, rowspan=1, sticky='ew', padx=5, pady=(1, 5))

        # Learning rate
        self.lr_button.config(text="LR", font=("Courier", 9), command=self.set_lr)
        self.lr_entry.config(textvariable=self.lr_var, relief="sunken", font=("Courier", 9))
        self.lr_button.grid(row=10, column=7, columnspan=1, rowspan=1, sticky='ew', padx=5, pady=5)
        self.lr_entry.grid(row=11, column=7, columnspan=1, rowspan=1, sticky='ew', padx=5, pady=5)

        # Loss function
        self.loss_function_label.config(text="Loss function", font=("Courier", 8), bg="beige")
        self.loss_function_listbox.config(font=("Courier", 9), height=3)
        self.loss_function_listbox.insert(1, "Binary CE")
        self.loss_function_listbox.insert(2, "Categorical CE")
        self.loss_function_listbox.insert(3, "MSE")
        # Event handler
        self.loss_function_listbox.bind("<<ListboxSelect>>", self.onselect_loss_function)
        self.loss_function_label.grid(row=12, column=7, columnspan=1, rowspan=1, sticky='ew', padx=5, pady=(5, 1))
        self.loss_function_listbox.grid(row=13, column=7, columnspan=1, rowspan=1, sticky='ew', padx=5, pady=(1, 5))

        # Metrics
        self.metrics_label.config(text="Metrics", font=("Courier", 8), bg="beige")
        self.metrics_listbox.config(font=("Courier", 9), height=2)
        self.metrics_listbox.insert(1, "ACC")
        self.metrics_listbox.insert(2, "MAE")
        # Event handler
        self.metrics_listbox.bind("<<ListboxSelect>>", self.onselect_metrics)
        self.metrics_label.grid(row=14, column=7, columnspan=1, rowspan=1, sticky='ew', padx=5, pady=(5, 1))
        self.metrics_listbox.grid(row=15, column=7, columnspan=1, rowspan=1, sticky='ew', padx=5, pady=(1, 5))

        # Add Layer
        self.layer_add_button.config(text="Add Layer", command=self.add_layer, width=15, font=("Courier", 9))
        self.layer_add_listbox.config(font=("Courier", 9), height=5)
        self.layer_add_listbox.insert(1, "Input")
        self.layer_add_listbox.insert(2, "Convolution2D")
        self.layer_add_listbox.insert(3, "MaxPooling2D")
        self.layer_add_listbox.insert(4, "Dropout")
        self.layer_add_listbox.insert(5, "Flatten")
        self.layer_add_listbox.insert(6, "Dense")
        self.layer_add_listbox.insert(7, "Batch Norm.")
        self.layer_add_button.grid(row=16, column=7, columnspan=1, rowspan=1, sticky='ew', padx=5, pady=5)
        self.layer_add_listbox.grid(row=17, column=7, columnspan=1, rowspan=1, sticky='ew', padx=5, pady=5)

        # Compile
        self.compile_button.config(text="Compile&Train", command=self.compile_and_train, font=("courier", 9))
        self.compile_button.grid(row=18, column=7, columnspan=1, rowspan=1, sticky='ew', padx=0, pady=5)

        # Network container
        self.network_frame.config(bg="beige")
        self.scrollable_frame.bind(
            "<Configure>", lambda e: self.canvas.config(
                scrollregion=self.canvas.bbox("all")
            )
        )
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.config(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="right", fill="both", expand=True, padx=(0, 10), pady=(5, 10))
        self.canvas.bind("<Button-2>", self.clear_network)
        self.scrollbar.config(relief="sunken")
        self.scrollbar.pack(side="left", fill="y", padx=(10, 0), pady=(5, 10))
        self.network_frame.grid(row=4, column=0, rowspan=15, columnspan=7, sticky='news')

        # Training frame
        self.train_frame.config(bg="beige")
        self.train_frame.rowconfigure(0, weight=100)
        self.train_frame.rowconfigure(1, weight=1)
        self.train_frame.grid(row=0, column=8, columnspan=1, rowspan=19, sticky='news', padx=5, pady=(5, 0))

        # Output frame
        self.output_frame.config()
        self.output_frame.grid(row=0, column=0, columnspan=1, rowspan=1, sticky="ns", pady=(0, 5))

        # Train info frame
        self.train_info_frame.config(bg="beige")
        self.train_info_frame.grid(row=0, column=1, columnspan=1, rowspan=1, sticky="n")

        # Training output
        self.output_text.config(yscrollcommand=self.output_scrollbar.set, width=75)
        self.output_text.pack(side="right", fill="y")
        self.output_scrollbar.pack(side="left", fill="y")

        # Model info
        # Optimizer info
        self.optimizer_info_type_label.config(text="Optimizer:", width=8, anchor="center", bg="beige",
                                              font=("courier", 9))
        self.optimizer_info_label.config(textvariable=self.optimizer_var, relief="sunken", width=16,
                                         font=("courier", 8))
        self.optimizer_var.set("N/A")
        self.optimizer_info_type_label.grid(row=0, column=0, columnspan=1, rowspan=1, sticky='we', padx=(5, 0),
                                            pady=(5, 0))
        self.optimizer_info_label.grid(row=1, column=0, columnspan=1, rowspan=1, sticky='we', padx=(5, 0), pady=5)

        # Loss function info
        self.loss_function_info_type_label.config(text="Loss function:", width=12, anchor="center", bg="beige",
                                                  font=("courier", 9))
        self.loss_function_info_label.config(textvariable=self.loss_function_var, relief="sunken", width=16,
                                             font=("courier", 8))
        self.loss_function_var.set("N/A")
        self.loss_function_info_type_label.grid(row=2, column=0, columnspan=1, rowspan=1, sticky='we', padx=(5, 0),
                                                pady=(5, 0))
        self.loss_function_info_label.grid(row=3, column=0, columnspan=1, rowspan=1, sticky='we', padx=(5, 0), pady=5)

        # Metrics info
        self.metrics_info_type_label.config(text="Metrics:", width=7, anchor="center", bg="beige", font=("courier", 9))
        self.metrics_info_label.config(textvariable=self.metrics_var, relief="sunken", width=16, font=("courier", 8))
        self.metrics_var.set("N/A")
        self.metrics_info_type_label.grid(row=4, column=0, columnspan=1, rowspan=1, sticky='we', padx=(5, 0),
                                          pady=(5, 0))
        self.metrics_info_label.grid(row=5, column=0, columnspan=1, rowspan=1, sticky='we', padx=(5, 0), pady=5)

        # Output shape info
        self.output_shape_name_label.config(text="Output shape:", anchor="center", bg="beige",
                                            font=("courier", 9))
        self.output_shape_info_label.config(textvariable=self.output_shape_var, relief="sunken", width=16,
                                            font=("courier", 8))
        self.output_shape_var.set("(N/A)\n(N/A)")
        self.output_shape_get_button.config(text="Get shape", width=10, command=self.calculate_pre_final_shape,
                                            font=("courier", 9))
        self.output_shape_name_label.grid(row=6, column=0, columnspan=1, rowspan=1, sticky='we', padx=(5, 0),
                                          pady=(5, 0))
        self.output_shape_info_label.grid(row=7, column=0, columnspan=1, rowspan=1, sticky='we', padx=(5, 0), pady=5)
        self.output_shape_get_button.grid(row=8, column=0, columnspan=1, rowspan=1, sticky='we', padx=(5, 0), pady=5)

        # Clear output button
        self.clear_output_button.config(text="Clear output", width=10, command=self.clear_output, font=("courier", 9))
        self.clear_output_button.grid(row=9, column=0, columnspan=1, rowspan=1, sticky='we', padx=(5, 0), pady=5)

        # Current training job
        # Job name
        self.train_job_name_label.config(textvariable=self.train_job_id_var, anchor="center", bg="beige",
                                         font=("courier", 11))
        self.train_job_id_var.set("Job N/A")
        self.train_job_name_label.grid(row=10, column=0, columnspan=1, rowspan=1, sticky='we', padx=(5, 0),
                                       pady=(10, 0))

        # Current epoch
        self.train_job_epoch_name_label.config(text="Epoch:", anchor="center", bg="beige", font=("courier", 9))
        self.train_job_epoch_info_label.config(textvariable=self.train_job_epoch_var, relief="sunken", width=16,
                                               font=("courier", 8))
        self.train_job_epoch_var.set("0/0")
        self.train_job_epoch_name_label.grid(row=11, column=0, columnspan=1, rowspan=1, sticky='we', padx=(5, 0),
                                             pady=(5, 0))
        self.train_job_epoch_info_label.grid(row=12, column=0, columnspan=1, rowspan=1, sticky='we', padx=(5, 0),
                                             pady=5)

        # Current batch
        self.train_job_batch_name_label.config(text="Batch:", anchor="center", bg="beige", font=("courier", 9))
        self.train_job_batch_info_label.config(textvariable=self.train_job_batch_var, relief="sunken", width=16,
                                               font=("courier", 8))
        self.train_job_batch_var.set("0/0")
        self.train_job_batch_name_label.grid(row=13, column=0, columnspan=1, rowspan=1, sticky='we', padx=(5, 0),
                                             pady=(5, 0))
        self.train_job_batch_info_label.grid(row=14, column=0, columnspan=1, rowspan=1, sticky='we', padx=(5, 0),
                                             pady=5)

        # Accuracy
        self.train_job_acc_name_label.config(text="Accuracy:", anchor="center", bg="beige", font=("courier", 9))
        self.train_job_acc_info_label.config(textvariable=self.train_job_acc_var, relief="sunken", width=16,
                                             font=("courier", 8))
        self.train_job_acc_name_label.grid(row=15, column=0, columnspan=1, rowspan=1, sticky='we', padx=(5, 0),
                                           pady=(5, 0))
        self.train_job_acc_info_label.grid(row=16, column=0, columnspan=1, rowspan=1, sticky='we', padx=(5, 0), pady=5)

        # Loss
        self.train_job_loss_name_label.config(text="Loss:", anchor="center", bg="beige", font=("courier", 9))
        self.train_job_loss_info_label.config(textvariable=self.train_job_loss_var, relief="sunken", width=16,
                                              font=("courier", 8))
        self.train_job_loss_name_label.grid(row=17, column=0, columnspan=1, rowspan=1, sticky='we', padx=(5, 0),
                                            pady=(5, 0))
        self.train_job_loss_info_label.grid(row=18, column=0, columnspan=1, rowspan=1, sticky='we', padx=(5, 0), pady=5)

        # Cancel button
        self.train_job_cancel_button.config(text="Cancel training", command=self.close_thread, width=10,
                                            font=("courier", 9))
        self.train_job_cancel_button.grid(row=19, column=0, columnspan=1, rowspan=1, sticky='we', padx=(5, 0), pady=5)

        # Save/load model
        self.save_network_button.config(text="Save network", command=None, width=10, font=("courier", 9))
        self.save_network_button.grid(row=20, column=0, columnspan=1, rowspan=1, sticky='we', padx=(5, 0), pady=5)
        self.load_network_button.config(text="Load network", command=None, width=10, font=("courier", 9))
        self.load_network_button.grid(row=21, column=0, columnspan=1, rowspan=1, sticky='we', padx=(5, 0), pady=5)

        # Resource monitor
        self.ram_usage_label.config(textvariable=self.ram_usage_var, relief="sunken", width=16, font=("courier", 8))
        self.cpu_usage_label.config(textvariable=self.cpu_usage_var, relief="sunken", width=16, font=("courier", 8))
        self.time_label.config(textvariable=self.time_var, relief="sunken", width=16, font=("courier", 8))
        self.ram_usage_label.grid(row=22, column=0, columnspan=1, rowspan=1, sticky='we', padx=(5, 0), pady=(5, 5))
        self.cpu_usage_label.grid(row=23, column=0, columnspan=1, rowspan=1, sticky='we', padx=(5, 0), pady=5)
        self.time_label.grid(row=24, column=0, columnspan=1, rowspan=1, sticky='we', padx=(5, 0), pady=5)

        # Set max/min window size
        self.master.maxsize(1650, 770)
        self.master.minsize(1650, 770)

        # Resetting output text
        self.clear_output()

    def set_lr(self):
        """
        Sets the default learning rate
        """
        self.lr_var.set(0.01)
        self.write_output_info("Learning rate set to default 0.01")

    def set_epochs(self):
        """
        Sets the default, recommended number of epochs
        """
        self.epochs_var.set(10)
        self.write_output_info("Number of epochs set to default 10")

    def set_batch_size(self):
        """
        Sets the default, recommended batch size
        """
        self.batch_size_var.set(32)
        self.write_output_info("Batch size set to default 32")

    def set_num_classes(self):
        """
        Counts the number of classes in the train folder and assigns that value to the number of classes
        """
        if self.train_folder_var.get() != "N/A" and self.test_folder_var.get() != "N/A":
            i = 0
            j = 0
            for file in os.listdir(self.train_folder_var.get()):
                if os.path.isdir(self.train_folder_var.get() + "/" + file):
                    i += 1
            for file in os.listdir(self.test_folder_var.get()):
                if os.path.isdir(self.test_folder_var.get() + "/" + file):
                    j += 1
            if i > 1:
                self.num_classes_var.set(i)
                self.write_output_info("Found " + str(i) + " classes")
                if i != j:
                    self.write_output_warning("Number of train classes differs from number of test classes")
            else:
                self.write_output_error("Didn't find any classes")
        elif self.train_folder_var.get() == "N/A" and self.test_folder_var.get() == "N/A":
            self.write_output_error("No train/test folder selected")
        elif self.train_folder_var.get() == "N/A":
            self.write_output_error("No train folder selected")
        else:
            self.write_output_error("No test folder selected")

    def set_train_folder(self):
        """
        Sets the train folder if any is selected
        """
        self.train_folder_var.set(self.get_folder("Train folder"))
        if self.train_folder_var.get() == "":
            self.train_folder_var.set("N/A")
        else:
            self.write_output_info("Train folder selected")

    def set_test_folder(self):
        """
        Sets the test folder if any is selected
        """
        self.test_folder_var.set(self.get_folder("Test folder"))
        if self.test_folder_var.get() == "":
            self.test_folder_var.set("N/A")
        else:
            self.write_output_info("Test folder selected")

    def set_save_location(self):
        """
        Sets the save folder if any is selected
        """
        self.save_folder_var.set(self.get_folder("Save location"))
        if self.save_folder_var.get() == "":
            self.save_folder_var.set("N/A")
        else:
            self.write_output_info("Save folder selected")

    def set_image_size(self):
        """
        Sets the default, recommended image size
        """
        self.img_size_var.set(64)
        self.write_output_info("Image size set to default 64x64")

    def get_folder(self, folder_type=None) -> str:
        """
        Asks the user for folder location and returns it

        :param folder_type: The heading in the search window
        :return:            Folder path
        """
        folder = filedialog.askdirectory(title=folder_type, parent=self.master)
        return folder

    def set_train_num_per_dir(self):
        """
        Sets the number of samples per train directory to default, recommended, maximum possible value
        (and displays it as -1)
        """
        self.train_size_var.set(-1)
        self.write_output_info("Number of train samples per dir set to auto (max)")

    def set_test_num_per_dir(self):
        """
        Sets the number of samples per test directory to default, recommended, maximum possible value
        (and displays it as -1)
        """
        self.test_size_var.set(-1)
        self.write_output_info("Number of test samples per dir set to auto (max)")

    def onselect_optimizer(self, event):
        """
        Event handler for optimizer listbox
        Sets the optimizer according to the selected value in listbox

        :param event: Event
        """
        widget = event.widget
        try:
            index = widget.curselection()[0]
            value = widget.get(index)
            self.optimizer_var.set(value)
        except IndexError:
            pass

    def onselect_loss_function(self, event):
        """
        Event handler for loss function listbox
        Sets the loss function according to the selected value in listbox

        :param event: Event
        """
        widget = event.widget
        try:
            index = widget.curselection()[0]
            value = widget.get(index)
            self.loss_function_var.set(value)
        except IndexError:
            pass

    def onselect_metrics(self, event):
        """
        Event handler for metrics listbox
        Sets the metrics according to the selected value in listbox

        :param event: Event
        """
        widget = event.widget
        try:
            index = widget.curselection()[0]
            value = widget.get(index)
            self.metrics_var.set(value)
        except IndexError:
            pass

    def clear_output(self):
        """
        Clears the contents of the output text window
        """
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, "   <time> " + self.start_time + "\n")
        self.output_text.config(state=tk.DISABLED)
        self.auto_scroll(self.output_text)

    def write_output_info(self, text: str):
        """
        Writes an information message to the output text window

        :param text: Text to write
        """
        self.output_text.config(state=tk.NORMAL)
        self.output_text.insert(tk.END, "   <info> " + text + "\n")
        self.output_text.config(state=tk.DISABLED)
        self.auto_scroll(self.output_text)
        self.output_text.update()

    def write_output_warning(self, text: str):
        """
        Writes a warning message to the output text window

        :param text: Text to write
        """
        self.output_text.config(state=tk.NORMAL)
        self.output_text.insert(tk.END, "<warning> " + text + "\n")
        self.output_text.config(state=tk.DISABLED)
        self.auto_scroll(self.output_text)
        self.output_text.update()

    def write_output_error(self, text: str):
        """
        Writes an error message to the output text window

        :param text: Text to write
        """
        self.output_text.config(state=tk.NORMAL)
        self.output_text.insert(tk.END, "  <error> " + text + "\n")
        self.output_text.config(state=tk.DISABLED)
        self.auto_scroll(self.output_text)
        self.output_text.update()

    def write_output_model(self, text: str):
        """
        Writes the model information message to the output text window

        :param text: Text to write
        """
        self.output_text.config(state=tk.NORMAL)
        self.output_text.insert(tk.END, "  <model> " + text + "\n")
        self.output_text.config(state=tk.DISABLED)
        self.auto_scroll(self.output_text)
        self.output_text.update()

    @staticmethod
    def auto_scroll(scroll_area):
        """
        Scrolls the output text window to the bottom
        """
        scroll_area.see("end")

    def add_layer(self, layer=None, args=None):
        """
        Adds the correct layer to the network
        """
        try:
            if layer is None:
                layer = self.layer_add_listbox.get(self.layer_add_listbox.curselection()[0])
            if layer == "Convolution2D":
                self.convolution2d(self.network_position_index, layer, args)
                self.network_position_index += 1
            elif layer == "MaxPooling2D":
                self.maxpooling2d(self.network_position_index, layer, args)
                self.network_position_index += 1
            elif layer == "Dropout":
                self.dropout(self.network_position_index, layer, args)
                self.network_position_index += 1
            elif layer == "Flatten":
                self.flatten(self.network_position_index, layer)
                self.network_position_index += 1
            elif layer == "Dense":
                self.dense(self.network_position_index, layer, args)
                self.network_position_index += 1
            elif layer == "Input":
                self.input(self.network_position_index, layer, args)
                self.network_position_index += 1
            elif layer == "Batch Norm.":
                self.batch_normalization(self.network_position_index, layer)
                self.network_position_index += 1
        except IndexError:
            pass

    # TODO What does the color-depth variable even do??
    def convolution2d(self, i: int, layer: str, args: dict):
        """
        Adds a Convolution2D layer

        :param args:   Layer arguments
        :param i:      Index in the network
        :param layer:  Layer name
        """
        frame = tk.Frame(master=self.scrollable_frame, bg="white", width=self.canvas.winfo_width(), bd=5)

        j = 0
        optimizer_var = tk.StringVar()
        num_filters_var = tk.IntVar()
        filter_size_var = tk.IntVar()
        color_depth_var = tk.IntVar()

        if args is not None:
            if args.get('optimizer') != '':
                optimizer_var.set(args.get('optimizer'))
            num_filters_var.set(args.get('num_filters'))
            filter_size_var.set(args.get('filter_size'))
            color_depth_var.set(args.get('color_depth'))

        self.network[i][j] = [layer, frame, {"optimizer": optimizer_var,
                                             "num_filters": num_filters_var,
                                             "filter_size": filter_size_var,
                                             "color_depth": color_depth_var}]
        name_label = tk.Label(master=frame, text="Convolution2D", bg="white")
        j += 1
        self.network[i][j] = [name_label]
        num_filters_label = tk.Label(master=frame, text="Num filters: ", anchor="w", bg="white")
        filter_size_label = tk.Label(master=frame, text="Filter size: ", anchor="w", bg="white")
        j += 1
        self.network[i][j] = [num_filters_label, filter_size_label]
        num_filters_entry = tk.Entry(master=frame, bg="white", textvariable=num_filters_var)
        filter_size_entry = tk.Entry(master=frame, bg="white", textvariable=filter_size_var)
        j += 1
        self.network[i][j] = [num_filters_entry, filter_size_entry]
        button_relu = tk.Radiobutton(master=frame, text="Relu", variable=optimizer_var, value="Relu", anchor="w",
                                     bg="white")
        button_softmax = tk.Radiobutton(master=frame, text="Softmax", variable=optimizer_var, value="Softmax",
                                        anchor="w", bg="white")
        j += 1
        self.network[i][j] = [button_relu, button_softmax]
        if self.first_layer:
            j += 1
            self.add_first_layer(frame, color_depth_var, i, j)
            self.first_layer = False
        self.add_remove_move_buttons(frame, i, j)
        self.grid_layer(i)
        frame.grid(row=self.network_position_index, column=0, padx=(5, 5), pady=(5, 5), sticky='w')
        self.write_output_info("Added Convolution 2D")

    def maxpooling2d(self, i: int, layer: str, args: dict):
        """
        Adds a MaxPooling2D layer

        :param args:  Layer arguments
        :param i:     Index int the network
        :param layer: Layer name
        """
        frame = tk.Frame(master=self.scrollable_frame, bg="white", width=self.canvas.winfo_width(), bd=5)

        pool_size_var = tk.IntVar()
        color_depth_var = tk.IntVar()

        if args is not None:
            pool_size_var.set(args.get('pool_size'))
            color_depth_var.set(args.get('color_depth'))

        j = 0
        self.network[i][j] = [layer, frame, {"pool_size": pool_size_var,
                                             "color_depth": color_depth_var}]
        name_label = tk.Label(master=frame, text="MaxPooling2D", bg="white")
        j += 1
        self.network[i][j] = [name_label]
        pool_size_label = tk.Label(master=frame, text="Pool size: ", anchor="w", bg="white")
        pool_size_entry = tk.Entry(master=frame, textvariable=pool_size_var, bg="white")
        j += 1
        self.network[i][j] = [pool_size_label, pool_size_entry]
        if self.first_layer:
            j += 1
            self.add_first_layer(frame, color_depth_var, i, j)
            self.first_layer = False
        self.add_remove_move_buttons(frame, i, j)
        self.grid_layer(i)

        frame.grid(row=self.network_position_index, column=0, padx=(5, 5), pady=(5, 5), sticky='w')
        self.write_output_info("Added Max Pooling 2D")

    def input(self, i: int, layer: str, args: dict):
        """
        Adds an Input layer

        :param args:  Layer arguments
        :param i:     Index in the network
        :param layer: Layer name
        """
        if self.first_layer:
            frame = tk.Frame(master=self.scrollable_frame, bg="white", width=self.canvas.winfo_width(), bd=5)

            color_depth_var = tk.IntVar()

            if args is not None:
                color_depth_var.set(args.get('color_depth'))

            j = 0
            self.network[i][j] = [layer, frame, {"color_depth": color_depth_var}]
            name_label = tk.Label(master=frame, text="Input", bg="white")
            j += 1
            self.network[i][j] = [name_label]

            j += 1
            self.add_first_layer(frame, color_depth_var, i, j)
            self.first_layer = False
            self.add_remove_move_buttons(frame, i, j)
            self.grid_layer(i)

            frame.grid(row=self.network_position_index, column=0, padx=(5, 5), pady=(5, 5), sticky='w')
            self.write_output_info("Added Input")
        else:
            self.network_position_index -= 1
            self.write_output_error("Cannot add input layer again")

    def dropout(self, i: int, layer: str, args: dict):
        """
        Adds a Dropout layer

        :param args:  Layer arguments
        :param i:     Index in the network
        :param layer: Layer name
        """
        if not self.first_layer:
            frame = tk.Frame(master=self.scrollable_frame, bg="white", width=self.canvas.winfo_width(), bd=5)
            j = 0
            dropout_rate_var = tk.DoubleVar()

            if args is not None:
                dropout_rate_var.set(args.get('dropout_rate'))

            self.network[i][j] = [layer, frame, {"dropout_rate": dropout_rate_var}]
            name_label = tk.Label(master=frame, text="Dropout", bg="white")
            j += 1
            self.network[i][j] = [name_label]
            dropout_rate_label = tk.Label(master=frame, text="Dropout rate: ", anchor="w", bg="white")
            dropout_rate_entry = tk.Entry(master=frame, textvariable=dropout_rate_var, bg="white")
            j += 1
            self.network[i][j] = [dropout_rate_label, dropout_rate_entry]
            self.add_remove_move_buttons(frame, i, j)
            self.grid_layer(i)

            frame.grid(row=self.network_position_index, column=0, padx=(5, 5), pady=(5, 5), sticky='w')
            self.write_output_info("Added Dropout")
        else:
            self.network_position_index -= 1
            self.write_output_error("Cannot add Dropout on the first layer")

    def flatten(self, i: int, layer: str):
        """
        Adds a Flatten layer

        :param i:     Index in the network
        :param layer: Layer name
        """
        if not self.first_layer:
            frame = tk.Frame(master=self.scrollable_frame, bg="white", width=self.canvas.winfo_width(), bd=5)
            j = 0
            self.network[i][j] = [layer, frame]
            name_label = tk.Label(master=frame, text="Flatten", bg="white")
            j += 1
            self.network[i][j] = [name_label]
            self.add_remove_move_buttons(frame, i, j)
            self.grid_layer(i)

            frame.grid(row=self.network_position_index, column=0, padx=(5, 5), pady=(5, 5), sticky='w')
            self.write_output_info("Added Flatten")
        else:
            self.network_position_index -= 1
            self.write_output_warning("Cannot add Flatten on the first layer")

    def dense(self, i: int, layer: str, args: dict):
        """
        Adds a Dense layer

        :param args:  Layer arguments
        :param i:     Index in the network
        :param layer: Layer name
        """
        if not self.first_layer:
            frame = tk.Frame(master=self.scrollable_frame, bg="white", width=self.canvas.winfo_width(), bd=5)
            j = 0
            optimizer_var = tk.StringVar()
            units_var = tk.IntVar()

            if args is not None:
                units_var.set(args.get('units'))
                if args.get('optimizer') != '':
                    optimizer_var.set(args.get('optimizer'))

            self.network[i][j] = [layer, frame, {"units": units_var,
                                                 "optimizer": optimizer_var}]
            name_label = tk.Label(master=frame, text="Dense", bg="white")
            j += 1
            self.network[i][j] = [name_label]
            units_label = tk.Label(master=frame, text="Units: ", anchor="w", bg="white")
            units_entry = tk.Entry(master=frame, textvariable=units_var, bg="white")
            j += 1
            self.network[i][j] = [units_label, units_entry]
            button_relu = tk.Radiobutton(master=frame, text="Relu", variable=optimizer_var, value="Relu", anchor="w",
                                         bg="white")
            button_softmax = tk.Radiobutton(master=frame, text="Softmax", variable=optimizer_var, value="Softmax",
                                            anchor="w", bg="white")
            button_sigmoid = tk.Radiobutton(master=frame, text="Sigmoid", variable=optimizer_var, value="Sigmoid",
                                            anchor="w", bg="white")
            j += 1
            self.network[i][j] = [button_relu, button_softmax, button_sigmoid]
            self.add_remove_move_buttons(frame, i, j)
            self.grid_layer(i)

            frame.grid(row=self.network_position_index, column=0, padx=(5, 5), pady=(5, 5), sticky='w')
            self.write_output_info("Added Dense")
        else:
            self.network_position_index -= 1
            self.write_output_warning("Cannot add Dense on the first layer")

    def batch_normalization(self, i: int, layer: str):
        """
        Adds a Batch Normalization layer

        :param i:     Index in the network
        :param layer: Layer name
        """
        if not self.first_layer:
            frame = tk.Frame(master=self.scrollable_frame, bg="white", width=self.canvas.winfo_width(), bd=5)
            j = 0
            self.network[i][j] = [layer, frame]
            name_label = tk.Label(master=frame, text="Batch Normalization", bg="white")
            j += 1
            self.network[i][j] = [name_label]
            self.add_remove_move_buttons(frame, i, j)
            self.grid_layer(i)

            frame.grid(row=self.network_position_index, column=0, padx=(5, 5), pady=(5, 5), sticky='w')
            self.write_output_info("Added Batch Normalization")
        else:
            self.network_position_index -= 1
            self.write_output_warning("Cannot add Batch Normalization on the first layer")

    def add_remove_move_buttons(self, frame: tk.Frame, i: int, j: int):
        """
        Adds remove and move up/down buttons to a frame

        :param frame: Frame we are adding to
        :param i:     Current layer index in the network
        :param j:     Column in the frame
        """
        j += 1
        remove_button = tk.Button(master=frame, text="Remove layer", bg="white")
        remove_button.bind("<Button-1>", self.destroy_layer)
        self.network[i][j] = [remove_button]
        j += 1
        move_up_button = tk.Button(master=frame, text="↑", bg="white")
        move_up_button.bind("<Button-1>", self.move_layer_up)
        move_down_button = tk.Button(master=frame, text="↓", bg="white")
        move_down_button.bind("<Button-1>", self.move_layer_down)
        self.network[i][j] = [move_up_button, move_down_button]

    def add_first_layer(self, frame: tk.Frame, color_depth_var: tk.IntVar, i: int, j: int):
        """
        Add the color depth property to the first layer

        :param frame:           Which frame we are adding the element to
        :param color_depth_var: Variable to store the color depth value
        :param i:               Index of the first dimension
        :param j:               Index of the second dimension
        """
        color_depth_label = tk.Label(master=frame, text="Color depth (1 / 3):", bg="white")
        color_depth_entry = tk.Entry(master=frame, bg="white", textvariable=color_depth_var)
        self.network[i][j] = [color_depth_label, color_depth_entry]

    def grid_layer(self, i):
        """
        Adds a layer to the grid

        :param i: Index of the first dimension
        """
        column = -1
        skip = False
        for j in self.network[i]:
            row = 0
            if skip:
                for k in j:
                    k.grid(row=row, column=column, sticky='ew', padx=5)
                    row += 1
            column += 1
            skip = True

    def destroy_layer(self, event, rm_index=-1):
        """
        Destroys all of the widgets on the layer and the layer itself

        :param event:    Event
        :param rm_index: Index of the layer we are removing if provided
        """
        if rm_index != -1:
            for x in self.network[rm_index]:
                for y in x:
                    try:
                        y.destroy()
                    except AttributeError:
                        pass
        else:
            widget = event.widget
            rm_index = 0
            for i in range(self.network.__len__()):
                try:
                    self.network[i].index([widget])
                    rm_index = i
                    for x in self.network[i]:
                        for y in x:
                            try:
                                y.destroy()
                            except AttributeError:
                                pass
                except ValueError:
                    pass
            self.move_layers_up(rm_index)
            if rm_index == 0:
                self.first_layer = True

    def move_layers_up(self, removed_layer_index: int):
        """
        Moves all the layers when destroying a layer in the middle of other elements to arrange them properly again

        :param removed_layer_index: Removed layer index in the network
        """
        bottom = removed_layer_index
        top = self.network_position_index
        if bottom != top:
            try:
                for i in range(bottom, top):
                    self.network[i] = self.network[i + 1]
                    self.network[i][0][1].grid(row=i, column=0, padx=(5, 5),
                                               pady=(5, 5), sticky='w')
            except:
                pass
            self.destroy_layer(None, top)
            self.network_position_index -= 1
            self.network[top] = [[] for x in range(7)]

    def move_layer_up(self, event):
        """
        Moves a layer up the network

        :param event: Event
        """
        widget = event.widget
        widget_index = 0
        for i in range(self.network.__len__()):
            for j in range(self.network[i].__len__()):
                try:
                    # Finding the widget location
                    self.network[i][j].index(widget)
                    widget_index = i
                except:
                    pass
        if 1 < widget_index:
            self.swap_layers(widget_index, widget_index - 1)

    def move_layer_down(self, event):
        """
        Moves the layer down the network

        :param event: Event
        """
        widget = event.widget
        widget_index = 0
        for i in range(self.network.__len__()):
            for j in range(self.network[i].__len__()):
                try:
                    # Finding the widget location
                    self.network[i][j].index(widget)
                    widget_index = i
                except:
                    pass
        if 0 < widget_index < self.network_position_index - 1:
            self.swap_layers(widget_index, widget_index + 1)

    def swap_layers(self, i: int, j: int):
        """
        Swaps 2 layers if possible

        :param i: Layer 1 index in the network
        :param j: Layer 2 index in the network
        """
        temp = self.network[i]
        self.network[i] = self.network[j]
        self.network[j] = temp
        self.update_swaps(i, j)

    def update_swaps(self, i, j):
        """
        Updates the swapped layers

        :param i: Layer 1 index in the network
        :param j: Layer 2 index in the network
        """
        self.network[i][0][1].grid(row=i, column=0, padx=(5, 5), pady=(5, 5), sticky='w')
        self.network[j][0][1].grid(row=j, column=0, padx=(5, 5), pady=(5, 5), sticky='w')

    def compile_and_train(self):
        """
        Compiles and trains the model
        """
        if self.check_model_validity():
            self.write_output_info("Model checked: Success!")
            self.build_and_compile_network()
            # Moving the calculations to another thread
            self.train_job_thread = Thread(target=self.fit_network)
            self.train_job_thread.start()
        else:
            self.write_output_error("Missing model information found, cannot compile")

    def check_model_validity(self) -> bool:
        """
        Checks if the model's layers and parameters are all valid
        """
        if self.network_position_index < 2:
            self.write_output_warning("Model is too small, won't compile")
            return False
        for i in range(self.network_position_index):
            valid = True
            if self.network[i][0][0] == "Input":
                valid = self.check_input(i)
            elif self.network[i][0][0] == "Convolution2D":
                valid = self.check_convolution2d(i)
            elif self.network[i][0][0] == "MaxPooling2D":
                valid = self.check_maxpooling2d(i)
            elif self.network[i][0][0] == "Dropout":
                valid = self.check_dropout(i)
            elif self.network[i][0][0] == "Flatten":
                valid = self.check_flatten()
            elif self.network[i][0][0] == "Dense":
                valid = self.check_dense(i)
            elif self.network[i][0][0] == "Batch Norm.":
                valid = self.check_batch_normalization()
            if not valid:
                return False
        if self.train_folder_var.get() == "N/A":
            return False
        if self.test_folder_var.get() == "N/A":
            return False
        if self.save_folder_var.get() == "N/A":
            return False
        if not (self.train_size_var.get() == -1 or self.train_size_var.get() > 0):
            return False
        if not (self.test_size_var.get() == -1 or self.test_size_var.get() > 0):
            return False
        if self.num_classes_var.get() < 2:
            return False
        if self.img_size_var.get() < 2:
            return False
        if self.batch_size_var.get() < 1:
            return False
        if self.epochs_var.get() < 1:
            return False
        if self.lr_var.get() <= 0.0:
            return False
        if self.optimizer_var.get() not in ["Adam", "SGD"]:
            return False
        if self.loss_function_var.get() not in ["Binary CE", "Categorical CE", "MSE"]:
            return False
        if self.metrics_var.get() not in ["ACC", "MAE"]:
            return False
        return True

    def check_input(self, i: int) -> bool:
        """
        Checks for validity of the input layer

        :param i: Layer index in the network
        :return:  Is layer valid
        """
        return self.network[i][0][2].get("color_depth").get() == 1 or \
               self.network[i][0][2].get("color_depth").get() == 3

    def check_convolution2d(self, i: int) -> bool:
        """
        Checks for validity of the convolution2d layer

        :param i: Layer index in the network
        :return:  Is layer valid
        """
        valid = True
        while True:
            if i == 0:
                if not self.check_input(i):
                    valid = False
                    break
            if self.network[i][0][2].get("num_filters").get() <= 0:
                valid = False
                break
            if self.network[i][0][2].get("filter_size").get() <= 1 or \
                    self.network[i][0][2].get("filter_size").get() > 10:
                valid = False
                break
            if self.network[i][0][2].get("optimizer").get() not in ["Relu", "Softmax"]:
                valid = False
                break
            if valid:
                break
        return valid

    def check_maxpooling2d(self, i: int) -> bool:
        """
        Checks for validity of the maxpooling2d layer

        :param i: Layer index in the network
        :return:  Is layer valid
        """
        valid = True
        while True:
            if i == 0:
                if not self.check_input(i):
                    valid = False
                    break
            if self.network[i][0][2].get("pool_size").get() < 2 or \
                    self.network[i][0][2].get("pool_size").get() > 10:
                valid = False
                break
            if valid:
                break
        return valid

    def check_dropout(self, i: int) -> bool:
        """
        Checks for validity of the dropout layer

        :param i: Layer index in the network
        :return:  Is layer valid
        """
        return 0.0 < self.network[i][0][2].get("dropout_rate").get() <= 1.0

    @staticmethod
    def check_flatten() -> bool:
        """
        Checks for validity of the flatten layer

        :return: Is layer valid
        """
        return True

    def check_dense(self, i: int) -> bool:
        """
        Checks for validity of the dense layer

        :param i: Layer index in the network
        :return:  Is layer valid
        """
        valid = True
        while True:
            if self.network[i][0][2].get("optimizer").get() not in ["Relu", "Softmax", "Sigmoid"]:
                valid = False
                break
            if self.network[i][0][2].get("units").get() <= 0:
                valid = False
                break
            break
        return valid

    @staticmethod
    def check_batch_normalization() -> bool:
        """
        Checks for validity of the batch normalization layer

        :return: Is layer valid
        """
        return True

    def build_and_compile_network(self):
        """
        Builds the network and compiles it
        """
        self.model = Sequential()
        input_shape = None
        input_layer = False
        for i in range(self.network_position_index):
            if i == 0:
                # Input shape needed only on the first layer
                input_shape = (self.img_size_var.get(), self.img_size_var.get(),
                               self.network[i][0][2].get("color_depth").get())
                input_layer = True
            if input_layer and self.network[i][0][0] != "Input":
                if self.network[i][0][0] == "Convolution2D":
                    optimizer = None
                    if self.network[i][0][2].get("optimizer").get() == "Relu":
                        optimizer = "relu"
                    elif self.network[i][0][2].get("optimizer").get() == "Softmax":
                        optimizer = "softmax"
                    self.model.add(Convolution2D(self.network[i][0][2].get("num_filters").get(),
                                                 (self.network[i][0][2].get("filter_size").get(),
                                                  self.network[i][0][2].get("filter_size").get()),
                                                 input_shape=input_shape,
                                                 activation=optimizer))
                if self.network[i][0][0] == "MaxPooling2D":
                    self.model.add(MaxPooling2D(pool_size=(self.network[i][0][2].get("pool_size").get(),
                                                           self.network[i][0][2].get("pool_size").get()),
                                                input_shape=input_shape))
                input_layer = False
            else:
                if self.network[i][0][0] == "Convolution2D":
                    optimizer = None
                    if self.network[i][0][2].get("optimizer").get() == "Relu":
                        optimizer = "relu"
                    elif self.network[i][0][2].get("optimizer").get() == "Softmax":
                        optimizer = "softmax"
                    self.model.add(Convolution2D(self.network[i][0][2].get("num_filters").get(),
                                                 (self.network[i][0][2].get("filter_size").get(),
                                                  self.network[i][0][2].get("filter_size").get()),
                                                 activation=optimizer))
                elif self.network[i][0][0] == "MaxPooling2D":
                    self.model.add(MaxPooling2D(pool_size=(self.network[i][0][2].get("pool_size").get(),
                                                           self.network[i][0][2].get("pool_size").get())))
                elif self.network[i][0][0] == "Dropout":
                    self.model.add(Dropout(self.network[i][0][2].get("dropout_rate").get()))
                elif self.network[i][0][0] == "Flatten":
                    self.model.add(Flatten())
                elif self.network[i][0][0] == "Batch Norm.":
                    self.model.add(BatchNormalization())
                elif self.network[i][0][0] == "Dense":
                    optimizer = None
                    if self.network[i][0][2].get("optimizer").get() == "Relu":
                        optimizer = "relu"
                    elif self.network[i][0][2].get("optimizer").get() == "Softmax":
                        optimizer = "softmax"
                    elif self.network[i][0][2].get("optimizer").get() == "Sigmoid":
                        optimizer = "sigmoid"
                    self.model.add(Dense(units=self.network[i][0][2].get("units").get(),
                                         activation=optimizer))

        # Optimizer
        optimizer = None
        if self.optimizer_var.get() == "Adam":
            optimizer = Adam(lr=self.lr_var.get())
        elif self.optimizer_var.get() == "SGD":
            optimizer = SGD(lr=self.lr_var.get())
        # Loss function
        loss = None
        if self.loss_function_var.get() == "Binary CE":
            loss = "binary_crossentropy"
        elif self.loss_function_var.get() == "Categorical CE":
            loss = "categorical_crossentropy"
        elif self.loss_function_var.get() == "MSE":
            loss = "mean_squared_error"
        # Metrics
        metrics = None
        if self.metrics_var.get() == "ACC":
            metrics = "acc"
        elif self.metrics_var.get() == "MAE":
            metrics = "mae"
        self.model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])
        self.model.summary(print_fn=self.write_output_model)
        self.write_output_info("Model compiled successfully!")
        self.output_text.update()

    def fit_network(self):
        """
        Fits the network - starts the training process
        """
        train_datagen = ImageDataGenerator(rescale=1. / 255,
                                           shear_range=0.2,
                                           zoom_range=0.2,
                                           horizontal_flip=False)

        test_datagen = ImageDataGenerator(rescale=1. / 255)

        # Selects class mode
        # Class mode depending on the number of classes
        if self.num_classes_var.get() > 2:
            class_mode = "categorical"
        else:
            class_mode = "binary"

        # Selects color mode
        if self.network[0][0][2].get("color_depth").get() == 1:
            color_mode = "grayscale"
        else:
            color_mode = "rgb"

        self.train_job_id_var.set("Job " + str(self.train_job_num_var.get()))

        self.train_job_cancel_button.config(state=tk.DISABLED)
        self.compile_button.config(state=tk.DISABLED)
        self.load_network_button.config(state=tk.DISABLED)
        self.save_network_button.config(state=tk.DISABLED)
        self.write_output_info("Getting train images...")
        train = train_datagen.flow_from_directory(self.train_folder_var.get(),
                                                  target_size=(self.img_size_var.get(), self.img_size_var.get()),
                                                  batch_size=self.batch_size_var.get(),
                                                  class_mode=class_mode,
                                                  color_mode=color_mode)

        self.write_output_info("Getting test images...")
        test = test_datagen.flow_from_directory(self.test_folder_var.get(),
                                                target_size=(self.img_size_var.get(), self.img_size_var.get()),
                                                batch_size=self.batch_size_var.get(),
                                                class_mode=class_mode,
                                                color_mode=color_mode)
        self.train_job_cancel_button.config(state=tk.NORMAL)

        if self.train_size_var.get() == -1:
            train_samples = train.__len__()
            self.train_len_var.set(train_samples)
        else:
            train_samples = self.train_size_var.get() * self.num_classes_var.get()
            self.train_len_var.set(train_samples)

        if self.test_size_var.get() == -1:
            test_samples = test.__len__()
        else:
            test_samples = self.test_size_var.get() * self.num_classes_var.get()

        try:
            self.write_output_model("Training started")
            self.model.fit_generator(train,
                                     steps_per_epoch=train_samples,
                                     epochs=self.epochs_var.get(),
                                     validation_data=test,
                                     validation_steps=test_samples,
                                     verbose=0,
                                     workers=8,
                                     callbacks=[self.callbacks])
            if not self.stopped_training_var.get():
                self.write_output_model("Training finished!")
        except ValueError:
            self.write_output_error("Model training failed: the model must have faulty logic")
        if not self.stopped_training_var.get():
            self.compile_button.config(state=tk.NORMAL)
            self.save_network_button.config(state=tk.NORMAL)
            self.load_network_button.config(state=tk.NORMAL)
            save_name = "model" + datetime.datetime.now().strftime('__%Y_%m_%d__%H_%M_%S') + ".h5"
            self.model.save(self.save_folder_var.get() + "/" + save_name)

    def close_thread(self):
        """
        Stops the training process, closes the thread
        """
        if self.train_job_epoch_var.get() != "0/0":
            self.write_output_warning("Training was canceled")
            self.compile_button.config(state=tk.NORMAL)
            self.load_network_button.config(state=tk.NORMAL)
            self.save_network_button.config(state=tk.NORMAL)
            self.train_job_id_var.set("Job N/A")
            self.train_job_epoch_var.set("0/0")
            self.train_job_batch_var.set("0/0")
            self.train_job_acc_var.set(0.0)
            self.train_job_loss_var.set(0.0)
            self.train_job_num_var.set(self.train_job_num_var.get() + 1)
            self.stopped_training_var.set(True)

    def calculate_pre_final_shape(self):
        """
        Calculates the output shape right before the flatten layer
        """
        size = self.img_size_var.get()
        max_convolution_size = -1
        for i in range(self.network_position_index):
            if self.network[i][0][0] == "Convolution2D":
                size -= self.network[i][0][2].get("filter_size").get() - 1
                max_convolution_size = self.network[i][0][2].get("num_filters").get()
            if self.network[i][0][0] == "MaxPooling2D":
                size /= self.network[i][0][2].get("pool_size").get()
        flatten_size = int(size) * int(size) * max_convolution_size
        self.output_shape_var.set("(0, " + str(int(size)) + ", " + str(int(size)) + ", " +
                                  str(int(max_convolution_size)) + ")\n" + str(flatten_size))

    def save_network(self):
        """
        Saves the network locally in a .nets file
        """
        try:
            if not self.first_layer:
                save_path = "./nets/nets" + datetime.datetime.now().strftime('_%Y_%m_%d__%H_%M_%S')
                modified_network = []
                self.write_output_info("Transforming the network...")
                for i in range(self.network_position_index):
                    args = {}
                    try:
                        if type(self.network[i][0][2]) is dict:
                            for key in self.network[i][0][2]:
                                try:
                                    args[key] = self.network[i][0][2].get(key).get()
                                except tk.TclError:
                                    if key == 'dropout_rate':
                                        args[key] = 0.0
                                    else:
                                        args[key] = 0
                    except IndexError:
                        pass
                    layer = [self.network[i][0][0], args]
                    modified_network.append(layer)
                np.save(save_path, modified_network, allow_pickle=True)
                self.write_output_info("Saved")
                os.renames("{0}.npy".format(save_path), "{0}.nets".format(save_path[0:len(save_path) - 3]))
        except tk.TclError:
            self.write_output_error("Failed to save")

    def load_network(self):
        """
        Loads a network from a .npy file
        """
        file_path = filedialog.askopenfilename(title="Select the .nets file", parent=self.master,
                                               filetypes=(("npy files", "*.nets"),))
        if file_path != "":
            usable_file_path = "{0}.npy".format("./nets/file")
            os.renames(file_path, usable_file_path)
            self.write_output_info("A network was loaded:")
            self.write_output_info(file_path)
            modified_network = np.load(usable_file_path, allow_pickle=True)
            self.clear_network(None)
            for _type in modified_network:
                self.add_layer(layer=_type[0], args=_type[1])
            os.renames(usable_file_path, file_path)

    def clear_network(self, event=None):
        """
        Removes all the layers
        """
        for index in range(self.network_position_index):
            self.destroy_layer(None, index)
            self.network_position_index -= 1
            self.network[index] = [[] for x in range(7)]
        self.first_layer = True

    def about(self):
        """
        About pop-up message
        """
        # About message
        messagebox.showinfo(title="About", message="CNN Training GUI\n"
                                                   "\nSimple, easy to use GUI for training a CNN model."
                                                   " Only theoretical knowledge about Convolutional Neural Networks is"
                                                   " needed to work with this program. The program was created for the"
                                                   " purpose of school and for the ease of use while creating a CNN "
                                                   "model without actually writing any code.\n"
                                                   "\nBackend: PlaidML (https://github.com/plaidml/plaidml)\n"
                                                   "\nAuthor: David Jerman"
                                                   "\nVersion: 2020.04.02", master=self.master)

    def help(self):
        """
        Help pop-up message
        """
        # Help message
        messagebox.showinfo(title="Help", message="How to work with this program?\n"
                                                  "\nIt is simple to use really, only buttons need to be clicked to"
                                                  " work with the program.\n"
                                                  "\nClicking a button next to an entry will result in the entry"
                                                  " being assigned a default, recommended value. Custom value can be"
                                                  " used by just typing it inside the entry field.\n"
                                                  "The program will check for any possible mistakes before"
                                                  " training, but it won't check the following:\n"
                                                  " > Validity of training and test samples\n"
                                                  " > Possible issues with model logic (user is responsible to"
                                                  "              make sure, there aren't any)\n"
                                                  "\nFor more information contact the developer (David Jerman).",
                            master=self.master)


class Callbacks(keras.callbacks.Callback):
    """
    Custom callbacks
    """

    def __init__(self, output_text=None, app_ref=None, acc_var=None, loss_var=None, epoch_var=None,
                 epochs_total_var=None, batch_var=None, train_len_var=None, stopped_training=None):
        """
        Initialization

        :param output_text:      Output tk.Text
        :param app_ref:          Main tkinter app
        :param acc_var:          Accuracy tkinter variable
        :param loss_var:         Loss tkinter variable
        :param epoch_var:        Current tkinter epoch variable
        :param epochs_total_var: Total epochs tkinter var
        :param batch_var:        Batch tkinter variable
        :param train_len_var:    Training data length tkinter variable
        :param stopped_training: Stopped training tkinter variable
        """
        super().__init__()
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []
        self.i = 1
        self.j = 1
        self.acc_avg = 0
        self.loss_avg = 0
        self.start_time = 0
        self.end_time = 0
        self.output_text = output_text
        self.app = app_ref
        self.acc_var = acc_var
        self.loss_var = loss_var
        self.epoch_var = epoch_var
        self.epochs_total_var = epochs_total_var
        self.batch_var = batch_var
        self.train_len_var = train_len_var
        self.stopped_training = stopped_training

    def on_train_begin(self, logs=None):
        """
        Method called once on the beginning of training

        :param logs: Logs
        """
        self.stopped_training.set(False)
        self.model.stop_training = False
        self.start_time = time.time()
        self.i = 1

    def on_epoch_end(self, batch, logs=None):
        """
        Method called on the end of each epoch
        Prints out all the important data

        :param batch: Batch
        :param logs:  Logs
        """
        if not self.stopped_training.get():
            self.end_time = time.time()
            if logs is None:
                logs = {}
            self.write_output("")
            self.write_output("Epoch " + str(self.i))
            self.write_output("Time elapsed [s]: " + str(round(self.end_time - self.start_time, 4)))
            self.write_output("Loss:             " + str(round(logs.get('loss'), 4)))
            self.write_output("Val loss:         " + str(round(logs.get('val_loss'), 4)))
            self.write_output("Acc:              " + str(round(logs.get('acc'), 4)))
            self.write_output("Val acc:          " + str(round(logs.get('val_acc'), 4)))
            self.i += 1
        self.model.stop_training = False

    def on_epoch_begin(self, epoch, logs=None):
        """
        Method called on the beginning of each epoch
        Sets the current epoch

        :param epoch: Epoch
        :param logs:  Logs
        """
        self.epoch_var.set(str(self.i) + "/" + str(self.epochs_total_var.get()))
        self.j = 1

    def on_batch_end(self, batch, logs=None):
        """
        Method called on the end of each batch
        Sets the current batch

        :param batch: Batch
        :param logs:  Logs
        """
        if self.j % 2 == 0:
            self.batch_var.set(str(self.j) + "/" + str(self.train_len_var.get()))
            self.acc_avg += logs.get('acc')
            self.loss_avg += logs.get('loss')
        if self.j % 20 == 0:
            self.acc_var.set(round(self.acc_avg / 10, 4))
            self.loss_var.set(round(self.loss_avg / 10, 4))
            self.acc_avg = 0
            self.loss_avg = 0
        if self.stopped_training.get():
            self.model.stop_training = True
            self.batch_var.set("0/0")
            self.epoch_var.set("0/0")
        self.j += 1

    def write_output(self, text: str):
        """
        Writes some text in the output_text widget

        :param text: Text to write
        """
        self.output_text.config(state=tk.NORMAL)
        self.output_text.insert(tk.END, "  <model> " + text + "\n")
        self.output_text.config(state=tk.DISABLED)
        self.app.auto_scroll(self.output_text)
        self.output_text.update()


class SystemResourceMonitor:
    """
    System resource monitor with CPU and RAM usage, time
    """

    def __init__(self, cpu_usage_var=None, ram_usage_var=None, time_var=None):
        """
        Initialization

        :param cpu_usage_var: CPU usage tkinter variable
        :param ram_usage_var: RAM usage tkinter variable
        :param time_var:      Time tkinter variable
        """
        self.cpu_usage_var = cpu_usage_var
        self.ram_usage_var = ram_usage_var
        self.time_var = time_var
        self.timer = RepeatedTimer(1, self.update_monitoring)

    def stop_timer(self):
        """
        Stops the timer
        """
        self.timer.stop()

    def update_monitoring(self):
        """
        Updated the monitoring labels
        """
        self.ram_usage_var.set("RAM: " + str(sm.virtual_memory().percent) + "%")
        self.cpu_usage_var.set("CPU: " + str(sm.cpu_percent()) + "%")
        self.time_var.set(datetime.datetime.now().strftime('%H:%M:%S'))


# https://stackoverflow.com/questions/474528/what-is-the-best-way-to-repeatedly-execute-a-function-every-x-seconds
class RepeatedTimer(object):
    def __init__(self, interval, function, *args, **kwargs):
        self._timer = None
        self.interval = interval
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.is_running = False
        self.start()

    def _run(self):
        self.is_running = False
        self.start()
        self.function(*self.args, **self.kwargs)

    def start(self):
        if not self.is_running:
            self._timer = Timer(self.interval, self._run)
            self._timer.start()
            self.is_running = True

    def stop(self):
        self._timer.cancel()
        self.is_running = False


# Main tkinter loop
if __name__ == "__main__":
    root = tk.Tk()
    root.title("CNN Training GUI")
    callback = Callbacks()
    app = Application(master=root, callbacks=callback)
    callback.__init__(output_text=app.output_text, app_ref=app, acc_var=app.train_job_acc_var,
                      loss_var=app.train_job_loss_var, epoch_var=app.train_job_epoch_var,
                      epochs_total_var=app.epochs_var, train_len_var=app.train_len_var,
                      batch_var=app.train_job_batch_var, stopped_training=app.stopped_training_var)
    app.mainloop()
