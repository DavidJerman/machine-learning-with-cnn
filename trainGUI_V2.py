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
        self.network = [[[] for _ in range(8)] for __ in range(50)]
        self.model = None
        self.train_job_thread = None

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
        self.close_process()
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

    def get_folder(self, window_title=None) -> str:
        """
        Asks the user for folder location and returns it

        :param window_title: The heading in the search window
        :return:            Folder path
        """
        folder = filedialog.askdirectory(title=window_title, parent=self.master)
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
            # TODO: Error message
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
            # TODO: Error message
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
            # TODO: Error message
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

    def write_output(self, msg_type: str, text: str):
        """
        Writes a message to the output text window

        :param msg_type: Type of the displaying message
        :param text:     Text to write
        """
        self.output_text.config(state=tk.NORMAL)
        self.output_text.insert(tk.END, msg_type + text + "\n")
        self.output_text.config(state=tk.DISABLED)
        self.auto_scroll(self.output_text)
        self.output_text.update()

    def write_output_info(self, text: str):
        """
        Writes an information message to the output text window

        :param text: Text to write
        """
        self.write_output("   <info> ", text)

    def write_output_warning(self, text: str):
        """
        Writes a warning message to the output text window

        :param text: Text to write
        """
        self.write_output("<warning> ", text)

    def write_output_error(self, text: str):
        """
        Writes an error message to the output text window

        :param text: Text to write
        """
        self.write_output("  <error> ", text)

    def write_output_model(self, text: str):
        """
        Writes the model information message to the output text window

        :param text: Text to write
        """
        self.write_output("  <model> ", text)

    @staticmethod
    def auto_scroll(scroll_area):
        """
        Scrolls the output text window to the bottom
        """
        scroll_area.see("end")




class Callbacks(keras.callbacks.Callback):
    def __init__(self):
        print("Callbacks created")


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


# Main Tkinter loop
if __name__ == '__main__':
    root = tk.Tk()
    root.title("CNN Training GUI")
    callback = Callbacks()
    app = Application(master=root, callbacks=callback)
