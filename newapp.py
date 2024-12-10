import tkinter as tk
from tkinter import filedialog, messagebox
import torch
import pickle
import pandas as pd
import numpy as np
import torch.nn as nn
from pyvirtualdisplay import Display

# Define the same model architecture used during training
class SimpleCNN(nn.Module):
    def __init__(self, input_channels, output_size):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=(3, 1))
        self.relu = nn.ReLU()
        self.fc = nn.Linear(16 * (input_channels - 2), output_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

# Load the saved model
model_path = "/Users/sathviksanka/Desktop/DDCS/models/global_model.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Set model to evaluation mode
model.eval()

# Define the main application window
class FederatedLearningApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Federated Learning GUI")
        self.root.geometry("500x300")

        # Buttons for different functionalities
        self.upload_button = tk.Button(root, text="Upload Data", command=self.upload_data)
        self.upload_button.pack(pady=10)

        self.integrate_button = tk.Button(root, text="Integrate Data", command=self.integrate_data)
        self.integrate_button.pack(pady=10)

        self.start_analysis_button = tk.Button(root, text="Start Analysis", command=self.start_analysis)
        self.start_analysis_button.pack(pady=10)

        # To store uploaded data
        self.data = None

    def upload_data(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            try:
                self.data = pd.read_csv(file_path)
                messagebox.showinfo("Success", "Data uploaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to upload data: {e}")

    def integrate_data(self):
        if self.data is not None:
            try:
                # Placeholder for data integration (e.g., cleaning, merging, etc.)
                self.data.fillna(0, inplace=True)
                messagebox.showinfo("Success", "Data integrated successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to integrate data: {e}")
        else:
            messagebox.showwarning("Warning", "Please upload data first.")

    def start_analysis(self):
        if self.data is not None:
            try:
                # Assuming data is preprocessed for model input
                features = self.data.iloc[0].values.astype(np.float32)  # Using the first row as an example
                features = features.reshape(1, 1, len(features), 1)  # Reshape for CNN input
                with torch.no_grad():
                    inputs = torch.from_numpy(features)
                    output = model(inputs)
                    _, predicted = torch.max(output, 1)
                messagebox.showinfo("Prediction", f"Predicted Class: {predicted.item()}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to start analysis: {e}")
        else:
            messagebox.showwarning("Warning", "Please upload and integrate data first.")

# Initialize and run the GUI application
if __name__ == "__main__":
    display = Display(visible=0, size=(800, 600)) # Create a virtual display
    display.start() # Start the virtual display

    root = tk.Tk()
    app = FederatedLearningApp(root)
    root.mainloop()

    display.stop()
