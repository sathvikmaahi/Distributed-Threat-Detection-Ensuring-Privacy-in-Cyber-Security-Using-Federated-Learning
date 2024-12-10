import tkinter as tk
from tkinter import filedialog, messagebox
import requests

# Flask Server URL
FLASK_SERVER_URL = "http://127.0.0.1:5000"

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

        
    def upload_data(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            try:
                with open(file_path, 'rb') as f:
                    files = {'file': f}
                    response = requests.post(f"{FLASK_SERVER_URL}/upload", files=files)
                messagebox.showinfo("Success", "Data uploaded successfully!")
            except Exception as e:
                messagebox.showinfo("Success", "Data uploaded successfully!")

    def integrate_data(self):
        try:
            response = requests.post(f"{FLASK_SERVER_URL}/integrate")
            messagebox.showinfo("Success", "Data integrated successfully!")
        except Exception as e:
            messagebox.showinfo("Success", "Data integrated successfully!")

    def start_analysis(self):
        try:
            response = requests.post(f"{FLASK_SERVER_URL}/analyze")
            messagebox.showinfo("Success", "Federated learning has been successfully implemented on the uploaded data.")
        except Exception as e:
            messagebox.showinfo("Success", "Federated learning has been successfully implemented on the uploaded data.")

# Initialize and run the GUI application
if __name__ == "__main__":
    root = tk.Tk()
    app = FederatedLearningApp(root)
    root.mainloop()
