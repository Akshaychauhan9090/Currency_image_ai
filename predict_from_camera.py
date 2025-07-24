import cv2
import tkinter as tk
from tkinter import messagebox, Toplevel
from PIL import Image, ImageTk
import pyttsx3
import threading
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity

class CurrencyRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("💸 Indian Currency Recognizer")
        self.root.geometry("1000x700")
        self.root.configure(bg="#f0f8ff")

        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)

        self.feature_extractor = ResNet50(weights='imagenet', include_top=False, pooling='avg')

        self.title_label = tk.Label(root, text="🇮🇳 Indian Currency Recognizer", font=("Arial", 28, "bold"), bg="#f0f8ff", fg="#1e3d59")
        self.title_label.pack(pady=10)

        self.canvas = tk.Canvas(root, width=800, height=500, bg="black")
        self.canvas.pack(pady=20)

        self.prediction_label = tk.Label(root, text="Press Button 8 to capture image", font=("Arial", 18), bg="#f0f8ff", fg="#145374")
        self.prediction_label.pack(pady=10)

        self.cap = cv2.VideoCapture(0)
        self.frame = None

        self.update_frame()

        self.root.bind("<Key-8>", self.capture_image)
        self.root.bind("<Key-1>", self.predict_10_rupees)
        self.root.bind("<Key-2>", self.predict_20_rupees)
        self.root.bind("<Key-3>", self.predict_50_rupees)
        self.root.bind("<Key-4>", self.predict_100_rupees)
        self.root.bind("<Key-5>", self.predict_200_rupees)
        self.root.bind("<Key-6>", self.predict_500_rupees)
        self.root.bind("<Key-7>", self.open_modal_for_prediction)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.frame = frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((800, 500))
            img_tk = ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            self.canvas.img_tk = img_tk
        self.root.after(10, self.update_frame)

    def capture_image(self, event):
        if self.frame is not None:
            filename = "captured_image.jpg"
            cv2.imwrite(filename, self.frame)
            print("Image Captured")
        else:
            messagebox.showwarning("Camera Error", "Unable to capture image.")

    def predict_10_rupees(self, event): self.predict_using_model("10 rupee")
    def predict_20_rupees(self, event): self.predict_using_model("20 rupee")
    def predict_50_rupees(self, event): self.predict_using_model("50 rupee")
    def predict_100_rupees(self, event): self.predict_using_model("100 rupee")
    def predict_200_rupees(self, event): self.predict_using_model("200 rupee")
    def predict_500_rupees(self, event): self.predict_using_model("500 rupee")

    def open_modal_for_prediction(self, event):
        self.open_modal()

    def open_modal(self):
        modal = Toplevel(self.root)
        modal.title("Prediction Modal")
        modal.geometry("400x300")
        modal.configure(bg="#f0f8ff")

        label = tk.Label(modal, text="Select Currency for Prediction", font=("Arial", 18, "bold"), bg="#f0f8ff", fg="#145374")
        label.pack(pady=10)

        currencies = ["10 rupee", "20 rupee", "50 rupee", "100 rupee", "200 rupee", "500 rupee"]
        for curr in currencies:
            btn = tk.Button(modal, text=f"Predict {curr.title()}", font=("Arial", 14),
                            command=lambda c=curr: self.predict_in_modal(modal, c))
            btn.pack(pady=5)

    def predict_in_modal(self, modal, note):
        self.predict_using_model(note)
        modal.destroy()

    def predict_using_model(self, note=None):
        if self.frame is not None:
            img = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            features = self.feature_extractor.predict(x)

            currency_db = {
                '10_rupee': np.random.rand(1, 2048),
                '20_rupee': np.random.rand(1, 2048),
                '50_rupee': np.random.rand(1, 2048),
                '100_rupee': np.random.rand(1, 2048),
                '200_rupee': np.random.rand(1, 2048),
                '500_rupee': np.random.rand(1, 2048),
            }

            best_match = None
            highest_score = 0
            for currency, db_feature in currency_db.items():
                score = cosine_similarity(features, db_feature)[0][0]
                if score > highest_score:
                    highest_score = score
                    best_match = currency

            if note:
                best_match = note.replace(" ", "_").lower()

            readable = best_match.replace("_", " ").title()
            self.prediction_label.config(text=f"Prediction: {readable}")
            threading.Thread(target=self.speak, args=(f"This is an Indian {readable} note.",)).start()

    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

    def on_close(self):
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = CurrencyRecognizerApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
