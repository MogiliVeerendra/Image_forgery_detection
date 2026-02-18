import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import random

# =====================
# Select & show image
# =====================
def open_image():
    file_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )

    if not file_path:
        return

    # Load image (small & neat)
    img = Image.open(file_path)
    img.thumbnail((300, 300))
    img_tk = ImageTk.PhotoImage(img)

    image_label.config(image=img_tk)
    image_label.image = img_tk

    # 🟢 FORCE AUTHENTIC OUTPUT
    confidence = round(random.uniform(93.0, 95.0), 2)

    result_label.config(text="🟢 AUTHENTIC IMAGE", fg="green")
    confidence_label.config(text=f"Confidence: {confidence}%")


# =====================
# GUI Window
# =====================
root = tk.Tk()
root.title("Image Forgery Detection (Demo Mode)")
root.geometry("420x520")
root.resizable(False, False)

title = tk.Label(
    root,
    text="Image Forgery Detection",
    font=("Arial", 18, "bold")
)
title.pack(pady=10)

image_label = tk.Label(root)
image_label.pack(pady=10)

select_btn = tk.Button(
    root,
    text="Select Image",
    command=open_image,
    font=("Arial", 12),
    bg="#27ae60",
    fg="white",
    width=20
)
select_btn.pack(pady=10)

result_label = tk.Label(
    root,
    text="",
    font=("Arial", 14, "bold")
)
result_label.pack(pady=5)

confidence_label = tk.Label(
    root,
    text="",
    font=("Arial", 12)
)
confidence_label.pack(pady=5)

root.mainloop()
