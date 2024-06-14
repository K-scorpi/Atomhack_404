import tkinter as tk
from tkinter import messagebox

def calculate_square():
    try:
        number = str(entry.get())
        result = number 
        result_label.config(text=f"Ответ по запросу {result}")
    except ValueError:
        messagebox.showerror("Ошибка")

root = tk.Tk()
root.title("GUI")
root.geometry("600x800")

entry = tk.Entry(root, width=100)
entry.pack(pady=10)

button = tk.Button(root, text="Отправить", command=calculate_square)
button.pack()

result_label = tk.Label(root, text="")
result_label.pack()

root.mainloop()