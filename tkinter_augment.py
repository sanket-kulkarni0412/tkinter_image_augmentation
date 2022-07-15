from tkinter import *
import os
import tkinter
from turtle import bgcolor, color, width
import cv2
import xml
import wand
import glob
import shutil
import numpy as np
from pathlib import Path
from PIL import Image as im
from wand.image import Image
from xml.etree import ElementTree as ET
from bs4 import BeautifulSoup
import tkinter as Tk
from tkinter import filedialog
from augment import *
from tkinter import ttk
from tkinter import messagebox

win = tkinter.Tk()
win.title('Image Augmentation')

win.geometry('825x800')

#-------------------browse_file-------------------------#


def open_file_in():
    global folder_in
    folder_in = filedialog.askdirectory(
        initialdir=r'D:\\Sanket Kulkarni\\API\\')
    input_listbox.insert(END,folder_in)
    print(folder_in)


def open_file_out():
    global folder_out
    folder_out = filedialog.askdirectory(
        initialdir=r'D:\\Sanket Kulkarni\\API\\')
    output_listbox.insert(END,folder_out)
    print(folder_out)


#-------------------browse_file-------------------------#
#-------------------Noise_Slider-------------------------#

current_value_noise = tkinter.DoubleVar()


def get_current_value_noise():
    return '{: .2f}'.format(current_value_noise.get())


def slider_changed_noise(event):
    noise_par_label.configure(text=get_current_value_noise())

# noise_slider_label=ttk.Label(win,text='Noise :')
# noise_slider_label.grid(row=3,column=0,pady=10)


noise_slider = ttk.Scale(win, from_=0, to=5, orient='horizontal',
                         command=slider_changed_noise, variable=current_value_noise)
noise_slider.grid(row=3, column=1)

current_noise_label = ttk.Label(win, text='Current Noise :')
current_noise_label.grid(row=4, column=1)
current_noise_label.config(background='white')

noise_par_label = ttk.Label(win, text=get_current_value_noise())
noise_par_label.grid(row=4, column=2, columnspan=2)
noise_par_label.config(background='white')

noise_cheak = tkinter.StringVar()

noi_label = ttk.Label(win, text='Noise :')
noi_label.grid(row=2, column=0, pady=10, padx=5)
noi_label.config(background='white')


def noise_cheakbox():
    messagebox.showinfo(title='Noise', message=noise_cheak.get())


noise_cheakbutton = ttk.Checkbutton(
    win, text='Click to Noise', command=noise_cheakbox, variable=noise_cheak, onvalue='True', offvalue=None)
noise_cheakbutton.grid(row=2, column=1)
#-------------------Noise_Slider-------------------------#
#-------------------Blur_Slider-------------------------#
blur_cheak = tkinter.StringVar()

blu_label = ttk.Label(win, text='Blur :')
blu_label.grid(row=5, column=0, pady=10, padx=5)
blu_label.config(background='white')


def blur_cheakbox():
    messagebox.showinfo(title=' Blur', message=blur_cheak.get())


blur_cheakbutton = ttk.Checkbutton(
    win, text='Click to Blur', command=blur_cheakbox, variable=blur_cheak, onvalue='True', offvalue=None)
blur_cheakbutton.grid(row=5, column=1)

current_value_blur = tkinter.DoubleVar()


def get_current_value_blur():
    return '{: .2f}'.format(current_value_blur.get())


def slider_changed_blur(event):
    blur_par_label.configure(text=get_current_value_blur())

# blur_slider_label=ttk.Label(win,text='Blur :')
# blur_slider_label.grid(row=6,column=0,pady=10)


blur_slider = ttk.Scale(win, from_=0, to=5, orient='horizontal',
                        command=slider_changed_blur, variable=current_value_blur)
blur_slider.grid(row=6, column=1)

current_blur_label = ttk.Label(win, text='Current Blur :')
current_blur_label.grid(row=7, column=1)
current_blur_label.config(background='white')

blur_par_label = ttk.Label(win, text=get_current_value_blur())
blur_par_label.grid(row=7, column=2, columnspan=2)
blur_par_label.config(background='white')
#-------------------Blur_Slider-------------------------#
#-------------------Exposure_Slider-------------------------#
exposure_cheak = tkinter.StringVar()

exp_label = ttk.Label(win, text='Exposure :')
exp_label.grid(row=8, column=0, pady=10, padx=5)
exp_label.config(background='white')


def exposure_cheakbox():
    messagebox.showinfo(title='Exposure', message=exposure_cheak.get())


exposure_cheakbutton = ttk.Checkbutton(
    win, text='Click to Exposure', command=exposure_cheakbox, variable=exposure_cheak, onvalue='True', offvalue=None)
exposure_cheakbutton.grid(row=8, column=1)

current_value_exposure = tkinter.DoubleVar()


def get_current_value_exposure():
    return '{: .2f}'.format(current_value_exposure.get())


def slider_changed_exposure(event):
    exposure_par_label.configure(text=get_current_value_exposure())

# exposure_slider_label=ttk.Label(win,text='Exposure :')
# exposure_slider_label.grid(row=9,column=0,pady=10)


exposure_slider = ttk.Scale(win, from_=0, to=5, orient='horizontal',
                            command=slider_changed_exposure, variable=current_value_exposure)
exposure_slider.grid(row=9, column=1)

current_exposure_label = ttk.Label(win, text='Current Exposure :')
current_exposure_label.grid(row=10, column=1)
current_exposure_label.config(background='white')

exposure_par_label = ttk.Label(win, text=get_current_value_exposure())
exposure_par_label.grid(row=10, column=2, columnspan=2)
exposure_par_label.config(background='white')

#-------------------Exposure_Slider-------------------------#
#-------------------Hue_Slider-------------------------#
hue_cheak = tkinter.StringVar()

hu_label = ttk.Label(win, text='Hue :')
hu_label.grid(row=11, column=0, pady=10, padx=5)
hu_label.config(background='white')


def hue_cheakbox():
    messagebox.showinfo(title='Hue', message=hue_cheak.get())


hue_cheakbutton = ttk.Checkbutton(
    win, text='Click to Hue', command=hue_cheakbox, variable=hue_cheak, onvalue='True', offvalue=None)
hue_cheakbutton.grid(row=11, column=1)

current_value_hue = tkinter.DoubleVar()


def get_current_value_hue():
    return '{: .2f}'.format(current_value_hue.get())


def slider_changed_hue(event):
    hue_par_label.configure(text=get_current_value_hue())

# hue_slider_label=ttk.Label(win,text='Hue :')
# hue_slider_label.grid(row=12,column=0,pady=10)


hue_slider = ttk.Scale(win, from_=0, to=5, orient='horizontal',
                       command=slider_changed_hue, variable=current_value_hue)
hue_slider.grid(row=12, column=1)

current_hue_label = ttk.Label(win, text='Current Hue :')
current_hue_label.grid(row=13, column=1)
current_hue_label.config(background='white')

hue_par_label = ttk.Label(win, text=get_current_value_hue())
hue_par_label.grid(row=13, column=2, columnspan=2)
hue_par_label.config(background='white')
#-------------------Hue_Slider-------------------------#
#-------------------Saturation_Slider-------------------------#
saturation_cheak = tkinter.StringVar()

sat_label = ttk.Label(win, text='Saturation :')
sat_label.grid(row=14, column=0, pady=10, padx=5)
sat_label.config(background='white')


def saturation_cheakbox():
    messagebox.showinfo(title='Saturation', message=saturation_cheak.get())


saturation_cheakbutton = ttk.Checkbutton(
    win, text='Click to Saturate', command=saturation_cheakbox, variable=saturation_cheak, onvalue='True', offvalue=None)
saturation_cheakbutton.grid(row=14, column=1)

current_value_saturation = tkinter.DoubleVar()


def get_current_value_saturation():
    return '{: .2f}'.format(current_value_saturation.get())


def slider_changed_saturation(event):
    saturation_par_label.configure(text=get_current_value_saturation())

# saturation_slider_label=ttk.Label(win,text='Saturation :')
# saturation_slider_label.grid(row=15,column=0,pady=10)


saturation_slider = ttk.Scale(win, from_=0, to=5, orient='horizontal',
                              command=slider_changed_saturation, variable=current_value_saturation)
saturation_slider.grid(row=15, column=1)

current_saturation_label = ttk.Label(win, text='Current Saturation :')
current_saturation_label.grid(row=16, column=1)
current_saturation_label.config(background='white')

saturation_par_label = ttk.Label(win, text=get_current_value_saturation())
saturation_par_label.grid(row=16, column=2, columnspan=2)
saturation_par_label.config(background='white')

#-------------------Saturation_Slider-------------------------#
#-------------------Brightness_Slider-------------------------#
brightness_cheak = tkinter.StringVar()

bri_label = ttk.Label(win, text='Brightness :')
bri_label.grid(row=17, column=0, pady=10, padx=5)


def brightness_cheakbox():
    messagebox.showinfo(title='Brightness', message=brightness_cheak.get())


brightness_cheakbutton = ttk.Checkbutton(
    win, text='Click to Bright', command=brightness_cheakbox, variable=brightness_cheak, onvalue='True', offvalue=None)
brightness_cheakbutton.grid(row=17, column=1)

current_value_bri_con = tkinter.DoubleVar()


def get_current_value_bri_con():
    return '{: .2f}'.format(current_value_bri_con.get())


def slider_changed_bri_con(event):
    bri_con_par_label.configure(text=get_current_value_bri_con())

# bri_con_slider_label=ttk.Label(win,text='Brightness :')
# bri_con_slider_label.grid(row=18,column=0,pady=10)


bri_con_slider = ttk.Scale(win, from_=0, to=100, orient='horizontal',
                           command=slider_changed_bri_con, variable=current_value_bri_con)
bri_con_slider.grid(row=18, column=1)

current_bri_con_label = ttk.Label(win, text='Current Contrast :')
current_bri_con_label.grid(row=19, column=1)
current_bri_con_label.config(background='white')

bri_con_par_label = ttk.Label(win, text=get_current_value_bri_con())
bri_con_par_label.grid(row=19, column=2, columnspan=2)
bri_con_par_label.config(background='white')
#----------------------Dark-----------------------------#
current_value_bri_dark = tkinter.DoubleVar()


def get_current_value_bri_dark():
    return '{: .2f}'.format(current_value_bri_dark.get())


def slider_changed_bri_dark(event):
    bri_dark_par_label.configure(text=get_current_value_bri_dark())


bri_dark_slider = ttk.Scale(win, from_=0, to=100, orient='horizontal',
                            command=slider_changed_bri_dark, variable=current_value_bri_dark)
bri_dark_slider.grid(row=20, column=1)

current_bri_dark_label = ttk.Label(win, text='Current Darkness :')
current_bri_dark_label.grid(row=21, column=1)
current_bri_dark_label.config(background='white')

bri_dark_par_label = ttk.Label(win, text=get_current_value_bri_dark())
bri_dark_par_label.grid(row=21, column=2, columnspan=2)
bri_dark_par_label.config(background='white')


#-------------------Brightness_Slider-------------------------#
#-------------------Sharpness_Slider-------------------------#
sharpness_cheak = tkinter.StringVar()

shrp_label = ttk.Label(win, text='Sharpness :')
shrp_label.grid(row=2, column=4, pady=10, padx=(0, 0))
shrp_label.config(background='white')


def sharpness_cheakbox():
    messagebox.showinfo(title='Sharpness', message=sharpness_cheak.get())


sharpness_cheakbutton = ttk.Checkbutton(
    win, text='Click to Sharp', command=sharpness_cheakbox, variable=sharpness_cheak, onvalue='True', offvalue=None,)
sharpness_cheakbutton.grid(row=2, column=5)
# sharpness_cheakbutton.config(style='italian')

current_value_Shrp_con = tkinter.DoubleVar()


def get_current_value_Shrp_con():
    return '{: .2f}'.format(current_value_Shrp_con.get())


def slider_changed_Shrp_con(event):
    Shrp_con_par_label.configure(text=get_current_value_Shrp_con())

# Shrp_con_slider_label=ttk.Label(win,text='Sharpness :')
# Shrp_con_slider_label.grid(row=2,column=6,pady=10,padx=5)


Shrp_con_slider = ttk.Scale(win, from_=0, to=100, orient='horizontal',
                            command=slider_changed_Shrp_con, variable=current_value_Shrp_con)
Shrp_con_slider.grid(row=3, column=5)

current_Shrp_con_label = ttk.Label(win, text='Current White Sharp :')
current_Shrp_con_label.grid(row=4, column=5)
current_Shrp_con_label.config(background='white')

Shrp_con_par_label = ttk.Label(win, text=get_current_value_Shrp_con())
Shrp_con_par_label.grid(row=4, column=6, columnspan=2,padx=(50,0))
Shrp_con_par_label.config(background='white')

#----------------------Dark-----------------------------#
current_value_Shrp_dark = tkinter.DoubleVar()


def get_current_value_Shrp_dark():
    return '{: .2f}'.format(current_value_Shrp_dark.get())


def slider_changed_Shrp_dark(event):
    Shrp_dark_par_label.configure(text=get_current_value_Shrp_dark())


Shrp_dark_slider = ttk.Scale(win, from_=0, to=100, orient='horizontal',
                             command=slider_changed_Shrp_dark, variable=current_value_Shrp_dark)
Shrp_dark_slider.grid(row=5, column=5, pady=10, padx=5)
Shrp_dark_slider.configure()

current_Shrp_dark_label = ttk.Label(win, text='Current Dark Sharp :')
current_Shrp_dark_label.grid(row=6, column=5)
current_Shrp_dark_label.config(background='white')

Shrp_dark_par_label = ttk.Label(win, text=get_current_value_Shrp_dark())
Shrp_dark_par_label.grid(row=6, column=6, columnspan=2,padx=(50,0))
Shrp_dark_par_label.config(background='white')

#-------------------Sharpness_Slider-------------------------#
#-------------------Grayscale_Cheakbox-------------------------#
grayscale_cheak = tkinter.StringVar()

gray_label = ttk.Label(win, text='Grayscale :')
gray_label.grid(row=8, column=4, pady=10, padx=(0, 0))
gray_label.config(background='white')


def grayscale_cheakbox():
    messagebox.showinfo(title='Grayscale', message=grayscale_cheak.get())


grayscale_cheakbutton = ttk.Checkbutton(
    win, text='Click to Gray', command=grayscale_cheakbox, variable=grayscale_cheak, onvalue='True', offvalue=None)
grayscale_cheakbutton.grid(row=8, column=5)

#-------------------Grayscale_Cheakbox-------------------------#
#-------------------VerticalFlip_Cheakbox-------------------------#
verticalflip_cheak = tkinter.StringVar()

vert_label = ttk.Label(win, text='Verticalflip :')
vert_label.grid(row=10, column=4, pady=10, padx=(0, 0))
vert_label.config(background='white')


def verticalflip_cheakbox():
    messagebox.showinfo(title='Verticalflip', message=verticalflip_cheak.get())


verticalflip_cheakbutton = ttk.Checkbutton(
    win, text='Click to Verticle-flip', command=verticalflip_cheakbox, variable=verticalflip_cheak, onvalue='True', offvalue=None)
verticalflip_cheakbutton.grid(row=10, column=5)
#-------------------VerticalFlip_Cheakbox-------------------------#
#-------------------Horizontalflip_Cheakbox-------------------------#
horizontal_cheak = tkinter.StringVar()

hori_label = ttk.Label(win, text='Horizontal :')
hori_label.grid(row=12, column=4, pady=10, padx=(0, 0))
hori_label.config(background='white')


def horizontal_cheakbox():
    messagebox.showinfo(title='Horizontal', message=horizontal_cheak.get())


horizontal_cheakbutton = ttk.Checkbutton(win, text='Click to Horizontal-flip',
                                         command=horizontal_cheakbox, variable=horizontal_cheak, onvalue='True', offvalue=None)
horizontal_cheakbutton.grid(row=12, column=5)

#-------------------Horizontalflip_Cheakbox-------------------------#
#-------------------color_combobox-------------------------#
color_cheak = tkinter.StringVar()

col_label = ttk.Label(win, text='Colorize :')
col_label.grid(row=14, column=4, pady=10, padx=(0, 0))
col_label.config(background='white')


def color_cheakbox():
    messagebox.showinfo(title='Color', message=color_cheak.get())


color_cheakbutton = ttk.Checkbutton(
    win, text='Click to Colorize', command=color_cheakbox, variable=color_cheak, onvalue='True', offvalue=None)
color_cheakbutton.grid(row=14, column=5)

# color_label = ttk.Label(text="Colorize :")
# color_label.grid(row=12,column=6,pady=10)

selected_color = tkinter.StringVar()
color_cb = ttk.Combobox(win, textvariable=selected_color)

color_cb['values'] = ['red', 'green', 'blue']
color_cb['state'] = 'readonly'

color_cb.grid(row=15, column=5)
color_cb.configure(background='white')


def color_changed(event):
    """ handle the month changed event """
    messagebox.showinfo(
        title='Result',
        message=f'You selected {selected_color.get()}!')


color_cb.bind('<<ComboboxSelected>>', color_changed)

#-------------------color_combobox-------------------------#
#-------------------rotation_entry-------------------------#
rotation_cheak = tkinter.StringVar()

rot_label = ttk.Label(win, text='Rotation :')
rot_label.grid(row=17, column=4, pady=10, padx=(0, 0))
rot_label.config(background='white')


def rotation_cheakbox():
    messagebox.showinfo(title='Rotation', message=rotation_cheak.get())


rotation_cheakbutton = ttk.Checkbutton(
    win, text='Click to Rotate', command=rotation_cheakbox, variable=rotation_cheak, onvalue='True', offvalue=None)
rotation_cheakbutton.grid(row=17, column=5)

rotate_text = StringVar()
# rotate_label = Label(win, text='Rotate :')
# rotate_label.grid(row=18, column=4, pady=10,padx=5)
rotate_entry = Entry(win, textvariable=rotate_text)
rotate_entry.config(background='white')
rotate_entry.grid(row=18, column=5)
rotate_entry.config(background='white')
#-------------------rotation_entry-------------------------#


def run():
    if (bool(noise_cheak.get) == True) and (int(current_value_noise.get()) != 0):
        augmentation(folder_in + '/', folder_out + '/', Noise=int(current_value_noise.get()), Grayscale=bool(
            grayscale_cheak.get()), Vertical_Flip=bool(verticalflip_cheak.get()), Horizontal_Flip=bool(horizontal_cheak.get()))

    if (bool(blur_cheak.get()) == True) and (int(current_value_blur.get()) != 0):
        augmentation(folder_in + '/', folder_out + '/', Blur=int(current_value_blur.get()), Grayscale=bool(
            grayscale_cheak.get()), Vertical_Flip=bool(verticalflip_cheak.get()), Horizontal_Flip=bool(horizontal_cheak.get()))

    if (bool(exposure_cheak.get()) == True) and (int(current_value_exposure.get()) != 0):
        augmentation(folder_in + '/', folder_out + '/', Exposure=int(current_value_exposure.get()), Grayscale=bool(
            grayscale_cheak.get()), Vertical_Flip=bool(verticalflip_cheak.get()), Horizontal_Flip=bool(horizontal_cheak.get()))

    if (bool(hue_cheak.get()) == True) and (int(current_value_hue.get()) != 0):
        augmentation(folder_in + '/', folder_out + '/', Hue=int(current_value_hue.get()), Grayscale=bool(grayscale_cheak.get()),
                     Vertical_Flip=bool(verticalflip_cheak.get()), Horizontal_Flip=bool(horizontal_cheak.get()))

    if (bool(saturation_cheak.get()) == True) and (int(current_value_saturation.get()) != 0):
        augmentation(folder_in + '/', folder_out + '/', Saturation=int(current_value_saturation.get()), Grayscale=bool(
            grayscale_cheak.get()), Vertical_Flip=bool(verticalflip_cheak.get()), Horizontal_Flip=bool(horizontal_cheak.get()))

    if (bool(brightness_cheak.get()) == True) and (int(current_value_bri_dark.get()) != 0) and (int(current_value_bri_con.get()) != 0):
        augmentation(folder_in + '/', folder_out + '/', Brightness=(int(current_value_bri_con.get()), int(current_value_bri_dark.get())),
                     Grayscale=bool(grayscale_cheak.get()), Vertical_Flip=bool(verticalflip_cheak.get()), Horizontal_Flip=bool(horizontal_cheak.get()))

    if (bool(sharpness_cheak.get()) == True) and (int(current_value_Shrp_con.get()) != 0) and int(current_value_Shrp_dark.get() != 0):
        augmentation(folder_in + '/', folder_out + '/', Sharpen=(int(current_value_Shrp_con.get()), int(current_value_Shrp_dark.get())),
                     Grayscale=bool(grayscale_cheak.get()), Vertical_Flip=bool(verticalflip_cheak.get()), Horizontal_Flip=bool(horizontal_cheak.get()))

    if bool(color_cheak.get()) == True:
        augmentation(folder_in + '/', folder_out + '/', Colorize=str(selected_color.get()), Grayscale=bool(
            grayscale_cheak.get()), Vertical_Flip=bool(verticalflip_cheak.get()), Horizontal_Flip=bool(horizontal_cheak.get()))

    if (bool(rotation_cheak.get()) == True) and int(rotate_text.get() != 0):
        if len(rotate_text.get()) >= 1:
            list_r = []
            list_rotate = rotate_text.get().split(',')
            for i in list_rotate:
                list_r.append(int(i))
        augmentation(folder_in + '/', folder_out + '/', Rotation=list_r, Grayscale=bool(grayscale_cheak.get()),
                     Vertical_Flip=bool(verticalflip_cheak.get()), Horizontal_Flip=bool(horizontal_cheak.get()))
    print(bool(sharpness_cheak.get()))
    # augmentation(folder_in + '/' ,folder_out + '/',Noise=int(current_value_noise.get()),
    # Blur=int(current_value_blur.get()),
    # Exposure=int(current_value_exposure.get()),
    # Hue=int(current_value_hue.get()),
    # Saturation=int(current_value_saturation.get()),
    # Brightness=(int(current_value_bri_con.get()),int(current_value_bri_dark.get())),
    # Sharpen=(int(current_value_Shrp_con.get()),int(current_value_Shrp_dark.get())),
    # Grayscale=bool(grayscale_cheak.get()),Vertical_Flip=bool(verticalflip_cheak.get()),Horizontal_Flip=bool(horizontal_cheak.get()),
    # Colorize=str(selected_color.get()),
    # Rotation=list_r)
    return


# input_text=StringVar()
input_label = Label(win, text='Input Directory :', font=('bold', 10), pady=20)
input_label.grid(row=0, column=0, sticky=W)
input_label.config(background='white')
# input_entry = Entry(win, textvariable=input_text)
# input_entry.grid(row=0, column=1)

# output_text=StringVar()s
output_label = Label(win, text='Output Directory :', font=('bold', 10), pady=20)
output_label.grid(row=1, column=0, sticky=W)
output_label.config(background='white')
# output_entry = Entry(win, textvariable=output_text)
# output_entry.grid(row=1, column=1)

run_btn = Button(win, text='RUN', font='bold', width=15, height=1, command=run)
run_btn.grid(row=23, column=4, pady=10)
run_btn.config(background='red',fg='white')

browse_button_in = Button(win, text='Browse', command=open_file_in)
browse_button_in.grid(row=0, column=4)
browse_button_in.config(background='green',fg='white')

browse_button_out = Button(win, text='Browse', command=open_file_out)
browse_button_out.grid(row=1, column=4)
browse_button_out.config(background='green',fg='white')

input_listbox=Listbox(win,height=1,width=50)
input_listbox.grid(row=0,column=1,columnspan=3)
input_listbox.config(background='white')

output_listbox=Listbox(win,height=1,width=50)
output_listbox.grid(row=1,column=1,columnspan=3)
output_listbox.config(background='white')

win.configure(bg='white')
win.mainloop()
