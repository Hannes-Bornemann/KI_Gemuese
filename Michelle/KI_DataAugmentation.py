# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 15:40:59 2023

@author: michl
"""

import numpy as np
import cv2
import os
import glob

folder = "/home/michelle/Documents/Master_Sem1/Künstliche_Intelligenz/Projekt"
folder_carrots = glob.glob(os.path.join(folder, "**carrots/**")) 
folder_potatos = glob.glob(os.path.join(folder, "**potatos/**")) 
folder_onions = glob.glob(os.path.join(folder, "**onions/**")) 

folder_carrots_path = "/home/michelle/Documents/Master_Sem1/Künstliche_Intelligenz/Projekt/carrots"
folder_potatos_path = "/home/michelle/Documents/Master_Sem1/Künstliche_Intelligenz/Projekt/potatos"
folder_onions_path = "/home/michelle/Documents/Master_Sem1/Künstliche_Intelligenz/Projekt/onions"

folder_carrots_flip_vert = os.path.join(os.path.dirname(folder_carrots_path), "carrots_flip_vertical")
folder_carrots_flip_hzt = os.path.join(os.path.dirname(folder_carrots_path), "carrots_flip_horizontal")
folder_potatos_flip_vert = os.path.join(os.path.dirname(folder_potatos_path), "potatos_flip_vertical")
folder_potatos_flip_hzt = os.path.join(os.path.dirname(folder_potatos_path), "potatos_flip_horizontal")
folder_onions_flip_vert = os.path.join(os.path.dirname(folder_onions_path), "onions_flip_vertical")
folder_onions_flip_hzt = os.path.join(os.path.dirname(folder_onions_path), "onions_flip_horizontal")

if not os.path.exists(folder_carrots_flip_vert):
    os.makedirs(folder_carrots_flip_vert)
if not os.path.exists(folder_potatos_flip_vert):
    os.makedirs(folder_potatos_flip_vert)
if not os.path.exists(folder_onions_flip_vert):
    os.makedirs(folder_onions_flip_vert)
if not os.path.exists(folder_carrots_flip_hzt):
    os.makedirs(folder_carrots_flip_hzt)
if not os.path.exists(folder_potatos_flip_hzt):
    os.makedirs(folder_potatos_flip_hzt)
if not os.path.exists(folder_onions_flip_hzt):
    os.makedirs(folder_onions_flip_hzt)


for counter_img in range(0,len(folder_carrots)):
    img_carrot = cv2.imread(folder_carrots[counter_img])
    img_carrot_flip_vert = cv2.flip(img_carrot, 1)
    img_carrot_flip_hzt = cv2.flip(img_carrot, 0)
    
    filename_carrot_flip_vert = os.path.join(folder_carrots_flip_vert, f"carrot_flip_vert_{counter_img}.png") #create a file to save the fliped image
    filename_carrot_flip_hzt = os.path.join(folder_carrots_flip_hzt, f"carrot_flip_hzt_{counter_img}.png") #create a file to save the fliped image
    
    cv2.imwrite(filename_carrot_flip_vert, img_carrot_flip_vert) #save the flipped image
    cv2.imwrite(filename_carrot_flip_hzt, img_carrot_flip_hzt) #save the flipped image

for counter_img in range(0,len(folder_potatos)):
    img_potato = cv2.imread(folder_potatos[counter_img])
    img_potato_flip_vert = cv2.flip(img_potato, 1)
    img_potato_flip_hzt = cv2.flip(img_potato, 0)
    
    filename_potato_flip_vert = os.path.join(folder_potatos_flip_vert, f"potato_flip_vert_{counter_img}.png") #create a file to save the fliped image
    filename_potato_flip_hzt = os.path.join(folder_potatos_flip_hzt, f"potato_flip_hzt_{counter_img}.png") #create a file to save the fliped image
    
    cv2.imwrite(filename_potato_flip_vert, img_potato_flip_vert) #save the flipped image
    cv2.imwrite(filename_potato_flip_hzt, img_potato_flip_hzt) #save the flipped image

for counter_img in range(0,len(folder_onions)):
    img_onion = cv2.imread(folder_onions[counter_img])
    img_onion_flip_vert = cv2.flip(img_onion, 1)
    img_onion_flip_hzt = cv2.flip(img_onion, 0)
    
    filename_onion_flip_vert = os.path.join(folder_onions_flip_vert, f"onion_flip_vert_{counter_img}.png") #create a file to save the fliped image
    filename_onion_flip_hzt = os.path.join(folder_onions_flip_hzt, f"onion_flip_hzt_{counter_img}.png") #create a file to save the fliped image
    
    cv2.imwrite(filename_onion_flip_vert, img_onion_flip_vert) #save the flipped image
    cv2.imwrite(filename_onion_flip_hzt, img_onion_flip_hzt) #save the flipped image


cv2.waitKey(0)


def resizing(image,w,h):
    cv2.resize(image(w,h))
    return image
