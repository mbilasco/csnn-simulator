# importing the module
from cgitb import grey
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

file_name = "KTH-2-M23-7-L2-2D1D"
file_path = "/home/melassal/Workspace/Results/Features-2d1dvs3d/Weight/KTH-2-M23-5-L2-2D1D/5/"

with open(file_path + "/" + file_name + ".json", "r") as s1:
    data = s1.read()

if data[0] != "[":
    new_data = "[" + data[:-1] + "]"

    with open(file_path + "/" + file_name + ".json", "w") as s1:
        s1.write(new_data)
