# importing the module
from cgitb import grey
import json
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageDraw
from pathlib import Path

file_names = ["KTH-2-M23-7-L2-2D1D_1"]

file_path = "/home/melassal/Workspace/Results/Features-2d1dvs3d/Weight/KTH-2-M23-7-L2-2D1D_1/3/"
for file_name in file_names:
    with open(file_path + file_name + ".json", "r") as s1:
        data = s1.read()

        if data[0] != "[":
            new_data = "[" + data[:-1] + "]"

            with open(file_path + "/" + file_name + ".json", "w") as s2:
                s2.write(new_data)
                #data = s1.read()

    file_path = file_path + file_name + "/"

    kernels = json.loads(data)

    kernel = kernels[-1]
    na = np.array(kernel["data"])
    draw_kernel = na.reshape(
        (kernel["dim_3"], kernel["dim_0"], kernel["dim_1"], kernel["dim_2"], kernel["dim_4"]))

    for filter_number in range(kernel["dim_3"]):  # the 16
        for filter_depth in range(kernel["dim_4"]):  # the td
            image_array_0 = draw_kernel[filter_number, :, :, 0, filter_depth]

            image_array_1 = draw_kernel[filter_number, :, :, 1, filter_depth]

            image_array = image_array_0 + image_array_1
            image_array = image_array / np.sum(image_array)
            # Save
            final_name = f'kernel_l{kernel["label"]}_f{filter_number}_td{filter_depth}.png'
            if not os.path.exists(file_path):
                os.makedirs(file_path)

            # cv2.imwrite(file_path + final_name, image_array)
            plt.imshow(image_array, interpolation='none', cmap='Blues')
            plt.savefig(file_path + final_name)

    temporal_depth = kernel["dim_4"]

    if temporal_depth > 1:
        if not os.path.exists(file_path + "/GIFs/"):
            os.makedirs(file_path + "/GIFs/")

        imageNames = [f for f in os.listdir(
            file_path) if os.path.splitext(f)[-1] == '.png']

        imageNames.sort()

        image_count = 0
        images = []
        for filename in enumerate(imageNames):
            # print(str(filename[1]))
            image = Image.open(file_path + "/" + str(filename[1]))
            images.append(image)
            image_count += 1
            if(image_count == temporal_depth):
                # Save
                images[0].save(file_path + "/GIFs/" + os.path.splitext(str(filename[1]))[0] + '.gif', format='GIF', append_images=images[1:],
                            save_all=True, optimize=False, duration=500, loop=0)
                image_count = 0
                images = []
