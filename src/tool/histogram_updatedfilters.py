# importing the module
import matplotlib.pyplot as plt
from collections import Counter

file_path = "/home/melassal/Workspace/Results/old2/Weights/KTH-j14-5-L2-2D1D_filter_updates/KTH-j14-5-L2-2D1D_5"

sum = 0
itam_number = 0

with open(file_path + ".txt", "r") as s1:
    data = s1.read().splitlines()

    # total number of updates
    for i in range(len(data)):
        sum += int(data[i]) 
    print("The total number of updates is: " + str(sum))

    counted = Counter(data)
    counted = dict(sorted(counted.items()))
    print(counted)

    plt.bar(counted.keys(), counted.values(), 0.5, color='g')

    plt.show()