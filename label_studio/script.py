import os


for file in os.listdir("images"):
    # print(file, os.path.join("rec", "images", "val_"+file))
    os.rename(os.path.join("images", file), os.path.join("rec", "images", "val_"+file))
