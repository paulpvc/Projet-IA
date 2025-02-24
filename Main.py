import PIL
from PIL import Image
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split



def computeHisto(image: PIL.Image.Image):
    return image.histogram()

def get_max_size(path1, path2):
    max_size_x, max_size_y = 0, 0
    max_size_x, max_size_y = compute_max_size_in_folder(max_size_x, max_size_y, path1)
    max_size_x, max_size_y = compute_max_size_in_folder(max_size_x, max_size_y, path2)
    return max_size_x, max_size_y


def compute_max_size_in_folder(max_size_x, max_size_y, path1):
    for image_path in os.listdir(path1):
        full_path = os.path.join(path1, image_path)
        image = Image.open(full_path)
        if max_size_x * max_size_y < image.size[0] * image.size[1]:
            max_size_x = image.size[0]
            max_size_y = image.size[1]
    return max_size_x, max_size_y


def resizeImage (i, h, l) :
    return i.resize((h,l), Image.LANCZOS)



def buildSampleFromPath(path1, path2, size=0):
    S = []

    max_size = get_max_size(path1, path2)

    for image_path in os.listdir(path1)[:size if size > 0 else -1]:

        compute_dict(S, image_path, path1, 1, max_size)

    for image_path in os.listdir(path2)[:size if size > 0 else -1]:
        compute_dict(S, image_path, path2, -1, max_size)

    return S



def compute_dict(S, image_path, path, y_true_value, max_size: tuple):
    full_path = os.path.join(path, image_path)
    image = Image.open(full_path)
    image = image.convert("RGB")
    resized = resizeImage(image, *max_size)
    #print("resized")
    S.append({"name_path": full_path,
              "resized_image": resized,
              "X_histo": computeHisto(resized),
              "y_true_class": y_true_value,
              "y_predicted_class": None})



def fitFromHisto(S, algo) :
    df = pd.DataFrame(S)

    #X = np.array(df["X_histo"])
    X = np.array([np.array(l) for l in df["X_histo"]])
    y = np.array(df["y_true_class"])


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    algo.fit(X_train, y_train)

    return algo

def predictFromHisto(S, model):
    #for image in S:
    #    image["y_predicted_class"] = model.predict(np.array(image["X_histo"]).reshape(1,-1))
    tab = model.predict(np.array([x["X_histo"] for x in S]))
    for i in range(len(S)):
        S[i]["y_predicted_class"] = tab[i]

def erreur_empirique(S):
    error_count = 0
    for image in S:
        if image["y_true_class"] != image["y_predicted_class"]:
            error_count += 1
    return error_count/len(S)

