import PIL
from PIL import Image
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


def buildSampleFromPath(path1, path2, size=0):
    """
    Build the sample list, a list of dictionnaires, representing the images
    used to train and test the model
    :param path1: path for the goods images, (score 1)
    :param path2: path for the bad images, (score -1)
    :param size: Optionnal if you want to restrict the image pool
    :return: list"""
    S = []

    max_size = getMaxSize(path1, path2)

    for image_path in os.listdir(path1)[:size if size > 0 else -1]:
        S.append(computeDict(image_path, path1, 1, max_size))

    for image_path in os.listdir(path2)[:size if size > 0 else -1]:
        S.append(computeDict(image_path, path2, -1, max_size))

    return S


def computeDict(image_path, path, y_true_value, max_size: tuple):
    """
    Middle function to construct each dict for each image. Resizing, and fetching the histogram,
    by calling ohter functions
    :param image_path: relative path of the image in the folder
    :param path: path of the folder containing the image
    :param y_true_value: is the image a good one (1) or a wrong one (-1)
    :param max_size: the size to resize the image
    :return: a dict representing the image
    """
    full_path = os.path.join(path, image_path)
    image = Image.open(full_path)
    image = image.convert("RGB")

    resized = resizeImage(image, *max_size)

    return {"name_path": full_path,
            "resized_image": resized,
            "X_histo": computeHisto(resized),
            "y_true_class": y_true_value,
            "y_predicted_class": None}


def getMaxSize(path1, path2):
    """
    fetch the size of the image with the most pixels in both images folder
    :param path1: first folder path
    :param path2: second folder path containing images
    :return: tuple with max width, max height
    """
    max_size_x, max_size_y = 0, 0
    max_size_x, max_size_y = computeMaxSizeInFolder(max_size_x, max_size_y, path1)
    max_size_x, max_size_y = computeMaxSizeInFolder(max_size_x, max_size_y, path2)
    return max_size_x, max_size_y


def computeMaxSizeInFolder(max_size_x, max_size_y, path1):
    """
    actually do the real calculation to get the max pixels
    :param max_size_x: current max width
    :param max_size_y: current max height
    :param path1: current folder path
    :return: new max width, new max height in a tuple
    """
    for image_path in os.listdir(path1):
        full_path = os.path.join(path1, image_path)
        image = Image.open(full_path)
        if max_size_x * max_size_y < image.size[0] * image.size[1]:
            max_size_x = image.size[0]
            max_size_y = image.size[1]
    return max_size_x, max_size_y


def resizeImage(i, h, l):
    """
    Resizing the image following the LANCZOS algorithm, with the given width and height
    :param i: the image to resize
    :param h: the new height
    :param l: the new length
    :return: the resized image (PIL.Image.Image)
    """
    return i.resize((h, l), Image.LANCZOS)


def computeHisto(image: PIL.Image.Image):
    """
    Return the color histogram of the image, using Pillow function
    :param image: image used
    :return: the color histogram in a list
    """
    return image.histogram()


def fitFromHisto(S, algo):
    """
    Fit the given algorithm (classifier) With the sample S, We cut in train/test lists.
    We use the syntax of models in skLearn for this method.
    :param S: the sample on which we train
    :param algo: the algo to fit the data on
    :return: the fitted algorithm given in parameters and test values
    """
    df = pd.DataFrame(S)

    y = np.array(df["y_true_class"])

    S_train, S_test, y_train, y_test = train_test_split(S, y, test_size=0.2, random_state=42)

    X_train = np.array([np.array(l["X_histo"]) for l in S_train])

    algo.fit(X_train, y_train)

    return algo, S_test, y_test


def predictFromHisto(S, model, list_dict=True):
    """
    Use the given model to predict the values on the images. Update the sample S to display the
    predicted values.
    :param S: the sample to test
    :param model: the model fitted
    :param list_dict: is the sample in list(dict)
    :return: None
    """
    tab = model.predict(np.array([x["X_histo"] for x in S]))
    if list_dict:
        for i in range(len(S)):
            S[i]["y_predicted_class"] = tab[i]
    else:
        return tab


def empiricalError(S):
    """
    Compute the empirical error of the model on the given sample.
    :param S: the sample to test
    :return: the empirical error of the model on the given sample.
    """
    error_count = 0
    for image in S:
        if image["y_true_class"] != image["y_predicted_class"]:
            error_count += 1
    return error_count / len(S)

def realError(S_test):
    """
    Compute the real error of the model on the test sample.
    :param S_test: the test sample to test
    :return: the real error of the model on the test sample.
    """
    error_count = 0
    for image in S_test:
        if image["y_predicted_class"] != image["y_true_class"]:
            error_count += 1
    return error_count/len(S_test)


