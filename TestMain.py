import os
from PIL import Image
from sklearn.naive_bayes import GaussianNB

import TP

path1_t = "/home/paul_pvc/PycharmProjects/pythonProject/TP_IA_L3/Init/Mer"
path2_t = "/home/paul_pvc/PycharmProjects/pythonProject/TP_IA_L3/Init/Ailleurs"

def test_Histogram():
    """On teste sur des images unies pour vérifier que les pixels sont bien notés"""
    img_red = Image.new('RGB', (200, 200), (255, 0, 0))
    img_green = Image.new('RGB', (200, 200), (0, 255, 0))
    img_blue = Image.new('RGB', (200, 200), (0, 0, 255))
    assert TP.computeHisto(img_red).count(40000) == 3
    assert TP.computeHisto(img_blue).count(40000) == 3
    assert TP.computeHisto(img_green).count(40000) == 3

def test_resizeImage(i, h, l) :
    resized = TP.resizeImage(i, h, l)
    assert isinstance(resized, Image.Image)

    assert resized.size == (h, l)

def test_sample():
    path1 = "/home/paul_pvc/PycharmProjects/pythonProject/TP_IA_L3/Init/Mer"
    path2 = "/home/paul_pvc/PycharmProjects/pythonProject/TP_IA_L3/Init/Ailleurs"
    TP.buildSampleFromPath(path1, path2)

def real_test():
    S = TP.buildSampleFromPath(path1_t, path2_t)
    print("built")
    classifieur, S_test, y_test = TP.fitFromHisto(S, GaussianNB())
    TP.predictFromHisto(S, classifieur)
    print(TP.empiricalError(S))
    print(TP.realError(S_test))

if __name__ == "__main__":
    image = Image.open("/home/paul_pvc/PycharmProjects/pythonProject/TP_IA_L3/Init/Mer/838s.jpg")
    test_Histogram()
    test_resizeImage(image, 1000, 1000)
    real_test()



