import numpy as np
import cv2
import imutils
from matplotlib import pyplot as plt
import pytesseract

# Configurando caminho do Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
config_tesseract = r'--tessdata-dir "C:\\Program Files\\Tesseract-OCR\\tessdata"'

def mostrar(img, title="Imagem"):
    """Exibe a imagem utilizando Matplotlib."""
    plt.figure(figsize=(10, 5))
    plt.axis("off")
    plt.title(title)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

def encontrar_contornos(img):
    """Encontra e retorna os 6 maiores contornos."""
    contornos = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contornos = imutils.grab_contours(contornos)
    return sorted(contornos, key=cv2.contourArea, reverse=True)[:6]

def ordenar_pontos(pontos):
    """Ordena os pontos para a transformação de perspectiva."""
    pontos = pontos.reshape((4, 2))
    pontos_ordenados = np.zeros((4, 1, 2), dtype=np.int32)

    add = pontos.sum(1)
    pontos_ordenados[0] = pontos[np.argmin(add)]
    pontos_ordenados[2] = pontos[np.argmax(add)]

    dif = np.diff(pontos, axis=1)
    pontos_ordenados[1] = pontos[np.argmin(dif)]
    pontos_ordenados[3] = pontos[np.argmax(dif)]

    return pontos_ordenados

def processamento_img(img):
    """Melhora a qualidade da imagem para OCR."""
    img_process = cv2.resize(img, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)
    img_process = cv2.cvtColor(img_process, cv2.COLOR_BGR2GRAY)
    img_process = cv2.adaptiveThreshold(img_process, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 9)
    return img_process

def extrair_texto(img):
    """Extrai texto usando Tesseract OCR."""
    return pytesseract.image_to_string(img, lang="eng", config=config_tesseract)
