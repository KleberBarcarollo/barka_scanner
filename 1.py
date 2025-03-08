import numpy as np
import cv2
import imutils
import pytesseract
from matplotlib import pyplot as plt

# Configuração do Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
tessdata_dir_config = r'--tessdata-dir "C:\\Program Files\\Tesseract-OCR\\tessdata"'

# Função para exibir imagem com Matplotlib
def mostrar(img):
    plt.figure(figsize=(10, 5))
    plt.axis("off")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

# Carregar imagem
img = cv2.imread(r"C:\PYTHON\SCANER\imagens\1.png")
original = img.copy()
mostrar(img)

# Conversão para tons de cinza
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
mostrar(gray)

# Aplicação de desfoque
blur = cv2.GaussianBlur(gray, (5, 5), 0)
mostrar(blur)

# Detecção de bordas
edged = cv2.Canny(blur, 60, 160)
mostrar(edged)

# Função para encontrar contornos
def encontrar_contornos(img):
    conts = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    conts = imutils.grab_contours(conts)
    conts = sorted(conts, key=cv2.contourArea, reverse=True)[:6]
    return conts

conts = encontrar_contornos(edged.copy())

# Localizando maior contorno
maior = None
for c in conts:
    perimetro = cv2.arcLength(c, True)
    aproximacao = cv2.approxPolyDP(c, 0.02 * perimetro, True)
    if len(aproximacao) == 4:
        maior = aproximacao
        break

cv2.drawContours(img, [maior], -1, (120, 255, 0), 2)
mostrar(img)

# Ordenação dos pontos
def ordenar_pontos(pontos):
    pontos = pontos.reshape((4, 2))
    pontos_novos = np.zeros((4, 1, 2), dtype=np.int32)

    add = pontos.sum(1)
    pontos_novos[0] = pontos[np.argmin(add)]
    pontos_novos[2] = pontos[np.argmax(add)]

    dif = np.diff(pontos, axis=1)
    pontos_novos[1] = pontos[np.argmin(dif)]
    pontos_novos[3] = pontos[np.argmax(dif)]

    return pontos_novos

pontos_maior = ordenar_pontos(maior)

# Matriz de transformação
(H, W) = img.shape[:2]
pts1 = np.float32(pontos_maior)
pts2 = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
matriz = cv2.getPerspectiveTransform(pts1, pts2)

# Aplicação da transformação
transform = cv2.warpPerspective(original, matriz, (W, H))
mostrar(transform)

# OCR com Tesseract
texto = pytesseract.image_to_string(transform, lang="por", config=tessdata_dir_config)
print(texto)

# Melhoria da qualidade da imagem
maior = cv2.resize(transform, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
texto = pytesseract.image_to_string(maior, lang="por", config=tessdata_dir_config)
print(texto)

# Aumento de contraste e brilho
brilho = 50
contraste = 80
ajustes = np.int16(transform)
ajustes = ajustes * (contraste / 127 + 1) - contraste + brilho
ajustes = np.clip(ajustes, 0, 255)
ajustes = np.uint8(ajustes)
mostrar(ajustes)

# Limiarização Adaptativa
img_process = cv2.cvtColor(transform, cv2.COLOR_BGR2GRAY)
img_process = cv2.adaptiveThreshold(img_process, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 9)
mostrar(img_process)

# Remoção de bordas
margem = 18
img_final = img_process[margem:H - margem, margem:W - margem]
mostrar(img_final)

# Comparação de imagens
fig, im = plt.subplots(2, 2, figsize=(15, 12))
for x in range(2):
    for y in range(2):
        im[x][y].axis('off')

im[0][0].imshow(original)
im[0][1].imshow(img)
im[1][0].imshow(transform, cmap='gray')
im[1][1].imshow(img_final, cmap='gray')
plt.show()
