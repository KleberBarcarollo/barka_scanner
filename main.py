import cv2
import numpy as np
from utils import mostrar, encontrar_contornos, ordenar_pontos, processamento_img, extrair_texto

# Caminho da imagem
caminho_imagem = r"C:\PYTHON\SCANER\imagens\07.png"

# Carregar imagem
img = cv2.imread(caminho_imagem)
original = img.copy()
(H, W) = img.shape[:2]
mostrar(img, "Imagem Original")

# Pré-processamento da imagem
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blur, 60, 160)
mostrar(edged, "Detecção de Bordas")

# Encontrar contornos
conts = encontrar_contornos(edged.copy())

# Localizar maior contorno de 4 lados
for c in conts:
    perimetro = cv2.arcLength(c, True)
    aprox = cv2.approxPolyDP(c, 0.02 * perimetro, True)
    if len(aprox) == 4:
        maior = aprox
        break

cv2.drawContours(img, [maior], -1, (120, 255, 0), 2)
mostrar(img, "Maior Contorno")

# Transformação de Perspectiva
pontos_maior = ordenar_pontos(maior)
pts1 = np.float32(pontos_maior)
pts2 = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
matriz = cv2.getPerspectiveTransform(pts1, pts2)
transform = cv2.warpPerspective(original, matriz, (W, H))
mostrar(transform, "Imagem Transformada")

# Processamento para OCR
img_final = processamento_img(transform)
mostrar(img_final, "Imagem Pós-processada")

# Extração de texto
texto_extraido = extrair_texto(img_final)
print("\nTexto Extraído:\n", texto_extraido)