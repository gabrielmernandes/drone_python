import cv2
import numpy as np

# Função para contar objetos detectados
def contar_objetos(detections, class_id):
    count = 0
    for detection in detections:
        if detection[1] == class_id:
            count += 1
    return count

# Carregar modelo YOLOv4 e configuração
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Definir classes de interesse (no caso, "cow" que representa gado)
class_id = classes.index("cow")

# Carregar imagem capturada pelo drone
image = cv2.imread("drone_image.jpg")

# Obter dimensões da imagem e prepará-la para detecção
height, width = image.shape[:2]
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

# Definir nomes das camadas de saída
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Passar a imagem através da rede neural
net.setInput(blob)
outputs = net.forward(output_layers)

# Processar as detecções
detections = []
for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5 and class_id == 17:  # 17 representa "cow" na lista de classes COCO
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            detections.append((center_x, center_y))

# Contar e imprimir o número de gados detectados
num_gados = contar_objetos(detections, class_id)
print("Número de gados detectados:", num_gados)

# Mostrar imagem com as detecções
for detection in detections:
    cv2.circle(image, detection, 5, (0, 255, 0), -1)
cv2.imshow("Detecção de gados", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
