from ultralytics import YOLO
import cv2
import numpy as np

# Dicionário de mapeamento para tornar os nomes mais apresentáveis
class_map = {
    'santa_ines': 'Santa Inês',
    'poll_dorset': 'Poll Dorset',
    'dorper': 'Dorper'
}

def classificar(image_bytes):
    # Carrega o modelo pré-treinado
    model = YOLO('static/model/best.pt')

    # Converte os bytes da imagem para um array do OpenCV
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Usa o modelo para prever a raça
    results = model.predict(source=image, save=False, conf=0.001)

    # Verifica se houve predições
    if len(results[0].boxes) > 0:
        predicted_class_id = int(results[0].boxes.cls[0].item())
        predicted_class = model.names[predicted_class_id]
        
        # Busca o nome apresentável no dicionário
        nome_formatado = class_map.get(predicted_class, predicted_class)  # Retorna o próprio nome caso não esteja no dicionário
        confiança = results[0].boxes.conf[0].item()

        # Retorna a raça apresentável e a confiança
        return {
            'raça': nome_formatado,
            'confiança': confiança
        }
    else:
        return {
            'raça': "Nenhuma raça detectada",
            'confiança': 0
        }
