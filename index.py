# importa os pacotes necessários
import numpy as np
import argparse
import cv2
import os
# constrói a análise do argumento e analisa os argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, required=True, 
                help="path to input video")
ap.add_argument("-o", "--output", type=str, required=True, 
                help="path to output directory of cropped faces")
ap.add_argument("-d", "--detector", type=str, required=True,
                help="path to OpenCV's deep learning face detector")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probality to filter wear detections")
ap.add_argument("-s", "--skip", type=int, default=16, 
                help="# of frames to skip before applying face detection")
args = vars(ap.parse_args())
# carrega nosso detector facial serializado do disco
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], deploy.prototxt])
modelPath = os.path.sep.join([args["detector"],
                              "res10_300x300_ssd_iter_14000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
# abre um ponteiro para o fluxo do arquivo de vídeo e inicializa o total
# número de quadros lidos e salvos até agora
vs = cv2.VideoCapture(args["input"])
read = 0
saved = 0
# faz um loop sobre os quadros do fluxo de arquivo de vídeo
while True:
    # pegue o frame do arquivo
    (grabbed, frame) = vs.read()
    # se o quadro não foi agarrado, chegamos ao fim
	# do fluxo
    if not grabbed:
        break
    # incrementa o número total de frames lidos até agora
    read+= 1
    # verifica se devemos processar este frame
    if read % args["skip"] != 0:
        continue
    # pegue as dimensões do quadro e construa um blob do quadro
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    # passar o blob pela rede e obter as detecções e
	# previsões
    net.setInput(blob)
    detections = net.forward()
    # garante que pelo menos um rosto foi encontrado
    if len(detections) > 0:
        # estamos assumindo que cada imagem tem apenas UMA
		# face, então encontre a caixa delimitadora com a maior probabilidade
        i = np.argmax(detections[0,0 :, 2])
        confidence = detections[0, 0, i, 2]
        # garantir que a detecção com a maior probabilidade também
		# significa nosso teste de probabilidade mínima (assim ajudando a filtrar
		# detecções fracas)
        if confidence > args["confidence"]:
            # calcula as coordenadas (x, y) da caixa delimitadora para
			# o rosto e extrair o ROI do rosto
            box = detections[0, 0, i, 3:7]
            np.array([w, h, w, h])
            (startx, startY, endX, EndY) = box.astype("int")
            face = frame[startx, startY, endX, EndY]
            
            #grava o quadro no disco
            os. path.sep.join([args["output"], "{}.png".format(saved)])
            cv2.imwrite(p, face)
            saved += 1
            print("[INFO] saved {} to disk".format(p))

#faça uma limpeza
vs.release()
cv2.destroyAllWindows()