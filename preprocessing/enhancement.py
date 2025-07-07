# preprocessing/enhancement.py

import cv2

def illumination_correction(img):
    """
    Aplica correção de iluminação para remover brilho central exagerado
    em imagens fundoscópicas.

    Parâmetros:
        img (np.ndarray): imagem RGB (numpy array)

    Retorno:
        img corrigida (np.ndarray)
    """
    blur = cv2.GaussianBlur(img, (101, 101), 0)
    corrected = cv2.addWeighted(img, 4, blur, -4, 128)
    return corrected
