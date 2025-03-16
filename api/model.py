import tensorflow as tf
import joblib
import numpy as np
from numpy import ndarray
from colorama import Fore

class Model:
    model = None
    scaler = None
    feature_weights = None

    def __init__(self, model_file: str, scaler_file: str, weights_file: str):
        try: 
            self.model = tf.keras.models.load_model(f'./res/{model_file}')
        except ValueError:
            print(Fore.RED + '[!]' + Fore.RESET + ' Модель нейронной сети не была загружена.')
        try:  
            self.scaler = joblib.load(f'./res/{scaler_file}')
        except FileNotFoundError:
            print(Fore.RED + '[!]' + Fore.RESET + ' Файл нормализатора не был загружен.')
        try:  
            self.feature_weights = np.load(f'./res/{weights_file}')  
        except FileNotFoundError:
            print(Fore.RED + '[!]' + Fore.RESET + ' Файл весов признаков не был загружен.')

    def get_model(self):
        """
        Геттер модели сверточной нейронной сети.
        
        :return: Модель CNN
        """
        return self.model

    def get_scaler(self):
        """
        Геттер нормализатора.
        
        :return: Нормализатор StandartScaler
        """
        return self.scaler
    
    def get_weights(self) -> ndarray:
        """
        Геттер массива весов признаков.
        
        :return: Массив numpy.ndarray
        """
        return self.feature_weights
    
    def get_FV(self) -> list:
        """
        Геттер весов в виде матрицы.
        
        :return: Нормализатор StandartScaler
        """
        o = self.feature_weights

        FV = [[o[0], o[10]], 
        [o[1], o[2], o[30], o[31], o[32], o[33], o[34], o[39], o[40],o[41], o[42]], 
        [o[14], o[15], o[16]], 
        [o[17], o[18], o[19], o[20], o[21], o[22], o[23], o[24], o[25], o[26], o[27], o[28], o[29], o[35], o[36], o[37]], 
        [o[3], o[4], o[5], o[6], o[7], o[8], o[9], o[38]], 
        [o[11], o[12], o[13]]]
        return FV
