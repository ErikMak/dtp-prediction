import numpy as np
from model import Model
from numpy import ndarray

class GIA:
    def get_image(self, FV, FM):
        """
        Алгоритм перевода матрицы признаков на основе матрицы весов этих признаков в серое изображение.
        
        :param FV: Матрица весов
        :param FM: Матрица признаков
        :return: Матрица, описывающая серое изображение
        """
        parent_sizes = [len(row) for row in FM]
        max_dim = max(parent_sizes)
        
        parent_weights = [sum(row) for row in FV]
        
        sorted_indices = sorted(range(len(FV)), key=lambda i: -parent_weights[i])
        sorted_FV = [FV[i] for i in sorted_indices]
        sorted_FM = [FM[i] for i in sorted_indices]
        
        for i in range(len(sorted_FV)):
            child_indices = sorted(range(len(sorted_FV[i])), key=lambda j: -sorted_FV[i][j])
            sorted_FM[i] = [sorted_FM[i][j] for j in child_indices]
        
        result = np.zeros((max_dim, max_dim))
        
        row_positions = list(range(max_dim))
        row_positions.sort(key=lambda x: abs(x - max_dim // 2))
        
        for i, (row_idx, fm_row) in enumerate(zip(row_positions, sorted_FM)):
            col_positions = list(range(max_dim))
            col_positions.sort(key=lambda x: abs(x - max_dim // 2))
            
            for j, (col_idx, value) in enumerate(zip(col_positions[:len(fm_row)], fm_row)):
                result[row_idx, col_idx] = value

        matrix_normalized = (result - result.min()) / (result.max() - result.min()) * 255
        # РЕЖИМ ОТОБРАЖЕНИЯ ИЗОБРАЖЕНИЯ
        # matrix_normalized = matrix_normalized.astype(np.uint8)
        # image = Image.fromarray(matrix_normalized, mode='L')  # 'L' — режим серого изображения
        image = matrix_normalized

        return image
    
    def convert(self, X, model: Model) -> ndarray:
        """
        Алгоритм перевода признаков в серое изображение.
        
        :param X: Массив признаков
        :return: Матрица ndarray? описывающая серое изображение
        """
        images = []
        for i in X:
            FM = [[i[0], i[10]], 
                [i[1], i[2], i[30], i[31], i[32], i[33], i[34], i[39], i[40],i[41], i[42]], 
                [i[14], i[15], i[16]], 
                [i[17], i[18], i[19], i[20], i[21], i[22], i[23], i[24], i[25], i[26], i[27], i[28], i[29], i[35], i[36], i[37]], 
                [i[3], i[4], i[5], i[6], i[7], i[8], i[9], i[38]], 
                [i[11], i[12], i[13]]]
            
            images.append(self.get_image(model.get_FV(), FM))
        images = np.array(images)
        return images[..., np.newaxis]