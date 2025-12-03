import numpy as np
import pandas as pd
import os
from geopy.distance import geodesic
from sklearn.cluster import DBSCAN

def within_distance_limit(point1, point2, distance_limit):
    """
    Calcula a distância geodésica entre dois pontos e retorna a distância
    em km se estiver dentro do limite, caso contrário retorna -1.
    """
    distance = geodesic((point1['r_lat'], point1['r_long']), (point2['r_lat'], point2['r_long'])).km
    if distance <= distance_limit:
        return distance
    else:
        return -1

def within_time_limit(date1, date2, time_limit):
    """
    Calcula a diferença de tempo em dias entre duas datas e retorna a diferença
    se estiver dentro do limite, caso contrário retorna -1.
    """
    delta = abs((date1['r_data'] - date2['r_data']).days)
    if delta <= time_limit:
        return delta
    else:
        return -1

def calculate_dist_time(X, time_limit, distance_limit):
    """
    Calcula as matrizes de distância espacial e temporal entre todos os pontos.
    """
    n = len(X)
    matrix_dist = np.zeros((n, n))
    matrix_time = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            distance = within_distance_limit(X.iloc[i], X.iloc[j], distance_limit)
            matrix_dist[i][j] = distance
            matrix_dist[j][i] = matrix_dist[i][j]

            dist_time = within_time_limit(X.iloc[i], X.iloc[j], time_limit)
            matrix_time[i][j] = dist_time
            matrix_time[j][i] = matrix_time[i][j]

    return matrix_dist, matrix_time

def calculate_normalized_matrices(distance_matrix, time_matrix):
    """
    Normaliza as matrizes de distância e tempo.
    """
    n = distance_matrix.shape[0]
    max_dist = np.max(distance_matrix[distance_matrix >= 0])
    max_time = np.max(time_matrix[time_matrix >= 0])

    norm_distance_matrix = np.zeros((n, n))
    norm_time_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            if distance_matrix[i][j] < 0:
                norm_distance_matrix[i][j] = 99999
            else:
                norm_distance_matrix[i][j] = distance_matrix[i][j] / max_dist
            norm_distance_matrix[j][i] = norm_distance_matrix[i][j]

            if time_matrix[i][j] < 0:
                norm_time_matrix[i][j] = 99999
            else:
                norm_time_matrix[i][j] = time_matrix[i][j] / max_time
            norm_time_matrix[j][i] = norm_time_matrix[i][j]

    return norm_distance_matrix, norm_time_matrix

def cluster_records(input_csv, output_csv, matrix_output_folder, time_limit, distance_limit):
    """
    Realiza a clusterização dos registros com base em distâncias espaciais e temporais
    e salva as matrizes calculadas.

    Args:
        input_csv (str): O caminho do arquivo CSV de entrada.
        output_csv (str): O caminho para salvar o arquivo CSV de saída com os clusters.
        matrix_output_folder (str): A pasta para salvar as matrizes de distância, tempo e total.
        time_limit (int): O limite de tempo em dias para a clusterização.
        distance_limit (float): O limite de distância em km para a clusterização.
    """
    try:
        # Lendo o arquivo de entrada
        base = pd.read_csv(input_csv)
        X = base.copy()

        # Converter a coluna de data para o formato datetime
        X['r_data'] = pd.to_datetime(X['r_data']).dt.date

        # Calcular as matrizes de tempo e distância
        print("Calculando matrizes de distância e tempo...")
        distance_matrix, time_matrix = calculate_dist_time(X, time_limit, distance_limit)

        # Salvar as matrizes de tempo e distância
        df_distance_matrix = pd.DataFrame(distance_matrix)
        df_distance_matrix.to_csv(os.path.join(matrix_output_folder, f"distance_matrix_{time_limit}d_{distance_limit}km.csv"), index=False)
        print(f"Matriz de distância salva como '{os.path.join(matrix_output_folder, f'distance_matrix_{time_limit}d_{distance_limit}km.csv')}'.")
        df_time_matrix = pd.DataFrame(time_matrix)
        df_time_matrix.to_csv(os.path.join(matrix_output_folder, f"time_matrix_{time_limit}d_{distance_limit}km.csv"), index=False)
        print(f"Matriz de tempo salva como '{os.path.join(matrix_output_folder, f'time_matrix_{time_limit}d_{distance_limit}km.csv')}'.")

        # Calcular as matrizes normalizadas 
        norm_distance_matrix, norm_time_matrix = calculate_normalized_matrices(distance_matrix, time_matrix)

        # Calcular a matriz total com a distância espacial e temporal
        total_distance = np.add(norm_distance_matrix, norm_time_matrix)        

        # Salvar a matriz de distância total
        df_total_distance = pd.DataFrame(total_distance)
        df_total_distance.to_csv(os.path.join(matrix_output_folder, f"total_distance_{time_limit}d_{distance_limit}km.csv"), index=False)
        print(f"Matriz de distância total salva como '{os.path.join(matrix_output_folder, f'total_distance_{time_limit}d_{distance_limit}km.csv')}'.")

        # Executar o DBSCAN com min_samples=1 para Cluster1
        dbs1 = DBSCAN(eps=500, min_samples=1, metric='precomputed')
        labels1 = dbs1.fit_predict(total_distance)
        X['Cluster1'] = labels1

        # Executar o DBSCAN com min_samples=2 para Cluster2
        dbs2 = DBSCAN(eps=500, min_samples=2, metric='precomputed')
        labels2 = dbs2.fit_predict(total_distance)
        X['Cluster2'] = labels2
        
        # Salvar o dataframe resultante
        X.to_csv(output_csv, index=False)
        print(f"O arquivo com os clusters foi salvo como '{output_csv}'.")

    except FileNotFoundError:
        print(f"Erro: O arquivo '{input_csv}' não foi encontrado.")
    except Exception as e:
        print(f"Ocorreu um erro durante a clusterização: {e}")

if __name__ == '__main__':
    # Defina os caminhos dos arquivos de entrada e saída
    # Define o nome do arquivo de entrada e saída
    ##-- PNH
    path = 'data_plos'
    file = 'registros_macacos_micos_01_05_2014_ate_31_12_2024_filtrado_limpo_geocode_6d'  
    ##-- Aves
    #path = 'data_E_vigilancia'
    #file = 'registros_aves_filtrado

    matrix_output_folder = f'{path}/distance_matrix'

    # Parâmetros de clusterização
    ##-- PNH
    time_limit_days = 30
    distance_limit_km = 1
    ##-- Aves
    #time_limit_days = 17
    #distance_limit_km = 303

    input_file = f'{path}/{file}.csv'
    output_file = f'{path}/clusters_{time_limit_days}d_{distance_limit_km}km.csv'

    # Chama a função principal
    cluster_records(input_file, output_file, matrix_output_folder, time_limit=time_limit_days, distance_limit=distance_limit_km)