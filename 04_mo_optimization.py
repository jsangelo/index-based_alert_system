import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def process_data(input_csv):
    """
    Lê o arquivo de entrada e prepara os dados para a análise.
    
    Args:
        input_csv (str): Caminho para o arquivo CSV de entrada com as
                         características dos clusters.
    Returns:
        pd.DataFrame: DataFrame original com os dados carregados.
    """
    try:
        df = pd.read_csv(input_csv, index_col=0)
        return df
    except FileNotFoundError:
        print(f"Erro: O arquivo '{input_csv}' não foi encontrado.")
        return None
    except Exception as e:
        print(f"Ocorreu um erro ao processar os dados: {e}")
        return None

def normalize_and_standardize_data(df, columns):
    """
    Padroniza (Z-score) e normaliza (Min-Max) colunas de um DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame de entrada.
        columns (list): Lista de colunas a serem processadas.
    Returns:
        tuple: Uma tupla contendo o DataFrame padronizado e o DataFrame normalizado.
    """
    df_standardized = df.copy()
    df_normalized = df.copy()

    # Padronização Z-score
    scaler_standard = StandardScaler()
    df_standardized[columns] = scaler_standard.fit_transform(df[columns])

    # Normalização Min-Max
    scaler_minmax = MinMaxScaler()
    df_normalized[columns] = scaler_minmax.fit_transform(df[columns])

    return df_standardized, df_normalized

def objective_function(weights, data, alpha):
    """
    Função objetivo para a otimização, buscando um equilíbrio
    entre a maximização do índice de alerta e a minimização da variância.
    
    A otimização busca minimizar o valor retornado.
    
    Args:
        weights (np.array): Pesos para cada característica.
        data (pd.DataFrame): DataFrame com os dados normalizados.
    
    Returns:
        float: Valor da função objetivo (a ser minimizado).
    """
    # Características a serem usadas no índice de alerta
    # A ordem deve ser a mesma dos pesos
    alert_cols = [ "num_reg", "intervalo", "freq_num_reg", "freq_a_quant", "extensao","perc_mortos", "morto"]

    # Calcula o índice de alerta para cada cluster
    alert_index = np.dot(data[alert_cols], weights)                 

    # Objetivo: maximizar o índice de alerta para os casos confirmados
    # e minimizar a variância do índice.
    
    f1 = np.average(alert_index,weights = data['morto']) # Maximizar a média do índice ponderado pelo num. de mortos
    f2 = np.var(alert_index)   # Minimizar a variância do índice
        
    # Soma ponderada dos pesos
    obj = alpha*f1 - (1-alpha)*f2

    return -obj #(maximizar)


def salve_results(path, solution):
    '''
    Salvar soluções em arquivo (padrão numpy e padrão tabela para Latex)
    '''
    ## INCOMPLETO!! CONCLUIR
    
    with open(f'{path}/results/solution_vectors.txt', 'w') as arquivo:
        for alpha, data in results.items():
            print(f"--- Resultados para alpha = {alpha} ---")
            for key, value in data.items():
                print(f"{key}: {value}")
            print("\n")
        

def calculate_objective_values(weights, data):
    """
    Esta função apenas calcula o valor dos objetivos separadamente.
    Não consegui utilizar a função objective_function() pois ela é chamada pelo métodos de otimização.
    Para minimize() a função objective_function() só pode retornar um valor de objetivo.
    Neste caso, preciso da informação dos dois objetivos para gerar a Fronteira de Pareto.
    """

    # Características a serem usadas no índice de alerta
    # A ordem deve ser a mesma dos pesos
    alert_cols = [ "num_reg", "intervalo", "freq_num_reg", "freq_a_quant", "extensao","perc_mortos", "morto"]

    # Calcula o índice de alerta para cada cluster
    alert_index = np.dot(data[alert_cols], weights)                 

    # Objetivo: maximizar o índice de alerta para os casos confirmados
    # e minimizar a variância do índice.
    
    f1 = np.average(alert_index,weights = data['morto']) # Maximizar a média do índice ponderado pelo num. de mortos
    f2 = np.var(alert_index)   # Minimizar a variância do índice

    return f1, f2
        

if __name__ == '__main__':

    # 1. Defina os caminhos e parâmetros
    path = 'data_plos'
    ##-- PNH
    time_limit = 30
    distance_limit = 1
    input_file = f"{path}/clusters_{time_limit}d_{distance_limit}km_caracterizados.csv"

    # 2. Carrega e prepara os dados
    df_original = process_data(input_file)

    if df_original is None:
        exit()

    base_col = ["morto", "vivo", "a_quant", "intervalo", "num_reg", "confirmado",
                "freq_num_reg", "freq_morto", "freq_vivo", "freq_a_quant",
                "perc_mortos", "perc_vivos", "perc_agressivo", "perc_doente",
                "perc_estranho", "perc_normal", "extensao"]

    # Normalize e padronize os dados
    df_standardized, df_normalized = normalize_and_standardize_data(df_original, base_col)
    
    # Salvar arquivos (opcional, como no script original)
    output_standardized_file = f"{path}/clusters_{time_limit}d_{distance_limit}km_caracterizados_padronizado.csv"
    output_normalized_file = f"{path}/clusters_{time_limit}d_{distance_limit}km_caracterizados_normalizado.csv"
    df_standardized.to_csv(output_standardized_file, index=True)
    df_normalized.to_csv(output_normalized_file, index=True)
    print(f"Dados padronizados (Z-score) salvos em '{output_standardized_file}'.")
    print(f"Dados normalizados (Min-Max) salvos em '{output_normalized_file}'.")

    # Filtrar por clusters com mais de um registros
    filtered_df_original = df_original[df_original['Cluster2'] != -1]    
    filtered_df_standardized = df_standardized[df_standardized['Cluster2'] != -1]

    # Filtrar apenas pelos casos confirmados (sem considerear clusters duplicados)   
    df_standardized_confirmed = filtered_df_standardized[filtered_df_original['confirmado'] != 0]  
    df_standardized_filtered = df_standardized_confirmed.drop_duplicates(subset=['Cluster1'])    

    # 3. Otimização com SLSQP
    
    # As características do índice de alerta devem ser normalizadas para a otimização
    # Usaremos os dados normalizados para garantir que todas as colunas estejam na mesma escala (0-1).
    alert_cols_for_opt = [ "num_reg", "intervalo", "freq_num_reg", "freq_a_quant", "extensao","perc_mortos", "morto"]
    
    # Número de pesos (variáveis de decisão)
    num_weights = len(alert_cols_for_opt)
    
    # Restrições para o SLSQP
    # 1. A soma dos pesos deve ser igual a 1
    # 2. Cada peso deve ser maior ou igual a 0
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1) for _ in range(num_weights)]
    
    # Definindo os valores de alpha
    alphas = np.arange(0, 1.1, 0.1) #[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,0.7, 0.8, 0.9, 1]
    
    # Dicionário para armazenar os resultados
    results = {}
    
    print("\nIniciando a otimização para diferentes valores de alpha...")
    
    solution = 1
    for alpha in alphas:
        # Chute inicial para os pesos (valores iguais, somando 1)
        initial_weights = np.ones(num_weights) / num_weights
        
        result = minimize(
            objective_function,
            initial_weights,
            args=(df_standardized_filtered, alpha), # Passa o alpha como argumento
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'disp': False, 'ftol':1e-10}
        )
        
        optimal_weights = result.x
        objective_1, objective_2 = calculate_objective_values(optimal_weights, df_standardized_filtered)        

        # Armazena os resultados
        results[solution] = {
            'success': result.success,
            'weights': result.x if result.success else None,
            'objective1' : objective_1,
            'objective2' : objective_2,
            'objective_value': result.fun if result.success else None,
            'message': result.message
        }
        solution = solution + 1
        print(f"Otimização para alpha={alpha:.1f} concluída.")   
        '''
        print("\n--- Resultado da Otimização ---")
        print(f"Sucesso: {result.success}")
        print(f"Mensagem de erro completa: {result.message}")
        print(f"Valor da função objetivo 1: ", objective_1)
        print(f"Valor da função objetivo 2: ", objective_2)
        print(f"Pesos finais: {result.x}")
        print("\n")
        '''

    print("\nCalculando objetivos para o caso sem otimização (pesos uniformes).\n")

    # Sem otimização de pesos
    initial_weights = np.ones(num_weights) / num_weights
    objective_1, objective_2 = calculate_objective_values(initial_weights, df_standardized_filtered)        
    objective_value = 0.5*objective_1 - 0.5*objective_2 # valor não utilizado
    
    # Armazena os resultados 
    results[solution] = {  
        'success': 'True',      
        'weights': initial_weights,
        'objective1' : objective_1,
        'objective2' : objective_2,
        'objective_value': objective_value,
    }

    '''
    # 4. Imprime os resultados da otimização
    if result.success:
        print("\nOtimização concluída com sucesso!\n")
        for alpha, data in results.items():
            print(f"--- Resultados para alpha = {alpha} ---")
            for key, value in data.items():
                print(f"{key}: {value}")
            print("\n")
    else:
        print("A otimização falhou.")
        print("Mensagem de erro:", result.message)
    '''

    print("\n Final results: \n")
    if result.success:
        #print(f"Sol. \t n_rec \t interval \t freq_rec \t freq_animal \t extention \t perc_death \t n_death \t f1 \t f2")
        print(f"{'Sol.':<6}{'n_rec':<8}{'interval':<10}{'freq_rec':<11}{'freq_animal':<13}"
            f"{'extention':<11}{'perc_death':<12}{'n_death':<9}{'f1':<7}{'f2':<7}")
        # Linhas de resultados
        for alpha, data in results.items():
            print(f"{alpha:<6}"                
                f"{data['weights'][0]:<8.3f}"
                f"{data['weights'][1]:<10.3f}"
                f"{data['weights'][2]:<11.3f}"
                f"{data['weights'][3]:<13.3f}"
                f"{data['weights'][4]:<11.3f}"
                f"{data['weights'][5]:<12.3f}"
                f"{data['weights'][6]:<9.3f}"
                f"{data['objective1']:<7.3f}"
                f"{data['objective2']:<7.3f}")
    else:
        print("A otimização falhou.")
        print("Mensagem de erro:", result.message)