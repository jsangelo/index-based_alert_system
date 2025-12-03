import pandas as pd

def preprocess_data(input_csv):
    """
    Realiza o pré-processamento de um arquivo CSV de registros.

    O pré-processamento inclui as seguintes etapas:
    1. Renomeia as colunas do DataFrame para nomes mais curtos e padronizados.
    2. Converte a coluna 'r_data' para o tipo de dado datetime e mantém apenas a data.
    3. Filtra os dados para incluir apenas registros onde a origem da localização
       é "Obtido pelo GPS ou informado explicitamente".
    4. Filtra ainda mais os dados para incluir apenas registros com uma precisão
       inferior a 100 metros (ou seja, 'r_precisao' < 100).
    5. Remove registros onde o valor de 'r_precisao' é -1.

    Args:
        input_csv (str): O caminho do arquivo CSV de entrada.

    Returns:
        pandas.DataFrame: O DataFrame processado e filtrado.
    """
    try:
        # 1 -- Lendo arquivos de entrada
        base_original = pd.read_csv(input_csv, encoding='latin-1', sep=';')

        # Renomeando colunas
        df = base_original.rename(columns={
            "Registro: Identificador": "r_reg",
            "Registro: Longitude": "r_long",
            "Registro: Latitude": "r_lat",
            "Registro: Data de observação (ISO)": "r_data",
            "Registro: Estado": "r_estado",
            "Registro: Município": "r_municipio",
            "Registro: Origem da localização": "r_origem",
            "Registro: Precisão": "r_precisao",
            "Animal: Identificador": "a_ident",
            "Animal: Tipo": "a_tipo",
            "Animal: Quantidade observada": "a_quantidade",
            "Animal: Situação": "a_situacao",
            "Animal: Comportamento": "a_comportamento",
            "Animal: Condição física": "a_condicao",
            "Animal: Causa morte": "a_causa_morte",
            "Desfecho: Doença": "d_doenca",
            "Desfecho: Classificação": "d_classificacao"
        })

        # Converter a coluna 'r_data' para o formato datetime e extrair apenas a data
        df['r_data'] = pd.to_datetime(df['r_data']).dt.date

        # 2 -- Filtrando os dados
        # Filtrar por origem de localização
        data_gps = df[df['r_origem'] == "Obtido pelo GPS ou informado explicitamente"]
        
        # Filtrar por precisão < 100
        data_gps_prec = data_gps[data_gps['r_precisao'] < 100]

        # Remover valores de precisão igual a -1
        data_gps_prec_final = data_gps_prec[data_gps_prec['r_precisao'] != -1]

        # Cria uma cópia do dataframe final 
        X = data_gps_prec_final.copy()
        
        print("Pré-processamento concluído.")
        print(f"O DataFrame final contém {len(X)} registros.")

        return X

    except FileNotFoundError:
        print(f"Erro: O arquivo '{input_csv}' não foi encontrado.")
        return None
    except Exception as e:
        print(f"Ocorreu um erro durante o processamento: {e}")
        return None

if __name__ == '__main__':
    # Define o nome do arquivo de entrada e saída
    ##-- PNH
    path = 'data_plos'
    file = 'registros_macacos_micos_01_05_2014_ate_31_12_2024'    
    ##-- Aves
    #path = 'data_E_vigilancia'
    #file = 'registros_aves'

    input_file = f'{path}/{file}.csv'
    output_file = f'{path}/{file}_filtrado.csv'

    # Chama a função de pré-processamento
    processed_df = preprocess_data(input_file)

    # Salva o DataFrame resultante em um novo arquivo CSV se o processamento foi bem-sucedido
    if processed_df is not None:
        processed_df.to_csv(output_file, index=False)
        print(f"O arquivo processado foi salvo como '{output_file}'.")