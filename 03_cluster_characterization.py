import pandas as pd
import numpy as np
from geopy.distance import geodesic

def calculate_spatial_extent(df):
    """
    Calcula a extensão espacial (distância geodésica entre os pontos
    mais distantes) para cada cluster.
    """
    loc_data = df.groupby(['Cluster1','Cluster2']).agg(
        r_lat_min=('r_lat', 'min'),
        r_lat_max=('r_lat', 'max'),
        r_long_min=('r_long', 'min'),
        r_long_max=('r_long', 'max')
    )
    
    extensao = []
    for index, row in loc_data.iterrows():
        coords1 = (row['r_lat_min'], row['r_long_min'])
        coords2 = (row['r_lat_max'], row['r_long_max'])
        dist = geodesic(coords1, coords2).km
        extensao.append(dist)
    
    loc_data['extensao'] = extensao
    return loc_data[['extensao']]

def standardize_data(df):
    """
    Uniformização dos campos 'classificação' e 'doença' no df original
    """
    df['d_classificacao'] = np.where( df['d_classificacao'].notna() & df['d_classificacao'].str.contains('"Confirmada"'), '"Confirmada"', df['d_classificacao']) 
    df['d_classificacao'] = np.where( df['d_classificacao'].notna() & df['d_classificacao'].str.contains('"Indeterminada"'), '"Indeterminada"', df['d_classificacao']) 
    df['d_classificacao'] = np.where( df['d_classificacao'].notna() & df['d_classificacao'].str.contains('"Descartada"'), '"Descartada"', df['d_classificacao']) 
    df['d_doenca'] = np.where( df['d_doenca'].notna() & df['d_doenca'].str.contains('"Febre Amarela"'), '"Febre Amarela"', df['d_classificacao']) 

    return df


def calculate_cluster_characteristics(input_csv, output_csv):
    """
    Calcula as características de cada cluster a partir de um arquivo CSV
    que já contém os registros com suas respectivas atribuições de cluster.

    Args:
        input_csv (str): Caminho para o arquivo CSV de entrada com os clusters.
        output_csv (str): Caminho para salvar o arquivo CSV de saída com as
                          características dos clusters.
    """
    try:
        # Leitura do arquivo de entrada com os clusters
        data = pd.read_csv(input_csv, index_col=0)
        
        # Converter a coluna de data para o formato datetime, se ainda não estiver
        data['r_data'] = pd.to_datetime(data['r_data'])

        # Uniformizar os atributos de classificação e doença  
        data_uniform = standardize_data(data)
            
        # df contendo os registros com os clusters        
        df_clusters = data_uniform.copy()        

        # 1. Calcular a quantidade de 'mortos' e 'vivos' por cluster
        count_situacao = df_clusters.groupby(["Cluster1","Cluster2","a_situacao"])['a_quantidade'].sum().unstack(fill_value=0)
        count_situacao = count_situacao.rename(columns={'Morto': 'morto', 'Vivo': 'vivo'})

        # 2. Calcular a quantidade de 'Normal', 'Doente', 'Estranho' e 'Agressivo' por cluster
        # As categorias 'Normal', 'Doente', 'Estranho' e 'Agressivo' só aparecem quando 'a_comportamento' é diferente de nulo
        #   Essas categorias só aparecem quando tem animal vivo no cluster
        # Neste caso, criei uma catedoria 'sem_info' quando não houver informação em 'a_comportamento', ou seja,
        #   quando no cluster só tiver animais mortos.
        # Em seguida, eu deleto a coluna 'sem_info'.
        # Assim, consegui garantir o valor 'zero' em todas as colunas onde só tem animais mortos no cluster
        df = df_clusters.copy()        
        df.loc[df['a_comportamento'].isnull(), 'a_comportamento'] = 'sem_info'        
        count_comportamento = df.groupby(["Cluster1","Cluster2","a_comportamento"])["a_quantidade"].sum().unstack(fill_value=0)
        count_comportamento = count_comportamento.rename(columns={'Normal':"normal",'Estranho':"estranho", 'Doente': "doente", 'Agressivo': "agressivo"})        
        count_comportamento = count_comportamento.drop(columns='sem_info')

        # 3. Calcular a soma da quantidade de animais por cluster
        quant_animais = df_clusters.groupby(["Cluster1","Cluster2"])['a_quantidade'].sum().rename('a_quant')        

        # 4. Calcular o intervalo de tempo de cada cluster
        dates = df_clusters.groupby(["Cluster1","Cluster2"])['r_data'].agg(['min', 'max']).rename(columns={'min':"data_ini",'max':"data_fim"})
        dates["intervalo"] = (dates["data_fim"] - dates["data_ini"]).dt.days 
        dates["intervalo"] = dates["intervalo"].apply(lambda x: x if x > 0 else 1)                

        # 5. Calcular a extensão espacial de cada cluster 
        extensao = calculate_spatial_extent(df_clusters)

        # 6. Calcula o número de registros confirmados por alguma doença
        confirmados_por_cluster = df_clusters.groupby(["Cluster1","Cluster2"])['d_classificacao'].apply(
            lambda x: (x == '"Confirmada"').sum()
        ).rename("confirmado")   

        # 6.1. Lista completa de geocodes por cluster
        geocodes_list = (
            df_clusters.groupby(["Cluster1","Cluster2"])["geocode"]
            .apply(lambda x: sorted(x.unique()))
            .rename("geocode_list")
        )   
        #print(geocodes_list[geocodes_list.apply(len) > 1])                                        

        # 7. Unir todos os atributos calculados
        df_clusters_union = (
            pd.DataFrame(quant_animais)
            .join(count_situacao, on=['Cluster1','Cluster2'])
            .join(count_comportamento, on=['Cluster1','Cluster2'])
            .join(dates[['intervalo','data_ini','data_fim']], on=['Cluster1','Cluster2'])
            .join(extensao, on=['Cluster1','Cluster2'])            
            .join(confirmados_por_cluster, on=['Cluster1','Cluster2'])
        )        

        # 8. Calcular número de registros por cluster
        df_clusters_union['num_reg'] = df_clusters.groupby(['Cluster1','Cluster2']).size()
        
        # 9. Calcular atributos de frequência
        df_clusters_union['freq_num_reg'] = df_clusters_union['num_reg'] / df_clusters_union['intervalo']
        df_clusters_union['freq_a_quant'] = df_clusters_union['a_quant'] / df_clusters_union['intervalo']
        df_clusters_union['freq_vivo'] = df_clusters_union['vivo'] / df_clusters_union['intervalo']
        df_clusters_union['freq_morto'] = df_clusters_union['morto'] / df_clusters_union['intervalo']
        
        # 10. Calcular atributos de percentual
        df_clusters_union['perc_mortos'] = (df_clusters_union['morto'] / df_clusters_union['a_quant']).fillna(0)
        df_clusters_union['perc_vivos'] = (df_clusters_union['vivo'] / df_clusters_union['a_quant']).fillna(0)        
        df_clusters_union['perc_normal'] = (df_clusters_union['normal'] / df_clusters_union['a_quant']).fillna(0)        
        df_clusters_union['perc_estranho'] = (df_clusters_union['estranho'] / df_clusters_union['a_quant']).fillna(0)        
        df_clusters_union['perc_doente'] = (df_clusters_union['doente'] / df_clusters_union['a_quant']).fillna(0)        
        df_clusters_union['perc_agressivo'] = (df_clusters_union['agressivo'] / df_clusters_union['a_quant']).fillna(0)        
        df_clusters_union.reset_index(inplace=True)    
        
        # 11. Adicionar lista de geocodes ao dataframe
        df_clusters_union = df_clusters_union.merge(
            geocodes_list,
            on=["Cluster1","Cluster2"],
            how="left"
        )
        # Expandir: uma linha por geocode
        df_clusters_exploded = df_clusters_union.explode("geocode_list")
        # Renomear a coluna explodida
        df_clusters_exploded = df_clusters_exploded.rename(columns={"geocode_list": "geocode"})
        
        # Adicionar as colunas MUN e UF com base no geocode
        df_clusters_exploded = df_clusters_exploded.merge(
            df_clusters[['geocode', 'MUN', 'UF']].drop_duplicates(),
            on='geocode',
            how='left'
        )

        # 12. Salvar o dataframe resultante
        #### df_clusters_union.to_csv(output_csv, index=True)
        df_clusters_exploded.to_csv(output_csv, index=True)

        print(f"O arquivo com as características dos clusters foi salvo como '{output_csv}'.")

    except FileNotFoundError:
        print(f"Erro: O arquivo '{input_csv}' não foi encontrado.")
    except Exception as e:
        print(f"Ocorreu um erro durante a análise dos clusters: {e}")


if __name__ == '__main__':
    # Defina os caminhos dos arquivos de entrada e saída
    # O arquivo de entrada deve conter as colunas 'Cluster2', 'r_data', 'a_situacao', 'a_quantidade',
    # 'r_lat', 'r_long', 'd_doenca' e 'd_classificacao'.

    path = 'data_plos'
    ##-- PNH
    time_limit_days = 30
    distance_limit_km = 1

    input_file = f'{path}/clusters_{time_limit_days}d_{distance_limit_km}km.csv'
    output_file = f'{path}/clusters_{time_limit_days}d_{distance_limit_km}km_caracterizados.csv'

    # Chama a função principal
    calculate_cluster_characteristics(input_file, output_file)