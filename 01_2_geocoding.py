import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

def geocode_data(input_csv, output_csv_7d, output_csv_6d, mun_shp, uf_shp):
    """
    Atribui geocodes a um DataFrame de pontos usando shapefiles do IBGE.

    Args:
        input_csv (str): O caminho do arquivo CSV de entrada que contém as coordenadas.
        output_csv_7d (str): O caminho para salvar o arquivo CSV com geocodes de 7 dígitos.
        output_csv_6d (str): O caminho para salvar o arquivo CSV com geocodes de 6 dígitos.
        mun_shp (str): O caminho do shapefile de municípios.
        uf_shp (str): O caminho do shapefile de unidades federativas (estados).

    Returns:
        tuple: Uma tupla contendo o DataFrame com geocodes de 7 dígitos e o DataFrame
               com geocodes de 6 dígitos. Retorna (None, None) se ocorrer um erro.
    """
    try:
        # Carregar os shapefiles
        gdf_mun = gpd.read_file(mun_shp)
        gdf_uf = gpd.read_file(uf_shp)

        # Selecionar colunas relevantes e renomear a sigla da UF
        gdf_uf_reduzido = gdf_uf[['CD_UF', 'SIGLA_UF']]

        # Juntar a sigla da UF ao GeoDataFrame de municípios
        gdf_mun = gdf_mun.merge(gdf_uf_reduzido, on='CD_UF', how='left')

        # Carregar o CSV de entrada
        df_pontos = pd.read_csv(input_csv)

        # Criar o GeoDataFrame a partir do DataFrame de pontos
        def coordenadas_para_geodf(df, lat_col='r_lat', lon_col='r_long'):
            geometry = [Point(lon, lat) for lat, lon in zip(df[lat_col], df[lon_col])]
            return gpd.GeoDataFrame(df.copy(), geometry=geometry, crs="EPSG:4326")

        gdf_coords = coordenadas_para_geodf(df_pontos)

        # Realizar o spatial join para encontrar o município de cada ponto
        gdf_join = gpd.sjoin(
            gdf_coords.to_crs(gdf_mun.crs),  # Reprojetar para o CRS do shapefile de municípios
            gdf_mun[['CD_MUN', 'NM_MUN', 'SIGLA_UF', 'geometry']],
            how='left',
            predicate='within'
        )

        # Adicionar as novas colunas ao DataFrame original
        df_pontos['geocode'] = gdf_join['CD_MUN']
        df_pontos['MUN'] = gdf_join['NM_MUN']
        df_pontos['UF'] = gdf_join['SIGLA_UF']

        # Verificar registros com UF nulo
        df_com_nulos = df_pontos[df_pontos.UF.isnull()]
        if not df_com_nulos.empty:
            print("Registros com UF nulo encontrados e removidos:")
            print(df_com_nulos[['r_reg', 'r_lat', 'r_long', 'r_estado', 'r_municipio']])
            df_novo = df_pontos.dropna(subset=['UF']).copy()
        else:
            df_novo = df_pontos.copy()

        # Salvar o arquivo com geocode de 7 dígitos
        df_novo.to_csv(output_csv_7d, index=False)
        print(f"Arquivo com geocodes de 7 dígitos salvo como '{output_csv_7d}'.")

        # Criar uma cópia para o arquivo de 6 dígitos
        X = df_novo.copy()
        X['geocode'] = X['geocode'].astype(str).str[:-1]
        
        # Salvar o arquivo com geocode de 6 dígitos
        X.to_csv(output_csv_6d, index=False)
        print(f"Arquivo com geocodes de 6 dígitos salvo como '{output_csv_6d}'.")

        return df_novo, X

    except FileNotFoundError as e:
        print(f"Erro: O arquivo {e.filename} não foi encontrado.")
        return None, None
    except Exception as e:
        print(f"Ocorreu um erro durante o geocoding: {e}")
        return None, None

if __name__ == '__main__':
    # Defina os caminhos dos arquivos de entrada e saída e dos shapefiles
    path = 'data_plos'
    file = 'registros_macacos_micos_01_05_2014_ate_31_12_2024_filtrado'    
    
    input_file = f'{path}/{file}.csv'
    output_7d_file = f'{path}/{file}_limpo_geocode_7d.csv'
    output_6d_file = f'{path}/{file}_limpo_geocode_6d.csv'
    municipios_shp = f'{path}/BR_Municipios_2023/BR_Municipios_2023.shp'
    uf_shp = f'{path}/BR_UF_2023/BR_UF_2023.shp'

    # Chama a função principal
    df_7d, df_6d = geocode_data(input_file, output_7d_file, output_6d_file, municipios_shp, uf_shp)

    if df_7d is not None and df_6d is not None:
        print("\nProcessamento concluído com sucesso.")
        print(f"DataFrame com geocode de 7 dígitos tem {len(df_7d)} registros.")
        print(f"DataFrame com geocode de 6 dígitos tem {len(df_6d)} registros.")