import pandas as pd
from collections import Counter
import ast

genres = pd.read_csv('https://github.com/brenjaras/PI-MLOps/blob/master/Dataset/genres.csv')
genres.genres = genres.genres.apply(ast.literal_eval)
names = pd.read_csv('https://github.com/brenjaras/PI-MLOps/blob/master/Dataset/names.csv')
specs_df = pd.read_csv('https://github.com/brenjaras/PI-MLOps/blob/master/Dataset/specs.csv')
specs_df.specs = specs_df.specs.apply(ast.literal_eval)
early_access_df = pd.read_csv('https://github.com/brenjaras/PI-MLOps/blob/master/Dataset/early_access.csv')
sentiment_df = pd.read_csv('https://github.com/brenjaras/PI-MLOps/blob/master/Dataset/sentiment.csv')
metascore_df = pd.read_csv('https://github.com/brenjaras/PI-MLOps/blob/master/Dataset/metascore.csv')

class Games():

    def get_top_five_genres(year):   
        year_filter = genres.genres[genres.release_year == year]
        all_genres = [i for lista in year_filter for i in lista]
        genre_counts = Counter(all_genres)
        top_5 = dict(genre_counts.most_common(5))
        
        return {f'Top 5 de generos mas vendidos para el año {year}':top_5}
    
    def get_games_of_year(year):
        year_filter = names.app_name[names.release_year == year]
        name = [i for i in year_filter]
        return {f'Juegos lanzados en el año {year}': name}
    
    def get_top_5_specs(year):
        year_filter = specs_df.specs[specs_df.release_year == year]
        all_specs = [i for row in specs_df.specs for i in row]
        specs_count = Counter(all_specs)
        top_5 = dict(specs_count.most_common(5))
        return top_5
    
    def earlyacces(year):   
        year_filter = early_access_df[early_access_df['release_year'] == year]
        cantidad = len(year_filter.id)
        return {f'Cantidad de juegos lanzdos en el año {year} con early acces': cantidad}
    
    def sentiment(year):
        year_filter = sentiment_df.sentiment[sentiment_df['release_year'] == year]
        all_values = [i for i in year_filter]
        count = Counter(all_values)
        return {'Cantidad de registros categorizados con analisis de sentimientos':count}
    
    def metascore(year):
        year_filter = metascore_df[metascore_df['release_year'] == year].sort_values(by='metascore', ascending=False)
        top_5_games = year_filter[['app_name', 'metascore']].head(5)
        top_5_games_list = top_5_games.to_dict(orient='records')  ##genera una lista de objetos las k son appname y metascore
        # top_5_games_dict = dict(zip(top_5_games['app_name'], top_5_games['metascore']))
        return {'Top 5 juegos con mayor metascore':top_5_games_list}