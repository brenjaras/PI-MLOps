from fastapi import FastAPI
from function import Games

app = FastAPI()

@app.get('/')
def index():
    return 'PI MLOps'

@app.get('/top-genres/{year}')
def genre(year: int):
    return Games.get_top_five_genres(year)

@app.get('/game/{year}')
def games(year: int):
    return Games.get_games_of_year(year)

@app.get('/top-specs/{year}')
def specs(year: int):
    return Games.get_top_5_specs(year)

@app.get('/earlyacces/{year}')
def earlyacces(year: int):
    return Games.earlyacces(year)

@app.get('/sentiment/{year}')
def sentiment(year: int):
    return Games.sentiment(year)

@app.get('/metascore/{year}')
def metascore(year: int):
    return Games.metascore(year)