from urllib.parse import urljoin
from urllib.parse import urlparse
from urllib.request import urlopen
from bs4 import BeautifulSoup
import requests


def WordFinder(prediction):

    header = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Safari/537.36'}

    hidden_word =  ''.join(prediction)
    raiz = f'https://palabr.as/buscador-de-palabras?ws=game&ws-r={hidden_word}&ws-p='
    handler = requests.get(raiz, headers=header).text

    soup = BeautifulSoup(handler, 'lxml')
    index = soup.find('tbody')
    if index: 
        words = index.find_all('tr')
    else: return ['No words found!']

    palabras = []
    for word in words:
        
        longitud = word.find('td', class_='wp-scrabble-weight-column')
        if longitud and int(longitud.text) == len(prediction):

            palabra = word.find('td', class_='wp-scrabble-word-column')
            letras = ''.join([i.text for i in palabra.find_all('span')])
            palabras.append(letras)
    print(palabras)
    return palabras

if __name__ == '__main__':
    
    prediction = ['R','O','C','A']
    WordFinder(prediction)