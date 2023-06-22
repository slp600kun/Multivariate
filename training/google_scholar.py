import requests
from bs4 import BeautifulSoup

# set the base URL for Google Scholar
base_url = 'https://scholar.google.com/scholar'

# set the search query and sort by citations
params = {'q': 'loss function multi-label cosine similarity', 'sort': 'citations'}

# send a request to Google Scholar
response = requests.get(base_url, params=params)

# parse the HTML content with BeautifulSoup
soup = BeautifulSoup(response.content, 'html.parser')

# extract the titles and citations for each article
results = soup.find_all('div', {'class': 'gs_ri'})

for result in results:
    # extract the article title
    title = result.find('h3', {'class': 'gs_rt'}).get_text().strip()

    # extract the number of citations
    citations = result.find('div', {'class': 'gs_fl'}).find_all('a')[-1].get_text()

    print(f'{title} - {citations}')