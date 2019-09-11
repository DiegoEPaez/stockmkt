import requests

SIMFIN_KEY = '02khgmJgxz3m0Acg4eOHB95HSwAHFQei'


def query(url):
    """
    Queries "Banxico" API to get specified series (for example CETES rate 28 days) in time
    frame given
    :param series:     list of series to query for
    :param start_date: begin date for query
    :param end_date:   end date for query
    :return: a JSON object built from API response
    """



    response = requests.get(url)

    if response.status_code == 200:

        # logging.debug("Obtained response from Banxico")

        data = response.json()
        return data
    else:

        # logging.debug("Response status code was not successful from Banxico API")
        return None


ticker = 'CVS'
url = f'https://simfin.com/api/v1/info/find-id/ticker/{ticker}?api-key={SIMFIN_KEY}'
id = query(url)[0]['simId']
ind = '4-13' #EPS Diluted


url = f'https://simfin.com/api/v1/companies/id/{id}/ratios?indicators={ind}&api-key={SIMFIN_KEY}'
print(query(url))

"""
year = 2017
period = 'Q1'
stype = 'pl'

url = f'https://simfin.com/api/v1/companies/id/{id}/statements/standardised?stype={stype}&ptype={period}&fyear={year}&api-key={SIMFIN_KEY}'
print(query(url))
"""
