import requests
import json


def snow_api_call():
    # Set the request parameters
    url = 'https://cgigroupincdemo15.service-now.com/api/now/table/incident'

    user = 'api_user'
    pwd = 'V.MLI9S&bV74'

    # Set proper headers
    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    # Do the HTTP request
    response = requests.get(url, auth=(user, pwd), headers=headers)

    # Check for HTTP codes other than 200
    if response.status_code != 200:
        print('Status:', response.status_code, 'Headers:', response.headers, 'Error Response:', response.json())
        exit()

    # Decode the JSON response into a dictionary and use the data
    data = response.json()
    print(data)

    with open('data/snow/response.json', 'w+') as f:
        json.dump(data, f)


snow_api_call()
