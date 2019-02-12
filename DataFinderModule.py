import pandas as pd
import numpy as np
import seaborn as sns
import requests
import datetime as dt
from datetime import timedelta
import calendar
import time

sns.set(font_scale=1.2)

raw_data = None
try:
    dates = []
    prices = []
    seller_fees = []
    buyer_fees = []
    gas_fees = []
    usd_values = []

    #'2017-11-22'
    cursor_string = ''
    page = 0
    print('Starting to download...')
    while (True):
        request_raw = 'https://api.idex.market/returnTradeHistory?market=ETH_AURA&start=%d&end=%d&sort=asc&count=100%s' % (time.mktime(((2017, 11, 22, 0, 0, 0, 0, 0, 0))),
           time.mktime(dt.datetime.now().timetuple()), cursor_string)
        response = requests.get(request_raw)
        raw_data = response.json()
        if (len(raw_data) == 0): break
        for entry in raw_data:
            dates.append(entry['date'])
            prices.append(float(entry['price']))
            seller_fees.append(float(entry['sellerFee']))
            buyer_fees.append(float(entry['buyerFee']))
            gas_fees.append(float(entry['gasFee']))
            usd_values.append(float(entry['usdValue']))
        print('Page ' + str(page) + ' - ' + cursor_string)
        cursor_string = '&cursor=' + response.headers['idex-next-cursor']
        page += 1

    
    composed_data = {'dates': dates, 'prices': prices,
                     'seller_fees': seller_fees, 'buyer_fees': buyer_fees,
                     'gas_fees': gas_fees, 'usd_values': usd_values}
    main_data = pd.DataFrame(data=composed_data)
    main_data['dates'] = pd.to_datetime(main_data['dates'], format='%Y-%m-%d %H:%M:%S', yearfirst=True, errors='ignore')
    print('Writing to file...')
    main_data.to_csv("full_idex_eth_aura_data.csv");
    print("Done!")
    
except requests.exceptions.ConnectTimeout:
    print('Oops. Connection timeout occured!')
