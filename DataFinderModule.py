import pandas as pd
import numpy as np
import seaborn as sns
import requests
import datetime as dt
from datetime import timedelta
import calendar
import time
import os
from apscheduler.schedulers.background import BackgroundScheduler


class DataProvider:

    def init_data_structs(self):
        self.main_data = None
        self.raw_data = None
        self.dates = []
        self.prices = []
        self.seller_fees = []
        self.buyer_fees = []
        self.gas_fees = []
        self.usd_values = []
        self.composed_data = {}
        self.data_file_name = 'full_idex_data.csv'

    def get_main_data(self, first='ETH', second='AURA', force_update=False, from_file=False):
        
        try:
            self.main_data
            self.raw_data
            self.dates
            self.prices
            self.seller_fees
            self.buyer_fees
            self.gas_fees
            self.usd_values
            self.composed_data
            self.data_file_name
        except AttributeError:
            self.init_data_structs()
            force_update = True
    
        if not force_update:
            if from_file:
                if not os.path.isfile('./' + self.data_file_name):
                    print('File not found. Loading...')
                    self.downloaded_data(first, second)
                    print('Saving to file...')
                    self.load_to_file()
                print('Reading...')
                self.main_data = pd.read_csv(self.data_file_name)
                print('Done reading...')
                return self.main_data
            elif not self.main_data.empty:
                print('Found cached version...')
                return self.main_data
            else:
                print('Force update accepted...')
                return self.downloaded_data(first, second)
        else:
            print('Force update accepted...')
            return self.downloaded_data(first, second)

    def downloaded_data(self, first, second):
        try:
            #'2017-11-22'
            cursor_string = ''
            page = 0
            print('Starting to download...')
            while (True):
                request_raw = 'https://api.idex.market/returnTradeHistory?market=%s_%s&start=%d&end=%d&sort=asc&count=100%s' % (
                    first, second,
                    time.mktime(((2017, 11, 22, 0, 0, 0, 0, 0, 0))),
                    time.mktime(dt.datetime.now().timetuple()),
                    cursor_string)
                response = requests.get(request_raw)
                self.raw_data = response.json()
                if (len(self.raw_data) == 0): break
                for entry in self.raw_data:
                    self.dates.append(entry['date'])
                    self.prices.append(float(entry['price']))
                    self.seller_fees.append(float(entry['sellerFee']))
                    self.buyer_fees.append(float(entry['buyerFee']))
                    self.gas_fees.append(float(entry['gasFee']))
                    self.usd_values.append(float(entry['usdValue']))
                print('Page ' + str(page) + ' - ' + cursor_string)
                cursor_string = '&cursor=' + response.headers['idex-next-cursor']
                page += 1
                
            self.composed_data = {'dates': self.dates, 'prices': self.prices,
                                 'seller_fees': self.seller_fees, 'buyer_fees': self.buyer_fees,
                                 'gas_fees': self.gas_fees, 'usd_values': self.usd_values}
            self.main_data = pd.DataFrame(data=self.composed_data)
            self.main_data['dates'] = pd.to_datetime(self.main_data['dates'], format='%Y-%m-%d %H:%M:%S', yearfirst=True, errors='ignore')
            return self.main_data
        except requests.exceptions.ConnectTimeout:
            print('Oops. Connection timeout occured!')

    def load_to_file(self):
            print('Writing to file...')
            self.main_data.to_csv(self.data_file_name);
            print('Done!')

    def cache_weekly(self):
        self.get_main_data(force_update=True)
        self.load_to_file()

    def __init__(self, debug):
        sns.set(font_scale=1.2)
        if not debug:
            self.scheduler = BackgroundScheduler()
            self.scheduler.scheduled_job(self.cache_weekly, 'interval', weeks=1)
            self.scheduler.start()
            
        
