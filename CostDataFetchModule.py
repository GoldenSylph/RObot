import requests
import pandas as pd
import datetime as dt
import os
import time

from apscheduler.schedulers.blocking import BlockingScheduler 

class DataFetcher:

    def load_price_data(self, filename, init=False):
        request_raw = 'https://api.idex.market/returnTicker?market=%s_%s' % (self.first, self.second)
        response = requests.get(request_raw)
        self.current_raw_data = response.json()
        path = './%s' % (filename)
        if init:
            if not os.path.isfile(path):
                with open(path, 'a') as f:
                    os.utime(path, None)
                    f.write(','.join(self.current_raw_data.keys()) + ',time\n')
                    f.close()
        else:
            with open(path, 'a+') as f:
                    os.utime(path, None)
                    f.write(','
                            .join([str(self.current_raw_data['last']),
                                   str(self.current_raw_data['high']),
                                   str(self.current_raw_data['low']),
                                   str(self.current_raw_data['lowestAsk']),
                                   str(self.current_raw_data['highestBid']),
                                   str(self.current_raw_data['percentChange']),
                                   str(self.current_raw_data['baseVolume']),
                                   str(self.current_raw_data['quoteVolume'])])
                                        + ',' + str(time.mktime(dt.datetime.now().timetuple())) + '\n')
                    f.close()
            

    def load_price_data_daily(self):
        print('Loading daily data...')
        self.load_price_data(self.daily_filename)

    def load_price_data_minutes(self):
        print('Loading minutes data...')
        self.load_price_data(self.minutes_filename)

    def load_price_data_weeks(self):
        print('Loading weeks data...')
        self.load_price_data(self.weeks_filename)

    def initialize_first_and_second(self):
        self.first = 'ETH'
        self.second = 'AURA'

    def initialize_filenames(self):
        self.daily_filename = 'daily.csv'
        self.minutes_filename = 'minutes.csv'
        self.weeks_filename = 'weeks.csv'

    def initialize_files(self):
        self.load_price_data(self.daily_filename, init=True)
        self.load_price_data(self.minutes_filename, init=True)
        self.load_price_data(self.weeks_filename, init=True)
        
    def __init__(self, debug=True):
        self.initialize_first_and_second()
        self.initialize_filenames()
        self.initialize_files()
        print('Environment set...')
        if not debug:
            self.scheduler = BlockingScheduler()
            self.scheduler.add_job(self.load_price_data_daily, 'interval', days=1)
            self.scheduler.add_job(self.load_price_data_minutes, 'interval', minutes=1)
            self.scheduler.add_job(self.load_price_data_weeks, 'interval', weeks=1)
            print('Start the jobs...')
            self.scheduler.start()

if __name__ == '__main__':
    print('Initialize DataFetcher...')
    a = DataFetcher(debug=False)
