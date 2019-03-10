import requests
import pandas as pd
import datetime as dt
import os
import time

from math import ceil
from apscheduler.schedulers.blocking import BlockingScheduler 

class DataFetcher:

    def week_of_month(self, date):
        first_day = date.replace(day=1)
        dom = date.day
        adjusted_dom = dom + first_day.weekday()
        return int(ceil(adjusted_dom / 7.0))

    def load_price_data(self, filename, init=False):
        request_raw = 'https://api.idex.market/returnTicker?market=%s_%s' % (self.first, self.second)
        response = requests.get(request_raw)
        self.current_raw_data = response.json()
        path = './data/%s' % (filename)
        if init:
            if not os.path.isfile(path):
                with open(path, 'a') as f:
                    os.utime(path, None)
                    f.write(','.join(self.current_raw_data.keys()) + ',time,minute,week_day,month,hour,second,week_number\n')
                    f.close()
        else:
            with open(path, 'a+') as f:
                    os.utime(path, None)
                    timetuple = dt.datetime.now().timetuple()
                    f.write(','
                            .join([str(self.current_raw_data['last']),
                                   str(self.current_raw_data['high']),
                                   str(self.current_raw_data['low']),
                                   str(self.current_raw_data['lowestAsk']),
                                   str(self.current_raw_data['highestBid']),
                                   str(self.current_raw_data['percentChange']),
                                   str(self.current_raw_data['baseVolume']),
                                   str(self.current_raw_data['quoteVolume'])])
                                        + ',' + str(time.mktime(timetuple))
                                        + ',' + str(timetuple[4])
                                        + ',' + str(timetuple[6])
                                        + ',' + str(timetuple[1])
                                        + ',' + str(timetuple[3])
                                        + ',' + str(self.week_of_month(dt.date(timetuple[0], timetuple[1], timetuple[2])))
                                        + '\n')
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

    def load_price_data_months(self):
        print('Loading months data...')
        self.load_price_data(self.months_filename)

    def load_price_data_hours(self):
        print('Loading hours data...')
        self.load_price_data(self.hours_filename)
        
    def load_price_data_seconds(self):
        print('Loading seconds data...')
        self.load_price_data(self.seconds_filename)

    def initialize_first_and_second(self):
        self.first = 'ETH'
        self.second = 'AURA'

    def initialize_filenames(self):
        self.daily_filename = 'daily.csv'
        self.minutes_filename = 'minutes.csv'
        self.weeks_filename = 'weeks.csv'
        self.months_filename = 'months.csv'
        self.hours_filename = 'hours.csv'
        self.seconds_filename = 'seconds.csv'

    def initialize_files(self):
        self.load_price_data(self.daily_filename, init=True)
        self.load_price_data(self.minutes_filename, init=True)
        self.load_price_data(self.weeks_filename, init=True)
        self.load_price_data(self.months_filename, init=True)
        self.load_price_data(self.hours_filename, init=True)
        self.load_price_data(self.seconds_filename, init=True)
        
    def __init__(self, debug=True):
        self.initialize_first_and_second()
        self.initialize_filenames()
        self.initialize_files()
        print('Environment set...')
        if not debug:
            self.scheduler = BlockingScheduler()
            self.scheduler.add_job(self.load_price_data_daily, 'cron', day='*')
            self.scheduler.add_job(self.load_price_data_minutes, 'cron', minute='*')
            self.scheduler.add_job(self.load_price_data_weeks, 'cron', week='*')
            self.scheduler.add_job(self.load_price_data_months, 'cron', month='*')
            self.scheduler.add_job(self.load_price_data_hours, 'cron', hour='*')
            self.scheduler.add_job(self.load_price_data_seconds, 'cron', second='*')
            print('Start the jobs...')
            self.scheduler.start()

if __name__ == '__main__':
    print('Initialize DataFetcher...')
    a = DataFetcher(debug=False)
