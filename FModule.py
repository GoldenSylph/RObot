 import pandas as pd
import numpy as np
import datetime as dt
import calendar
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from API import Initable
from apscheduler.schedulers.background import BackgroundScheduler
from sklearn.svm import SVR

class ProbabilityModel(Initable):

    # x1 = cycle times, x2 = low, x3 = high, y = last, timestamp

    def init_filename(self):
        print('Initialization of filenames...')
        self.folder = './data/'
        self.seconds_filename = self.folder + 'seconds.csv'
        self.minutes_filename = self.folder + 'minutes.csv'
        self.hours_filename = self.folder + 'hours.csv'
        self.daily_filename = self.folder + 'daily.csv'
        self.weeks_filename = self.folder + 'weeks.csv'
        self.months_filename = self.folder + 'months.csv'

    def init_signatures(self):
        print('Initalization of filters and signatures...')
        self.main_signature = ['last', 'high', 'low', 'second', 'minute', 'hour', 'week_day', 'week_number', 'month']
        self.main_signature_with_time = ['last', 'high', 'low', 'second', 'minute', 'hour', 'week_day', 'week_number', 'month', 'time']
        self.arguments_signature = ['high', 'low', 'second', 'minute', 'hour', 'week_day', 'week_number', 'month']
        self.result_signature = ['last', 'time']

    def bootstrap(self):
        print('Bootstraping...')

    def init_main_data(self):
        print('Initalization of main data...')
        self.main_seconds_data = self.raw_seconds_data[self.main_signature_with_time]
        self.main_minutes_data = self.raw_minutes_data[self.main_signature_with_time]
        self.main_hours_data = self.raw_hours_data[self.main_signature_with_time]
        self.main_daily_data = self.raw_daily_data[self.main_signature_with_time]
        self.main_weeks_data = self.raw_weeks_data[self.main_signature_with_time]
        self.main_months_data = self.raw_months_data[self.main_signature_with_time]

    def init_raw_data(self):
        print('Loading the raw data...')
        self.raw_seconds_data = pd.read_csv(self.seconds_filename)
        self.raw_minutes_data = pd.read_csv(self.minutes_filename)
        self.raw_hours_data = pd.read_csv(self.hours_filename)
        self.raw_daily_data = pd.read_csv(self.daily_filename)
        self.raw_weeks_data = pd.read_csv(self.weeks_filename)
        self.raw_months_data = pd.read_csv(self.months_filename)

    def init_test_data_parts(self):
        print('Initialization of test data parts...')
        self.seconds_part = 0.5
        self.minutes_part = 0.5
        self.hours_part = 0.5
        self.daily_part = 0.5
        self.weeks_part = 0.5
        self.months_part = 0.5

    def init_seconds_model(self):
        print('Initialization of seconds classifier...')
        self.X_seconds, self.y_seconds = self.main_seconds_data[self.arguments_signature], self.main_seconds_data[self.result_signature]
        self.X_seconds_train, self.X_seconds_test, self.y_seconds_train, self.y_seconds_test = train_test_split(self.X_seconds,
                                                                                        self.y_seconds, test_size=self.seconds_part)
        self.seconds_rfc = RandomForestClassifier(criterion='entropy').fit(X_seconds_train, y_seconds_train)

    def init_minutes_model(self):
        print('Initialization of minutes classifier...')
        self.X_minutes, self.y_minutes = self.main_minutes_data[self.arguments_signature], self.main_minutes_data[self.result_signature]
        self.X_minutes_train, self.X_minutes_test, self.y_minutes_train, self.y_minutes_test = train_test_split(self.X_minutes,
                                                                                        self.y_minutes, test_size=self.minutes_part)
        self.minutes_rfc = RandomForestClassifier(criterion='entropy').fit(X_minutes_train, y_minutes_train)

    def init_hours_model(self):
        print('Initialization of hours classifier...')
        self.X_hours, self.y_hours = self.main_hours_data[self.arguments_signature], self.main_hours_data[self.result_signature]
        self.X_hours_train, self.X_hours_test, self.y_hours_train, self.y_hours_test = train_test_split(self.X_hours,
                                                                                        self.y_hours, test_size=self.hours_part)
        self.hours_rfc = RandomForestClassifier(criterion='entropy').fit(X_hours_train, y_hours_train)

    def init_daily_model(self):
        print('Initialization of days classifier...')
        self.X_daily, self.y_daily = self.main_daily_data[self.arguments_signature], self.main_daily_data[self.result_signature]
        self.X_daily_train, self.X_daily_test, self.y_daily_train, self.y_daily_test = train_test_split(self.X_daily,
                                                                                        self.y_daily, test_size=self.daily_part)
        self.daily_rfc = RandomForestClassifier(criterion='entropy').fit(X_daily_train, y_daily_train)

    def init_weeks_model(self):
        print('Initialization of weeks classifier...')
        self.X_weeks, self.y_weeks = self.main_weeks_data[self.arguments_signature], self.main_weeks_data[self.result_signature]
        self.X_weeks_train, self.X_weeks_test, self.y_weeks_train, self.y_weeks_test = train_test_split(self.X_weeks,
                                                                                        self.y_weeks, test_size=self.weeks_part)
        self.weeks_rfc = RandomForestClassifier(criterion='entropy').fit(X_weeks_train, y_weeks_train)

    def init_months_model(self):
        print('Initialization of months classifier...')
        self.X_months, self.y_months = self.main_months_data[self.arguments_signature], self.main_months_data[self.result_signature]
        self.X_months_train, self.X_months_test, self.y_months_train, self.y_months_test = train_test_split(self.X_months,
                                                                                        self.y_months, test_size=self.months_part)
        self.months_rfc = RandomForestClassifier(criterion='entropy').fit(X_months_train, y_months_train)

    def init_svr_model(self):
        """self.aggregated_rfc_x = self.X_seconds_train.append([self.X_minutes_train, self.X_hours_train,
                                                             self.X_daily_train, self.X_weeks_train, self.X_months_train])
        self.aggregated_rfc_y = self.y_seconds_train.append([self.y_minutes_train, self.y_hours_train,
                                                             self.y_daily_train, self.y_weeks_train, self.y_months_train])"""
        self.svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1).fit(self.aggregated_train_rfc_x, self.aggregated_train_rfc_y)
    
    def init_model(self):
        print('Initialization of model...')
        self.init_signatures()
        self.init_filename()
        self.init_test_data_parts()
        self.reinit_specific_models()

    def reinit_specific_models(self):
        print('Reinit specific models begins...')
        self.init_raw_data()
        self.init_main_data()
        self.bootstrap()
        self.init_seconds_model()
        self.init_minutes_model()
        self.init_hours_model()
        self.init_daily_model()
        self.init_weeks_model()
        self.init_months_model()
        self.init_svr_model()
        print('Reinit specific models complete...')

    def start_updating_data(self):
        self.scheduler = BackgroundScheduler()
        self.scheduler.add_job(self.reinit_specific_models, 'cron', week='*')
        print('Start updating data job...')
        self.scheduler.start()
            
    def initialize(self):
        print('Initialization in progress...')
        self.init_model()
        self.start_updating_data()
        print('Probability module is ready to work. Standby...')

    def get_probability(self, high, low, second, minute, hour, week_day, week_number, month):
        print('Probability requested: High(%s), Low(%s), Second(%s), Minute(%s), Hour(%s), WeekDay(%s), WeekNumber(%s), Month(%s)' %
              (str(high), str(low), str(second), str(minute), str(hour), str(week_day), str(week_number), str(month)))
        argument = [[high, low, second, minute, hour, week_day, week_number, month]]
        y_sec = self.seconds_rfc.predict(argument)
        y_min = self.minutes_rfc.predict(argument)
        y_hour = self.hours_rfc.predict(argument)
        y_day = self.daily_rfc.predict(argument)
        y_week = self.weeks_rfc.predict(argument)
        y_month = self.months_rfc.predict(argument)
        
