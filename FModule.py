import pandas as pd
import numpy as np
import datetime as dt
import calendar
import time
import os
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from API import Initable
from apscheduler.schedulers.background import BackgroundScheduler
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score

class ProbabilityModel(Initable):
    
    #time,minute,week_day,month,hour,second,week_number
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
        self.main_signature_with_time = ['last', 'high', 'low', 'time', 'minute', 'week_day', 'month', 'hour', 'second', 'week_number', 'hash_result']
        self.arguments_signature = ['minute', 'week_day', 'month', 'hour', 'second', 'week_number']
        self.result_signature = ['last', 'high', 'low', 'time']
        self.hash_result_signature = 'hash_result'

    def bootstrap(self):
        print('Bootstraping...')

    def init_main_data(self):
        print('Initalization of main data...')
        self.main_seconds_data = self.raw_seconds_data[self.main_signature_with_time].sample(300)
        self.main_minutes_data = self.raw_minutes_data[self.main_signature_with_time].sample(300)
        self.main_hours_data = self.raw_hours_data[self.main_signature_with_time]
        self.main_daily_data = self.raw_daily_data[self.main_signature_with_time]
        self.main_weeks_data = self.raw_weeks_data[self.main_signature_with_time]
        self.main_months_data = self.raw_months_data[self.main_signature_with_time]
        self.main_data = self.main_seconds_data.append([self.main_minutes_data, self.main_hours_data,
                                                        self.main_hours_data, self.main_daily_data,
                                                        self.main_weeks_data, self.main_months_data])

    def init_raw_data(self):
        print('Loading the raw data...')
        self.raw_seconds_data = pd.read_csv(self.seconds_filename)
        self.raw_seconds_data[self.hash_result_signature] = pd.Series((sum(hash(e) for e in row)
                                                                for i, row in self.raw_seconds_data[self.result_signature].iterrows()))

        self.raw_minutes_data = pd.read_csv(self.minutes_filename)
        self.raw_minutes_data[self.hash_result_signature] = pd.Series((sum(hash(e) for e in row)
                                                                for i, row in self.raw_minutes_data[self.result_signature].iterrows()))
        
        self.raw_hours_data = pd.read_csv(self.hours_filename)
        self.raw_hours_data[self.hash_result_signature] = pd.Series((sum(hash(e) for e in row)
                                                                for i, row in self.raw_hours_data[self.result_signature].iterrows()))
        
        self.raw_daily_data = pd.read_csv(self.daily_filename)
        self.raw_daily_data[self.hash_result_signature] = pd.Series((sum(hash(e) for e in row)
                                                                for i, row in self.raw_daily_data[self.result_signature].iterrows()))
        
        self.raw_weeks_data = pd.read_csv(self.weeks_filename)
        self.raw_weeks_data[self.hash_result_signature] = pd.Series((sum(hash(e) for e in row)
                                                                for i, row in self.raw_weeks_data[self.result_signature].iterrows()))
        
        self.raw_months_data = pd.read_csv(self.months_filename)
        self.raw_months_data[self.hash_result_signature] = pd.Series((sum(hash(e) for e in row)
                                                                for i, row in self.raw_months_data[self.result_signature].iterrows()))

    def init_test_data_parts(self):
        print('Initialization of test data parts...')
        self.seconds_part = 1
        self.minutes_part = 1
        self.hours_part = 1
        self.daily_part = 1
        self.weeks_part = 1
        self.months_part = 1

    def init_seconds_model(self):
        print('Initialization of seconds classifier...')
        self.X_seconds, self.y_seconds = self.main_seconds_data[self.arguments_signature], self.main_seconds_data[self.hash_result_signature]
        self.X_seconds_train, self.X_seconds_test, self.y_seconds_train, self.y_seconds_test = train_test_split(self.X_seconds,
                                                                                        self.y_seconds, test_size=self.seconds_part)
        self.seconds_rfc = RandomForestClassifier(criterion='entropy', n_estimators=100).fit(self.X_seconds_train, self.y_seconds_train)
        print('Seconds RFC configured...')

    def init_minutes_model(self):
        print('Initialization of minutes classifier...')
        self.X_minutes, self.y_minutes = self.main_minutes_data[self.arguments_signature], self.main_minutes_data[self.hash_result_signature]
        self.X_minutes_train, self.X_minutes_test, self.y_minutes_train, self.y_minutes_test = train_test_split(self.X_minutes,
                                                                                        self.y_minutes, test_size=self.minutes_part)
        self.minutes_rfc = RandomForestClassifier(criterion='entropy', n_estimators=100).fit(self.X_minutes_train, self.y_minutes_train)
        print('Minutes RFC configured...')

    def init_hours_model(self):
        print('Initialization of hours classifier...')
        self.X_hours, self.y_hours = self.main_hours_data[self.arguments_signature], self.main_hours_data[self.hash_result_signature]
        self.X_hours_train, self.X_hours_test, self.y_hours_train, self.y_hours_test = train_test_split(self.X_hours,
                                                                                        self.y_hours, test_size=self.hours_part)
        self.hours_rfc = RandomForestClassifier(criterion='entropy', n_estimators=100).fit(self.X_hours_train, self.y_hours_train)
        print('Hours RFC Score configured...')

    def init_daily_model(self):
        print('Initialization of days classifier...')
        self.X_daily, self.y_daily = self.main_daily_data[self.arguments_signature], self.main_daily_data[self.hash_result_signature]
        self.X_daily_train, self.X_daily_test, self.y_daily_train, self.y_daily_test = train_test_split(self.X_daily,
                                                                                        self.y_daily, test_size=self.daily_part)
        self.daily_rfc = RandomForestClassifier(criterion='entropy', n_estimators=100).fit(self.X_daily_train, self.y_daily_train)
        print('Daily RFC Score configured...')

    def init_weeks_model(self):
        print('Initialization of weeks classifier...')
        self.X_weeks, self.y_weeks = self.main_weeks_data[self.arguments_signature], self.main_weeks_data[self.hash_result_signature]
        self.X_weeks_train, self.X_weeks_test, self.y_weeks_train, self.y_weeks_test = train_test_split(self.X_weeks,
                                                                                        self.y_weeks, test_size=self.weeks_part)
        self.weeks_rfc = RandomForestClassifier(criterion='entropy', n_estimators=100).fit(self.X_weeks_train, self.y_weeks_train)
        print('Weeks RFC Score configured...')

    def init_months_model(self):
        print('Initialization of months classifier...')
        self.X_months, self.y_months = self.main_months_data[self.arguments_signature], self.main_months_data[self.hash_result_signature]
        self.X_months_train, self.X_months_test, self.y_months_train, self.y_months_test = train_test_split(self.X_months,
                                                                                        self.y_months, test_size=self.months_part)
        self.months_rfc = RandomForestClassifier(criterion='entropy', n_estimators=100).fit(self.X_months_train, self.y_months_train)
        print('Months RFC configured...')

    def find_rfc_results_by_hashes(self, hashes):
        result = pd.merge(self.main_data, hashes, how='inner').drop_duplicates()
        return result       

    def get_svr_model(self, X_for_rfcs):
        print('Initialization of SVR...')
        data_seconds_train = pd.DataFrame(self.seconds_rfc.predict(X_for_rfcs), columns=[self.hash_result_signature])
        data_minutes_train = pd.DataFrame(self.minutes_rfc.predict(X_for_rfcs), columns=[self.hash_result_signature])
        data_hours_train = pd.DataFrame(self.hours_rfc.predict(X_for_rfcs), columns=[self.hash_result_signature])
        data_daily_train = pd.DataFrame(self.daily_rfc.predict(X_for_rfcs), columns=[self.hash_result_signature])
        data_weeks_train = pd.DataFrame(self.weeks_rfc.predict(X_for_rfcs), columns=[self.hash_result_signature])
        data_months_train = pd.DataFrame(self.months_rfc.predict(X_for_rfcs), columns=[self.hash_result_signature])
        
        svr_train_data_hashes = data_seconds_train.append([data_minutes_train, data_hours_train,
                                                    data_daily_train, data_weeks_train, data_months_train])
        
        svr_data_x_signature = ['time', 'high', 'low']
        svr_data_y_signature = 'last'
        
        svr_train_data = self.find_rfc_results_by_hashes(svr_train_data_hashes)
        svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1).fit(svr_train_data[svr_data_x_signature],
                                                                             svr_train_data[svr_data_y_signature])
        return svr_rbf

    def init_save_filenames(self):
        self.main_save_directory = 'model_saves/first_module'
        self.save_seconds_rfc_path = self.main_save_directory + '/seconds_model.sav'
        self.save_minutes_rfc_path = self.main_save_directory + '/minutes_model.sav'
        self.save_hours_rfc_path = self.main_save_directory +   '/hours_model.sav'
        self.save_daily_rfc_path = self.main_save_directory +   '/daily_model.sav'
        self.save_weeks_rfc_path = self.main_save_directory +   '/weeks_model.sav'
        self.save_months_rfc_path = self.main_save_directory +  '/months_model.sav'
    
    def init_model(self):
        print('Initialization of model...')
        self.init_signatures()
        self.init_filename()
        self.init_save_filenames()
        self.init_test_data_parts()
        self.reinit_specific_models()
        
    def reinit_specific_models(self):
        self.init_raw_data()
        self.init_main_data()
        self.bootstrap()
        if os.path.isfile(self.save_seconds_rfc_path) and os.path.isfile(self.save_minutes_rfc_path) and os.path.isfile(self.save_hours_rfc_path) and os.path.isfile(self.save_daily_rfc_path) and os.path.isfile(self.save_weeks_rfc_path) and os.path.isfile(self.save_months_rfc_path):
            print('Specific models are persist, loading...')
            self.seconds_rfc = pickle.load(open(self.save_seconds_rfc_path, 'rb'))
            self.minutes_rfc = pickle.load(open(self.save_minutes_rfc_path, 'rb'))
            self.hours_rfc = pickle.load(open(self.save_hours_rfc_path, 'rb'))
            self.daily_rfc = pickle.load(open(self.save_daily_rfc_path, 'rb'))
            self.weeks_rfc = pickle.load(open(self.save_weeks_rfc_path, 'rb'))
            self.months_rfc = pickle.load(open(self.save_months_rfc_path, 'rb'))
            print('Models loading is ended...')
        else:
            print('Reinit specific models begins...')
            self.init_seconds_model()
            self.init_minutes_model()
            self.init_hours_model()
            self.init_daily_model()
            self.init_weeks_model()
            self.init_months_model()
            pickle.dump(self.seconds_rfc, open(self.save_seconds_rfc_path, 'wb'))
            pickle.dump(self.minutes_rfc, open(self.save_minutes_rfc_path, 'wb'))
            pickle.dump(self.hours_rfc, open(self.save_hours_rfc_path, 'wb'))
            pickle.dump(self.daily_rfc, open(self.save_daily_rfc_path, 'wb'))
            pickle.dump(self.weeks_rfc, open(self.save_weeks_rfc_path, 'wb'))
            pickle.dump(self.months_rfc, open(self.save_months_rfc_path, 'wb'))
            print('Reinit specific models complete...')

    def start_updating_data(self):
        self.scheduler = BackgroundScheduler()
        self.scheduler.add_job(self.reinit_specific_models, 'cron', week='*')
        print('Start updating data job...')
        self.scheduler.start()

    def print_score(self):
        sample_svr = self.get_svr_model(self.main_data[self.arguments_signature])
        #temp_last_vals = list(map(lambda x: [x], self.main_data['last'].values))
        #print(temp_last_vals)
        #predicted_last_vals = sample_svr.predict()
        #predicted_last_vals = list(map(lambda x: [x], predicted_last_vals))
        #print(predicted_last_vals)
        accuracy_score(sample_svr.predict(self.main_data[['high', 'low', 'time']].values), self.main_data['last'].values)
        #print(sample_svr.score(self.main_data[['high', 'low', 'time']].values, self.main_data['last'].values))
    
    def initialize(self):
        print('Initialization in progress...')
        self.init_model()
        #self.print_score()
        self.start_updating_data()
        print('Probability module is ready to work. Standby...')

    def get_probability(self, high, low, second, minute, hour, week_day, week_number, month, time):
        print('Probability requested: High(%s), Low(%s), Second(%s), Minute(%s), Hour(%s), WeekDay(%s), WeekNumber(%s), Month(%s)' %
              (str(high), str(low), str(second), str(minute), str(hour), str(week_day), str(week_number), str(month)))
        rfc_argument = [[second, minute, hour, week_day, week_number, month]]
        svr_argument = [[time, high, low]]
        self.svr_rbf = self.get_svr_model(rfc_argument)
        result = self.svr_rbf.predict(svr_argument)
        print('Sending result: ' + str(result))
        return result
        """
        y_sec = self.seconds_rfc.predict(argument)
        y_min = self.minutes_rfc.predict(argument)
        y_hour = self.hours_rfc.predict(argument)
        y_day = self.daily_rfc.predict(argument)
        y_week = self.weeks_rfc.predict(argument)
        y_month = self.months_rfc.predict(argument)
       
        svr_y_sec = self.svr_rbf.predict(y_sec)
        svr_y_min = self.svr_rbf.predict(y_min)
        svr_y_hour = self.svr_rbf.predict(y_hour)
        svr_y_day = self.svr_rbf.predict(y_day)
        svr_y_week = self.svr_rbf.predict(y_week)
        svr_y_month = self.svr_rbf.predict(y_month)
        result = {
            'SRFC': y_sec,
            'MRFC': y_min,
            'HRFC': y_hour,
            'DRFC': y_day,
            'WRFC': y_week,
            'MTRFC': y_month,
            'SSVR': svr_y_sec,
            'MSVR': svr_y_min,
            'HSVR': svr_y_hour,
            'DSVR': svr_y_day,
            'WSVR': svr_y_week,
            'MTSVR': svr_y_month
        }"""
        #return result

if __name__ == '__main__':
    a = ProbabilityModel()
    a.initialize()
