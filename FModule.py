import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process.kernels import Matern
from sklearn.cluster import KMeans
import seaborn as sns
import datetime as dt
from datetime import timedelta
from sklearn.model_selection import train_test_split
import calendar
import time
from API import Initable

class ProbabilityModel(Initable):

    with_time = '%Y-%m-%d %H:%M:%S'
    without_time = '%Y-%m-%d'
    seed = 42
    part = 0.33
    data_file_name = 'full_idex_eth_aura_data.csv'
    
    def __init__(self):
        sns.set(font_scale=1.2)

    def dates_to_datetime(self, data, format):
        return pd.to_datetime(data['dates'], format=format, yearfirst=True, errors='ignore')

    def reduced_mean(self, data, p):
        result = 0
        n = len(data)
        for i in range(p + 1, n - p):
            result += data[i]
        result /= n - 2 * p
        return result

    def show_box(self, box_data):
        box_data['dates'] = box_data['dates'].apply(lambda x: x[0:10])
        box_data_head = box_data.head(3000)
        sns.boxplot(x='dates', y='prices', data=box_data_head)
        plt.xticks(rotation = 45)
        plt.show()

    def show_hist(self, tmp_main_data):
        fr_data = tmp_main_data['prices'].value_counts()
        fr_data.plot(kind='hist')
        plt.show()

    def show_density(self, tmp_main_data):
        fr_data = tmp_main_data['prices'].value_counts()
        fr_data.plot(kind='density')
        plt.show()

    def prepare_kmeans(self, tmp_main_data):
        pr_content = []
        for (index, value) in tmp_main_data['prices'].iteritems():
           pr_content.append([index, value])
        kmeans = KMeans(n_clusters=10).fit(pr_content)
        tmp_main_data['cluster'] = kmeans.fit_predict(pr_content)
        
    def show_barh(self, tmp_main_data):
        prepare_kmeans(tmp_main_data)
        tmp_main_data['cluster'].value_counts().plot(kind='barh')
        plt.show()
        

    def show_joint_plot(self, tmp_main_data):
        prepare_kmeans(tmp_main_data)
        sns.jointplot(tmp_data['cluster'], tmp_data['prices'], kind="kde", height=7, space=0)
        plt.show()

    def show_heatmap(self, tmp_main_data):
        prepare_kmeans(tmp_main_data)
        sns.heatmap(tmp_main_data['cluster'].value_counts().to_frame(), annot=True)
        plt.show()

    def show_violin(self, vio_data):
        vio_data['dates'] = vio_data['dates'].apply(lambda x: x[0:10])
        vio_data_head = vio_data.head(3000)
        sns.violinplot(x='dates', y='prices', data=vio_data_head)
        plt.xticks(rotation = 45)
        plt.show()

    def prepare_c3(self, tmp_main_data):
        low_high_data = []
        for i in range(0, len(main_data['dates']) - 1):
            if (tmp_main_data['prices'][i] > tmp_main_data['prices'][i + 1]):
                low_high_data.append(1)
            elif (tmp_main_data['prices'][i] < tmp_main_data['prices'][i + 1]):
                low_high_data.append(-1)
            else:
                low_high_data.append(0)
        low_high_data.append(0)
        tmp_main_data['action'] = low_high_data

    def show_c3_hist(self, tmp_main_data):
        prepare_c3(tmp_main_data)
        tmp_main_data['action'].plot(kind='hist')
        plt.show()

    def show_statistics(self, main_data):
        max_price = main_data['prices'].max()
        min_price = main_data['prices'].min()
        describe_price = main_data['prices'].describe()
        print('Statistics: ')
        print(describe_price)
        print('Reduced mean - ' + str(reduced_mean(main_data['prices'], 10000)))
        print('Mode - ' + str(main_data['prices'].mode()))
        print('Mean absolute deviation - ' + str(main_data['prices'].mad()))
        print('Range - ' + str(max_price - min_price))

    def get_dispersion(self, main_data):
        t_std = main_data['prices'].std()
        return t_std * t_std

    def init_model(self):
        print('init model')

    def start_updating_data(self):
        print('start updating data')

    def initialize(self):
        self.start_updating_data()
        self.init_model()

    def get_probability(self, time, high, low):
        print('Getting probability - ' + str(time) + ' - ' + str(high) + ', ' + str(low))
    
    def demonstrate(self):
        main_data = pd.read_csv(data_file_name)
        self.prepare_c3(main_data)
        
        main_data['dates'] = dates_to_datetime(main_data, with_time)
        main_data['dates'] = main_data['dates'].apply(lambda x: x.timestamp())

        X, y = main_data[['dates']], main_data['action']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=part, random_state=seed)
        
        rfc = RandomForestClassifier(random_state=seed).fit(X_train, y_train)

        print(rfc.predict_proba([[time.time()]]))
        
        rfc_results = rfc.predict_proba(X_test)

        print(rfc.score(X_test, y_test))

        rfc_data = pd.DataFrame(data=rfc_results, columns=['-1', '0', '1'])
        rfc_data['dates'] = X_test['dates']
                
        view_data = rfc_data
        
        sns.relplot(x='dates', y='-1', hue='gpc_rfc', data=view_data)
        plt.show()

        sns.relplot(x='dates', y='1', hue='gpc_rfc', data=view_data)
        plt.show()

        sns.relplot(x='dates', y='0', hue='gpc_rfc', data=view_data)
        plt.show()

    
