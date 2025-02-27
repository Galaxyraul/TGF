from models import *
from utils.data_loader import DatasetLoader
import json
import os
import time
import csv
class experiment():
    def __init__(self,config):
        self.sizes = config['sizes']
        self.data_path = config['data_path']
        self.n_cpu = config['n_cpu']
        self.models_saved = config['models_saved']
        self.sample_interval = config['sample_interval']
        self.resume = config['resume']
        self.batch_size = config['batch_size']  
        self.n_epochs = config['epochs'] 
        self.images_path=config['images_saved']
        self.models_configs = {}
        for c_file in os.listdir(config['configs_path']):
            with open(os.path.join(config['configs_path'],c_file),'r') as f:
                self.models_configs[c_file.split('.')[0]] = json.load(f)
        print(self.models_configs)
        self.times_path = config['times_path']
        self.log_path = config['log_folder']
        with open (self.times_path,'w') as f:
            writer = csv.writer(f)
            writer.writerow(['Model&Key','ex_time'])
        os.makedirs(self.log_path,exist_ok=True)
    def train(self):
        for size in self.sizes:
            data = DatasetLoader(self.data_path+f'/{size}x{size}',self.batch_size,self.n_cpu).get_train()
            print('Dataset Loaded')
            for key in models.keys():
                start = time.time()
                model = models[key](data,self.models_configs[key],size,self.sample_interval,f'{self.models_saved}/{key}/{size}x{size}',self.resume,f'{self.images_path}/{key}/{size}x{size}')
                model.train(self.n_epochs)
                end = time.time()
                print(f'Tiempo tomado para el modelo {key}:{end-start}s')
with open('run.json','r') as f:
    config = json.load(f)
exp = experiment(config)
exp.train()