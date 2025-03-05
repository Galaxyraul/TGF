from models import *
from utils.data_loader import DatasetLoader
import json
import os
import time
import csv
import torch
class experiment():
    def __init__(self,config):
        self.sizes = config['sizes']
        self.data_path = config['data_path']
        self.n_cpu = config['n_cpu']
        self.n_epochs = config['epochs'] 
        self.batch_size = config['batch_size']
        if config['execute_selected']:
            self.to_execute = config['to_execute']
        else:
            self.to_execute = models.keys()
        self.models_configs = {}
        for c_file in os.listdir(config['configs_path']):
            print(f'Opening {c_file}')
            with open(os.path.join(config['configs_path'],c_file),'r') as f:
                try:
                    self.models_configs[c_file.split('.')[0]] = json.load(f)
                except:
                    self.models_configs[c_file.split('.')[0]] = []
        print(self.models_configs)
        self.times_path = config['times_file']
        self.log_path = config['log_folder']
        with open (self.times_path,'w') as f:
            writer = csv.writer(f)
            writer.writerow(['Model&Key','ex_time'])
        os.makedirs(self.log_path,exist_ok=True)
    def train(self):
        for size in self.sizes:
            data = DatasetLoader(self.data_path+f'/{size}x{size}',self.batch_size,self.n_cpu).get_train()
            print('Dataset Loaded')
            for key in self.to_execute:
                start = time.time()
                model = models[key](data,self.models_configs[key],config,size)
                model.train(self.n_epochs)
                end = time.time()
                torch.cuda.empty_cache()
                print(f'Tiempo tomado para el modelo {key}:{end-start}s')
with open('run.json','r') as f:
    config = json.load(f)
exp = experiment(config)
exp.train()