import os
import argparse
import subprocess
import sys
import time
import csv


parser = argparse.ArgumentParser()
parser.add_argument('-p','--path',type=str,default='develop/gans/',help='Path of the models from the base directory')
parser.add_argument('-d','--dataset',type=str,default='develop/datasets_test/',help='Path of the datset folder from the base directory')
parser.add_argument('-i','--img_sizes', type=list, default=[16], help='list of sizes of each image dimension')
parser.add_argument('-n','--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('-si','--sample_interval', type=int, default=100, help='interval between image sampling')
parser.add_argument('-b','--batch_size',type=int,default=16,help='Size of the training batch')
args = parser.parse_args()
executed_file = 'executed.txt'
executed_models=[]
if os.path.exists(executed_file):
    with open (executed_file,'r') as f:
        executed_models = f.read().splitlines()

filename = 'times.csv'
models = os.listdir(path=args.path)
models = sorted(models)
models_name = [model for model in models if '.py' in model and model not in executed_models] 
models = [os.path.join(args.path,model) for model in models_name if 'data_loader' not in model]
print(models_name)
print(len(models))
conda_env = 'pytorch-gan'
sizes = args.img_sizes
total_time=0

with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Model&Key','ex_time'])
    
for size in sizes:
    i=0
    for model in models:
        log_path = f'logs/{models_name[i]}/{size}'
        os.makedirs(log_path,exist_ok=True)
        try:
            kargs = f'--img_size={size} --n_epochs={args.n_epochs} --path={os.path.join(args.dataset,f"{size}x{size}")} --sample_interval={args.sample_interval} --channels=3 --batch_size={args.batch_size}'
            print(f'Lanzando modelo:{model}:\nParametros:{kargs}')
            start = time.time()
            log = subprocess.run(f'conda run -n {conda_env} python {model} {kargs}',shell=True,capture_output=True,text=True)
            end = time.time()
            with open(f'{log_path}/exec.log','w') as f:
                f.write(log.stdout)
            with open(filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([f'{models_name[i]}_{size}', end-start])
                total_time+=end-start
            with open(executed_file,'a') as f:
                f.write(models_name[i])
            i+=1
        except Exception as e:
            print(f"Error occurred: {e}", file=sys.stderr)
            continue

