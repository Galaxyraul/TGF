import os
import argparse
import subprocess
import sys
import time
import csv

parser = argparse.ArgumentParser()
parser.add_argument('-p','--path',type=str,default='develop/tested_models/',help='Path of the models from the base directory')
parser.add_argument('-d','--dataset',type=str,default='develop/datasets_augmented/',help='Path of the datset folder from the base directory')
parser.add_argument('-i','--img_sizes', type=list, default=[16,32,64,128,256], help='list of sizes of each image dimension')
parser.add_argument('-n','--n_epochs', type=int, default=2000, help='number of epochs of training')
parser.add_argument('-si','--sample_interval', type=int, default=50, help='interval between image sampling')
args = parser.parse_args()

models = os.listdir(path=args.path)
models_name = [model.split('.')[0] for model in models if '.py' in model] 
models = [os.path.join(args.path,model) for model in models if '.py' in model and 'data_loader' not in model]
print(models_name)
print(len(models))
conda_env = 'pytorch-gan'
sizes = args.img_sizes
times={}
for size in sizes:
    i=0
    for model in models:
        log_path = f'logs/{models_name[i]}/{size}'
        os.makedirs(log_path,exist_ok=True)
        try:
            kargs = f'--img_size={size} --n_epochs={args.n_epochs} --path={os.path.join(args.dataset,f"{size}x{size}")} --sample_interval={args.sample_interval} --channels=3'
            print(f'Lanzando modelo:{model}:\nParametros:{kargs}')
            start = time.time()
            log = subprocess.run(f'conda run -n {conda_env} python {model} {kargs}',shell=True,capture_output=True,text=True)
            end = time.time()
            with open(f'{log_path}/exec.log','w') as f:
                f.write(log.stdout)
            times[f'{models_name[i]}_{size}'] = end-start
            print(f'Finalizado modelo:{model} en {end-start}s\n')
            i+=1
        except Exception as e:
            print(f"Error occurred: {e}", file=sys.stderr)

filename = 'times.csv'
total_time=0
with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Write the rows (key-value pairs)
    writer.writerow(['Model&Key','ex_time'])
    for key, value in times.items():
        writer.writerow([key, value])
        total_time+=value
    writer.writerow(['Total',total_time])
print(f'Tiempo total:{total_time}')