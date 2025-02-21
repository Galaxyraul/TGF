import os
import argparse
import subprocess
import sys
parser = argparse.ArgumentParser()
parser.add_argument('-p','--path',type=str,help='Path of the models from the base directory')
parser.add_argument('-d','--dataset',type=str,help='Path of the datset folder from the base directory')
parser.add_argument('-i','--img_sizes', type=list, default=[8,16,32,64,128,256], help='list of sizes of each image dimension')
parser.add_argument('-n','--n_epochs', type=int, default=2000, help='number of epochs of training')
parser.add_argument('-si','--sample_interval', type=int, default=100, help='interval between image sampling')
args = parser.parse_args()

models = os.listdir(path=args.path)
models = [os.path.join(args.path,model) for model in models if '.py' in model]
print(models)
conda_env = 'pytorch-gan'
sizes = args.img_sizes
print(sizes)
for size in sizes:
    for model in models:
        try:
            kargs = f'--img_size={size} --n_epochs={args.n_epochs} --path={args.dataset} --sample-interval={args.sample_interval}'
            subprocess.run(f'conda run -n {conda_env} python {model} {kargs}')
        except Exception as e:
            print(f"Error occurred: {e}", file=sys.stderr)