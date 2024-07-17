#!/usr/bin/env python-real
import json
# import sys
import os
import argparse
from pathlib import Path
import sys
# sys.path.append('/Users/lumargot/Documents/ShapeAXI/')
# import shapeaxi
import urllib
import subprocess


## OK
def check_environment_wsl():
      '''
      check if the file is running into wsl
      '''
      try:
            with open('/proc/version', 'r') as file:
                  content = file.read().lower()
            if 'microsoft' in content or 'wsl' in content:
                  return True
            else:
                  return False
      except FileNotFoundError:
            return False

## OK
def find_best_model(datatype):
   
  if 'Condyle' in datatype.split(' '):
    print("model for condyles")

    # model_path = 'condyles_4classes.ckpt'
    model_path = '/Users/lumargot/Documents/Data/Condyles/epoch=137-val_loss=0.97.ckpt'
    nn = 'SaxiMHAFBClassification'

  elif 'Airway' in datatype.split(' '):
    print("model for airway")

    if args.task == 'binary':
      model_path='airways_2classes.ckpt'
    elif args.task == 'severity':
      model_path='airways_4classes.ckpt'
    elif args.task == 'regression':
      model_path='airways_regression.ckpt'
    else:
       print("no model found for undefined task")

    model_path='/Users/lumargot/Documents/Data/Airways/models/model_mha_fb.ckpt'
    nn = 'SaxiMHAFBClassification'

  elif 'Cleft' in datatype.split(' '):
    print("model for cleft")
    model_path='cleft_4classes.ckpt'
    model_path=None
    nn = 'FlyBy'

  else:
    print("No model found")
    return None
  return model_path, nn

def csv_edit(args, iswindows):
    """
    Check if the surfaces files are present in the input directory and edit csv file with surface path
    Args: Arguments from the command line
    """
    # surf_dir = linux2windows_path(args.input_dir, iswindows)
    surf_dir =args.input_dir
    for surf in os.listdir(surf_dir):
      surf_path = os.path.join(surf_dir, surf)
      if os.path.splitext(surf)[1] == '.vtk':
        if not os.path.exists(surf_path):
          print(f"Missing files: {surf}")
        else:
          with open(args.input_csv, 'a') as f:
              f.write(f"{surf},\n")

# def download_model(args, output_path):
#     json_path = linux2windows_path(os.path.join(os.path.dirname(__file__), "model_path.json"))
#     with open(json_path, 'r') as file:
#         model_info = json.load(file)
#     model_url = model_info["model"]["url"]
#     urllib.request.urlretrieve(model_url, output_path)

def run_prediction(args,out_model_path):
    # os.chdir(os.path.join(os.path.dirname(__file__), 'ShapeAXI'))
    # os.chdir('/Users/lumargot/Documents/ShapeAXI/')

    command = ['python -m shapeaxi.saxi_predict ', 
                    '--nn', args.nn, 
                    '--csv', args.input_csv, 
                    '--model', out_model_path, 
                    '--out', args.output_dir,
                    '--mount_point', args.input_dir]
    
    result = subprocess.run(command,stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print("Output : ",result.stdout)
    print("Error : ",result.stderr)

def linux2windows_path(filepath, iswindows):
  if iswindows :
    if ':' in filepath:
      drive, path_without_drive = filepath.split(':', 1)
      filepath = "/mnt/" + drive.lower() + path_without_drive
      return filepath
    # return filepath.replace("/","\\")
  else:
    return filepath

def create_csv(input_file):
  with open(input_file, "w") as f:
    f.write("path\n")
   
def main(args):

  ## check distribution
  iswindows = False
  if check_environment_wsl():
    iswindows = True

  # convert path if windows distribution
  args.input_csv = linux2windows_path(os.path.join(args.output_dir, "files.csv"), iswindows)
  args.input_dir = linux2windows_path(args.input_dir, iswindows)
  args.output_dir = linux2windows_path(args.output_dir, iswindows)
  
  ## TODO: Here problem when downloading model
  model_name, args.nn = find_best_model(args.data_type)

  out_model_path = linux2windows_path(os.path.join(args.output_dir, model_name), iswindows)
  if not os.path.exists(args.input_csv):
    create_csv(args.input_csv)
    csv_edit(args, iswindows)

  ## TODO: Here problem when running shapeaxi command
  run_prediction(args, out_model_path)


if __name__ == '__main__':
  print("_________________ CLI ________________")

  parser = argparse.ArgumentParser()
  parser.add_argument('input_dir',type = str)
  parser.add_argument('output_dir',type=str)
  parser.add_argument('data_type',type = str)
  parser.add_argument('task',type=str)

  args = parser.parse_args()
  main(args)