#!/usr/bin/env python-real
import json
import os
import argparse
from urllib import request
import subprocess
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm

import shapeaxi
from shapeaxi.saxi_dataset import SaxiDataset
from shapeaxi.saxi_transforms import TrainTransform, EvalTransform

from shapeaxi.saxi_gradcam import gradcam_process 


from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import vtk

from shapeaxi import saxi_nets, post_process as psp, utils


def gradcam_save(args, gradcam_path, surf_path, surf):
    '''
    Function to save the GradCAM on the surface

    Args : 
        gradcam_path : path to save the GradCAM
        surf_path : path to the surface
        surf : surface read by utils.ReadSurf
    '''

    if not os.path.exists(gradcam_path):
        os.makedirs(gradcam_path)
    
    out_surf_path = os.path.join(gradcam_path, os.path.basename(surf_path))

    subprocess.call(["cp", surf_path, out_surf_path])

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(out_surf_path)
    writer.SetInputData(surf)
    writer.Write()

class MultiHead(nn.Module):
    def __init__(self, mha_fb):
        super().__init__()
        self.mha_fb = mha_fb

    def forward(self, x):
        x, score = self.mha_fb(x,x,x)
        return x

class SelfAttention(nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.attn = attn

    def forward(self, x):
        x, score = self.attn(x,x)
        return x


# def find_best_model(datatype):
   
#   if 'Condyle' in datatype.split(' '):
#     model_name='condyles_4_class'
#     nn = 'SaxiMHAFBClassification'
#     args.num_classes = 4

#   elif 'Airway' in datatype.split(' '):
#     if args.task == 'binary':
#       model_name='airways_2_class'
#       nn = 'SaxiMHAFBClassification'
#       args.num_classes = 2

#     elif args.task == 'severity':
#       model_name='airways_4_class'
#       nn = 'SaxiMHAFBClassification'
#       args.num_classes = 4

#     elif args.task == 'regression':
#       model_name='airways_4_regress'
#       nn = 'SaxiMHAFBRegression'
#       args.num_classes = 1
#     else:
#        print("no model found for undefined task")

#   elif 'Cleft' in datatype.split(' '):
#     model_name='clefts_4_class'
#     nn = 'SaxiMHAFBClassification'
#     args.num_classes = 4

#   else:
#     print("No model found")
#     return None, None
#   return model_name, nn

def csv_edit(args):
    """
    Check if the surfaces files are present in the input directory and edit csv file with surface path
    Args: Arguments from the command line
    """
    surf_dir =args.input_dir
    for surf in os.listdir(surf_dir):
      surf_path = os.path.join(surf_dir, surf)
      if os.path.splitext(surf)[1] == '.vtk':
        if not os.path.exists(surf_path):
          print(f"Missing files: {surf}")
        else:
          with open(args.input_csv, 'a') as f:
              f.write(f"{surf}\n")

def download_model(model_name, output_path):
    json_path = os.path.join(os.path.dirname(__file__), "model_path.json")
    with open(json_path, 'r') as file:
        model_info = json.load(file)
    model_url = model_info[model_name]["url"]
    request.urlretrieve(model_url, output_path)


def saxi_gradcam(args, out_model_path):
  print("Running Explainability....")
  with open(args.log_path,'w+') as log_f :
    log_f.write(f"{args.task},explainability,NaN,{args.num_classes}")

  NN = getattr(saxi_nets, args.nn)    
  model = NN.load_from_checkpoint(out_model_path, strict=False)
  model.ico_sphere(radius=model.hparams.radius, subdivision_level=model.hparams.subdivision_level)

  model.eval()
  model.to(args.device)

  fname = os.path.basename(args.input_csv)
  predicted_csv = os.path.join(args.output_dir, fname.replace('.csv', "_prediction.csv"))
  df = pd.read_csv(predicted_csv)


  model_cam_mv = nn.Sequential(
      model.convnet,
      model.ff_fb,
      MultiHead(model.mha_fb),
      SelfAttention(model.attn_fb),
      nn.Linear(model.hparams.output_dim, args.num_classes)
      )

  model_cam_mv.to(args.device)
    
  test_ds = SaxiDataset(df, transform=EvalTransform(), CN=True, 
                        surf_column=model.hparams.surf_column, mount_point = args.input_dir, 
                        class_column=None, scalar_column=model.hparams.scalar_column, **vars(args))
  
  test_loader = DataLoader(test_ds, batch_size=1, pin_memory=False)

  target_layer = getattr(model_cam_mv[0].module, '_blocks')
  target_layers = None 

  if isinstance(target_layer, nn.Sequential):
      target_layer = target_layer[-1]
      target_layers = [target_layer]

  cam = GradCAM(model=model_cam_mv, target_layers=target_layers)

  targets = None
  out_dir = os.path.join(args.output_dir, "explainability", args.task)
  
  for idx, (V, F, CN) in tqdm(enumerate(test_loader), total=len(test_loader)):
    # The generated CAM is processed and added to the input surface mesh (surf) as a point data array
    V = V.to(args.device)
    F = F.to(args.device)
    CN = CN.to(args.device)
    
    X_mesh = model.create_mesh(V, F, CN)
    X_views, PF = model.render(X_mesh)

    surf_path = os.path.join(args.input_dir, df.loc[idx]['surf'])
    surf = test_ds.getSurf(idx)

    for class_idx in range(args.num_classes):
      if args.nn == 'SaxiMHAFBRegression':
        args.target_class = None
      else:
        args.target_class = class_idx
        targets = [ClassifierOutputTarget(args.target_class)]

      gcam_np = cam(input_tensor=X_views, targets=targets)

      Vcam = gradcam_process(args, gcam_np, F, PF, V, device='cuda')

      surf.GetPointData().AddArray(Vcam)
      psp.MedianFilter(surf, Vcam)

    gradcam_save(args, out_dir, surf_path, surf)
    
    with open(args.log_path,'w+') as log_f :
      log_f.write(f"{args.task},explainability,{idx},{args.num_classes}")


def saxi_predict(args,out_model_path):
    print("Running Prediction....")

    df = pd.read_csv(args.input_csv)
    with open(args.log_path,'w+') as log_f :
      log_f.write(f"{args.task},predict,NaN,{args.num_classes}")


    NN = getattr(saxi_nets, args.nn)    
    model = NN.load_from_checkpoint(out_model_path, strict=False)
    model.eval()
    model.to(args.device)

    scale_factor = None
    if hasattr(model.hparams, 'scale_factor'):
        scale_factor = model.hparams.scale_factor
    
    test_ds = SaxiDataset(df, transform=EvalTransform(scale_factor), CN=True, 
                          surf_column=model.hparams.surf_column, mount_point = args.input_dir, 
                          class_column=None, scalar_column=model.hparams.scalar_column, **vars(args))
    
    test_loader = DataLoader(test_ds, batch_size=1, pin_memory=False)

    fname = os.path.basename(args.input_csv)

    with torch.no_grad():
      predictions = []
      softmax = nn.Softmax(dim=1)

      for idx, (V, F, CN) in tqdm(enumerate(test_loader), total=len(test_loader)):
        V = V.to(args.device)
        F = F.to(args.device)
        CN = CN.to(args.device)
        
        X_mesh = model.create_mesh(V, F, CN)
        x, x_w, X = model(X_mesh)
        
        if args.nn == 'SaxiMHAFBClassification': # no argmax for regression
          x = softmax(x).detach()
          x = torch.argmax(x, dim=1, keepdim=True)
        predictions.append(x)

        with open(args.log_path,'w+') as log_f :
          log_f.write(f"{args.task},predict,{idx+1},{args.num_classes}")


      predictions = torch.cat(predictions).cpu().numpy().squeeze()

      out_name = os.path.join(args.output_dir, fname.replace(".csv", "_prediction.csv"))
      if os.path.exists(out_name):
        df = pd.read_csv(out_name)

      df[f'{args.task}_prediction'] = predictions
      df.to_csv(out_name, index=False)

def linux2windows_path(filepath):
  if ':' in filepath:
    if '\\' in filepath:
      filepath = filepath.replace('\\', '/')
    drive, path_without_drive = filepath.split(':', 1)
    filepath = "/mnt/" + drive.lower() + path_without_drive
    return filepath
  else:
    return filepath

def create_csv(input_file):
  with open(input_file, "w") as f:
    f.write("surf\n")
   
def main(args):
  import torch

  args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # model_name, args.nn = find_best_model(args.data_type)
  # convert path if windows distribution
  args.input_csv = linux2windows_path(os.path.join(args.output_dir, f"files_{args.data_type}.csv"))
  args.input_dir = linux2windows_path(args.input_dir)
  args.output_dir = linux2windows_path(args.output_dir)
  args.log_path = linux2windows_path(args.log_path)

  with open(args.log_path,'w') as log_f:
    # clear log file
    log_f.truncate(0)
  

  out_model_path = os.path.join(args.output_dir, args.model + '.ckpt')
  if os.path.exists(args.output_dir):
    if not os.path.exists(out_model_path):
      print("Downloading model...")
      download_model(args.model, out_model_path)

  if not os.path.exists(args.input_csv):
    create_csv(args.input_csv)
    csv_edit(args)

  saxi_predict(args, out_model_path)
  print("End prediction, starting explainability")

  saxi_gradcam(args, out_model_path)

  print("End explainability \nProcess Completed")


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('input_dir',type = str)
  parser.add_argument('output_dir',type=str)
  parser.add_argument('data_type',type = str)
  parser.add_argument('task', type=str)
  parser.add_argument('model',type=str)
  parser.add_argument('nn',type=str)
  parser.add_argument('num_classes',type=int)
  parser.add_argument('log_path',type=str)

  args = parser.parse_args()

  # if  'Airway' in args.data_type.split(' '):
  #   tasks = ['severity', 'binary', 'regression']
  #   for task in tasks:
  #     args.task = task
  #     main(args)

  # else:
  #   args.task = 'severity'
  main(args)

  with open(linux2windows_path(args.log_path),'w+') as log_f :
    log_f.write(f"Complete,NaN,NaN,NaN")