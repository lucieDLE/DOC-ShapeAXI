import os
import vtk, qt, slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin, pip_install
from enum import Enum
import subprocess
import platform

import webbrowser
import csv
import io
import sys
import time
import threading

sys.path.append('../../../ShapeAXI/')

from pathlib import Path
import re
#
# ShapeClassification
#


def func_import(install=False): 
  # try : 
  #   import shapeaxi 
  # except ImportError:
  #   pip_install('shapeaxi')

  try:
    import torch 
    pytorch3d = pkg_resources.get_distribution("pytorch3d").version 
  except:
    print("torch not found")
    try:
      pip_install('torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu')
      import torch
      print("torch installed")
    except:
      print("Unable to install torch")
    
    try:
      import pytorch3d
    except ImportError:
      try : 
        import torch
        pyt_version_str = torch.__version__.split("+")[0].replace(".", "")
        version_str = "".join([f"py3{sys.version_info.minor}_cu", torch.version.cuda.replace(".", ""), f"_pyt{pyt_version_str}"])
        pip_install('--upgrade pip')
        pip_install('fvcore==0.1.5.post20220305')
        pip_install(f'--no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/{version_str}/download.html')
      except:
        pip_install('--no-cache-dir torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113')
        pip_install('--no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu113_pyt1110/download.html')




class ShapeClassification(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "ShapeClassification"  # TODO: make this more human readable by adding spaces
    # TODO: set categories (folders where the module shows up in the module selector)
    self.parent.categories = ["Classification"]
    self.parent.dependencies = []  # TODO: add here list of module names that this module requires
    
    self.parent.contributors = ["Mathieu Leclercq (University of North Carolina)", 
    "Juan Carlos Prieto (University of North Carolina)",
    "Martin Styner (University of North Carolina)",
    "Lucia Cevidanes (University of Michigan)",
    "Beatriz Paniagua (Kitware)",
    "Connor Bowley (Kitware)",
    "Antonio Ruellas (University of Michigan)",
    "Marcela Gurgel (University of Michigan)",
    "Marilia Yatabe (University of Michigan)",
    "Jonas Bianchi (University of Michigan)"] 
    
    # TODO: update with short description of the module and a link to online module documentation
    self.parent.helpText = """
    This extension provides a GUI for a deep learning automated classification algorithm for Alveolar Bone Defect in Cleft, Nasopharynx Airway Obstruction and Mandibular Condyles. The inputs are vtk files, and are classified in 4 categories (0: Healthy, 1: Mild, 2: ? 3: Severe)

    - The input file must be a folder containing a list of vtk files. 
    You can find examples in the "Examples" folder. <br> <br> 

    - data type for classification:  <br><br> 

    - output directory: path to save output files (prediction and saliency maps)

    When prediction is over, you can open the output csv file which will containing the path of each .vtk file as well as the predicted class.
    <br><br>

    More help can be found on the <a href="https://github.com/DCBIA-OrthoLab/SlicerDentalModelSeg">Github repository</a> for the extension.
    """
    # TODO: replace with organization, grant and thanks
    self.parent.acknowledgementText = """
    This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
    and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
    """


      
class ShapeClassificationWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent=None) -> None:
    """Called when the user opens the module the first time and the widget is initialized."""
    ScriptedLoadableModuleWidget.__init__(self, parent)
    VTKObservationMixin.__init__(self)  # needed for parameter node observation

    self.logic = None
    self._parameterNode = None
    self._parameterNodeGuiTag = None
    self._updatingGUIFromParameterNode = False

    self.input = ""
    self.outputFolder = ""
    self.mount_point = ""
    self.data_type = ""

    self.log_path = os.path.join(slicer.util.tempDirectory(), 'process.log')
    self.time_log = 0 # for progress bar
    self.progress = 0
    self.currentPredDict = {}
    self.previous_time = 0
    self.start_time = 0

  def setup(self) -> None:
    self.removeObservers()

    """Called when the user opens the module the first time and the widget is initialized."""
    ScriptedLoadableModuleWidget.setup(self)

    # Load widget from .ui file (created by Qt Designer).
    # Additional widgets can be instantiated manually and added to self.layout. 
    uiWidget = slicer.util.loadUI(self.resourcePath("UI/ShapeClassification.ui"))
    self.layout.addWidget(uiWidget)
    self.ui = slicer.util.childWidgetVariables(uiWidget)

    # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
    # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
    # "setMRMLScene(vtkMRMLScene*)" slot.
    uiWidget.setMRMLScene(slicer.mrmlScene)

    # Create logic class. Logic implements all computations that should be possible to run
    # in batch mode, without a graphical user interface.
    self.logic = ShapeClassificationLogic()

    # Connections
    # These connections ensure that we update parameter node when scene is closed
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

    # UI elements 

    self.ui.browseDirectoryButton.connect('clicked(bool)',self.onBrowseOutputButton)
    self.ui.browseMountPointButton.connect('clicked(bool)',self.onBrowseMountPointButton)
    self.ui.cancelButton.connect('clicked(bool)', self.onCancel)

    self.ui.dataTypeComboBox.currentTextChanged.connect(self.onDataType)
    self.ui.resetButton.connect('clicked(bool)',self.onReset)

    self.ui.outputLineEdit.textChanged.connect(self.onEditOutputLine)
    self.ui.mountPointLineEdit.textChanged.connect(self.onEditMountPointLine)

    # initialize variables
    self.output = self.ui.outputLineEdit.text
    self.input_dir = self.ui.mountPointLineEdit.text
    self.data_type = self.ui.dataTypeComboBox.currentText


    # hidden buttons (installations & Github)
    self.ui.cancelButton.setHidden(True)
    self.ui.doneLabel.setHidden(True)
    self.ui.timeLabel.setHidden(True)
    self.ui.progressBar.setHidden(True)
    self.ui.progressLabel.setHidden(True)
    

    self.ui.applyChangesButton.connect('clicked(bool)',self.onApplyChangesButton)

    # Make sure parameter node is initialized (needed for module reload)
    self.initializeParameterNode()

  def cleanup(self) -> None:
    """Called when the application closes and the module widget is destroyed."""
    self.removeObservers()

  def enter(self) -> None:
    """Called each time the user opens this module."""
    # Make sure parameter node exists and observed
    self.initializeParameterNode()

  def exit(self) -> None:
    """Called each time the user opens a different module."""
    # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
    # if self._parameterNode:
    #   self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
    #   self._parameterNodeGuiTag = None
    self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

  def onSceneStartClose(self, caller, event) -> None:
    """Called just before the scene is closed."""
    # Parameter node will be reset, do not use it anymore
    self.setParameterNode(None)

  def onSceneEndClose(self, caller, event) -> None:
    """Called just after the scene is closed."""
    # If this module is shown while the scene is closed then recreate a new parameter node immediately
    if self.parent.isEntered:
      self.initializeParameterNode()

  def initializeParameterNode(self) -> None:
    """Ensure parameter node exists and observed."""
    # Parameter node stores all user choices in parameter values, node selections, etc.
    # so that when the scene is saved and reloaded, these settings are restored.

    self.setParameterNode(self.logic.getParameterNode())

  def setParameterNode(self, inputParameterNode) -> None:
    """
    Set and observe parameter node.
    Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
    """

    if self._parameterNode:
      self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
    self._parameterNode = inputParameterNode
    if self._parameterNode:
      self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

    # Initial GUI update
    self.updateGUIFromParameterNode()

  def updateGUIFromParameterNode(self, caller=None, event=None):
    """
    This method is called whenever parameter node is changed.
    The module GUI is updated to show the current state of the parameter node.
    """

    if self._parameterNode is None or self._updatingGUIFromParameterNode:
      return
    # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
    self._updatingGUIFromParameterNode = True

    # All the GUI updates are done
    self._updatingGUIFromParameterNode = False

  def updateParameterNodeFromGUI(self, caller=None, event=None):
    """
    This method is called when the user makes any change in the GUI.
    The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
    """
    if self._parameterNode is None or self._updatingGUIFromParameterNode:
      return
    wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch
    self._parameterNode.EndModify(wasModified)


  ## 
  ## Inputs
  ##

  def onBrowseMountPointButton(self):
    mount_point = qt.QFileDialog.getExistingDirectory(self.parent, "Select a folder containing vtk files")
    if mount_point != '':
      self.input_dir = mount_point
      self.ui.mountPointLineEdit.setText(self.input_dir)

  def onEditMountPointLine(self):
    self.input_dir = self.ui.mountPointLineEdit.text

  def onDataType(self):
    self.data_type = self.ui.dataTypeComboBox.currentText
  
  ##
  ## Output
  ##
    
  def onBrowseOutputButton(self):
    newoutputFolder = qt.QFileDialog.getExistingDirectory(self.parent, "Select a directory")
    if newoutputFolder != '':
      if newoutputFolder[-1] != "/":
        newoutputFolder += '/'
    self.outputFolder = newoutputFolder
    self.ui.outputLineEdit.setText(self.outputFolder)

  def onEditOutputLine(self):
    self.output = self.ui.outputLineEdit.text

  ##
  ##  Process
  ##


  
  def check_pythonpath_windows(self,name_env,file):
      '''
      Check if the environment env_name in wsl know the path to a specific file (ex : Crownsegmentationcli.py)
      return : bool
      '''
      conda_exe = self.conda_wsl.getCondaExecutable()
      command = [conda_exe, "run", "-n", name_env, "python" ,"-c", f"\"import {file} as check;import os; print(os.path.isfile(check.__file__))\""]
      result = self.conda_wsl.condaRunCommand(command)
      print("output CHECK python path: ", result)
      if "True" in result :
          return True
      return False
    
  def give_pythonpath_windows(self,name_env):
      '''
      take the pythonpath of Slicer and give it to the environment name_env in wsl.
      '''
      paths = slicer.app.moduleManager().factoryManager().searchPaths
      mnt_paths = []
      for path in paths :
          mnt_paths.append(f"\"{self.windows_to_linux_path(path)}\"")
      pythonpath_arg = 'PYTHONPATH=' + ':'.join(mnt_paths)
      conda_exe = self.conda_wsl.getCondaExecutable()
      # print("Conda_exe : ",conda_exe)
      argument = [conda_exe, 'env', 'config', 'vars', 'set', '-n', name_env, pythonpath_arg]
      results = self.conda_wsl.condaRunCommand(argument)
      print("output GIVE python path: ", results)


  def onApplyChangesButton(self):
    '''
    This function is called when the user want to run dentalmodelseg
    For Linux system : - check the installation of shapeaxi and pytorch3d in Slicer, if no with consent of user it install them
    - run crownsegmentationcli as a module of Slicer

    For Windows sytem : - check the installation of wsl, if no show a message asking asking for the user to do it and stop the process
    - check the installation of miniconda in wsl, if no show a message asking for the user to do it and sop the process
    - check if the environnement "shapeaxi" exist, if no it will create it (with consent of user) with the required librairies (shapeaxi and pytorch3d)
    - run the file CrownSegmentationcli.py into wsl in the environment 'shapeaxi'
    '''

    self.ui.applyChangesButton.setEnabled(False)

    msg = qt.QMessageBox()
    if not(os.path.isdir(self.output)):
      if not(os.path.isdir(self.output)):
        msg.setText("Output directory : \nIncorrect path.")
        print('Error: Incorrect path for output directory.')
        self.ui.outputLineEdit.setText('')
        print(f'output folder : {self.output}')
      else:
        msg.setText('Unknown error.')

      msg.setWindowTitle("Error")
      msg.exec_()
      return

    elif not(os.path.isdir(self.input_dir)):
      msg.setText("input file : \nIncorrect path.")
      print('Error: Incorrect path for input directory.')
      self.ui.mountPointLineEdit.setText('')

      msg.setWindowTitle("Error")
      msg.exec_()
      return

    else:
        
      self.ui.timeLabel.setHidden(False)
      if platform.system() != "Windows" : #if linux system
        print("Linux")
        env_ok = func_import(False)
        if not env_ok : 
          userResponse = slicer.util.confirmYesNoDisplay("Some of the required libraries are not installed in Slicer. Would you like to install them?\nThis may take a few minutes.", windowTitle="Env doesn't exist")
          if userResponse : 
            self.parall_process(func_import,[True],"Installing the required packages in Slicer")
            env_ok = True
        if env_ok : 
          slicer_path = slicer.app.applicationDirPath()


        if 'Airway' in self.data_type.split(' '):
          for model_type in ['binary', 'severity', 'regression']:
            self.logic = ShapeClassificationLogic(input_dir=self.input_dir, output_dir=self.output, data_type=self.data_type, task=model_type)                
            self.logic.process()
            self.addObserver(self.logic.cliNode,vtk.vtkCommand.ModifiedEvent,self.onProcessUpdate)
            self.onProcessStarted()
            file_path = os.path.abspath(__file__)
            folder_path = os.path.dirname(file_path)
            self.ui.applyChangesButton.setEnabled(True)

        else:
          self.logic = ShapeClassificationLogic(input_dir=self.input_dir, output_dir=self.output, data_type=self.data_type, task='severity')                
          self.logic.process()
          self.addObserver(self.logic.cliNode,vtk.vtkCommand.ModifiedEvent,self.onProcessUpdate)
          self.onProcessStarted()
          file_path = os.path.abspath(__file__)
          folder_path = os.path.dirname(file_path)

          self.ui.applyChangesButton.setEnabled(True)

      ### TODO: Windows test and changes
      else:
        from CondaSetUp import  CondaSetUpCall,CondaSetUpCallWsl
        print("windows!!")

        self.conda_wsl = CondaSetUpCallWsl()  
        wsl = self.conda_wsl.testWslAvailable()
        ready = True
        self.ui.timeLabel.setHidden(False)
        self.ui.timeLabel.setText(f"Checking if wsl is installed, this task may take a moments")
        slicer.app.processEvents()

        if wsl : # if wsl is install
          self.ui.timeLabel.setText("WSL installed")
          lib = self.check_lib_wsl()
          if not lib : # if lib required are not install
            self.ui.timeLabel.setText(f"Checking if the required librairies are installed, this task may take a moments")
            messageBox = qt.QMessageBox()
            text = "Code can't be launch. \nWSL doen't have all the necessary libraries, please download the installer and follow the instructin here : https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/wsl2_windows/installer_wsl2.zip\nDownloading may be blocked by Chrome, this is normal, just authorize it."
            ready = False
            messageBox.information(None, "Information", text)
        else : # if wsl not install, ask user to install it ans stop process
          messageBox = qt.QMessageBox()
          text = "Code can't be launch. \nWSL is not installed, please download the installer and follow the instructin here : https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/wsl2_windows/installer_wsl2.zip\nDownloading may be blocked by Chrome, this is normal, just authorize it."
          ready = False
          messageBox.information(None, "Information", text)
        if ready : # checking if miniconda installed on wsl
          self.ui.timeLabel.setText(f"Checking if miniconda is installed")
          if "Error" in self.conda_wsl.condaRunCommand([self.conda_wsl.getCondaExecutable(),"--version"]): # if conda is setup
              messageBox = qt.QMessageBox()
              text = "Code can't be launch. \nConda is not setup in WSL. Please go the extension CondaSetUp in SlicerConda to do it."
              ready = False
              messageBox.information(None, "Information", text)       

        if ready : # checking if environment 'shapeaxi' exist on wsl and if no ask user permission to create and install required lib in it
          self.ui.timeLabel.setText(f"Checking if environnement exist")
          if not self.conda_wsl.condaTestEnv('shapeaxi') : # check is environnement exist, if not ask user the permission to do it
            userResponse = slicer.util.confirmYesNoDisplay("The environnement to run the classification doesn't exist, do you want to create it ? ", windowTitle="Env doesn't exist")
            if userResponse :
              start_time = time.time()
              previous_time = start_time
              self.ui.timeLabel.setText(f"Creation of the new environment. This task may take a few minutes.\ntime: 0.0s")
              name_env = "shapeaxi"
              process = threading.Thread(target=self.conda_wsl.condaCreateEnv, args=(name_env,"3.9",["shapeaxi"],)) #run in paralle to not block slicer
              process.start()
              
              while process.is_alive():
                slicer.app.processEvents()
                current_time = time.time()
                gap=current_time-previous_time
                if gap>0.3:
                  previous_time = current_time
                  elapsed_time = current_time - start_time
                  self.ui.timeLabel.setText(f"Creation of the new environment. This task may take a few minutes.\ntime: {elapsed_time:.1f}s")
          
              start_time = time.time()
              previous_time = start_time
              self.ui.timeLabel.setText(f"Installation of librairies into the new environnement. This task may take a few minutes.\ntime: 0.0s")

              name_env = "shapeaxi"
              result_pythonpath = self.check_pythonpath_windows(name_env,"CrownSegmentation_utils.install_pytorch")
              if not result_pythonpath : 
                self.give_pythonpath_windows(name_env)
                # result_pythonpath = self.check_pythonpath_windows(name_env,"ALI_IOS_utils.requirement") # THIS LINE IS WORKING
                result_pythonpath = self.check_pythonpath_windows(name_env,"CrownSegmentation_utils.install_pytorch")
                
              if result_pythonpath : 
                conda_exe = self.conda_wsl.getCondaExecutable()
                path_pip = self.conda_wsl.getCondaPath()+f"/envs/{name_env}/bin/pip"
                # command = [conda_exe, "run", "-n", name_env, "python" ,"-m", f"ALI_IOS_utils.requirement",path_pip] # THIS LINE IS WORKING
                command = [conda_exe, "run", "-n", name_env, "python" ,"-m", f"CrownSegmentation_utils.install_pytorch",path_pip]
                print("command : ",command)
              
                process = threading.Thread(target=self.conda_wsl.condaRunCommand, args=(command,)) # launch install_pythorch.py with the environnement ali_ios to install pytorch3d on it
                process.start()
              # file_path = os.path.realpath(__file__)
              # folder = os.path.dirname(file_path)
              # utils_folder = os.path.join(folder, "utils")
              # utils_folder_norm = os.path.normpath(utils_folder)
              # install_path = self.windows_to_linux_path(os.path.join(utils_folder_norm, 'install_pytorch.py'))
              # path_pip = self.conda_wsl.getCondaPath()+"/envs/shapeaxi/bin/pip"
              # process = threading.Thread(target=self.conda_wsl.condaRunFilePython, args=(install_path,name_env,[path_pip],)) # launch install_pythorch.py with the environnement ali_ios to install pytorch3d on it
              # process.start()
              
              while process.is_alive():
                slicer.app.processEvents()
                current_time = time.time()
                gap=current_time-previous_time
                if gap>0.3:
                  previous_time = current_time
                  elapsed_time = current_time - start_time
                  self.ui.timeLabel.setText(f"Installation of librairies into the new environnement. This task may take a few minutes.\ntime: {elapsed_time:.1f}s")
              
              ready=True
            else :
              ready = False
          else:
            print("shapeaxi already exists!")
        if ready : # if everything is ready launch dentalmodelseg on the environnement shapeaxi in wsl
          name_env = "shapeaxi"

          result_pythonpath = self.check_pythonpath_windows(name_env,"ShapeClassificationcli")
          if not result_pythonpath : 
            self.give_pythonpath_windows(name_env)
            result_pythonpath = self.check_pythonpath_windows(name_env,"ShapeClassificationcli")
          
          if 'Airway' in self.data_type.split(' '):
            for model_type in ['binary', 'severity', 'regression']:
              args = [self.input_dir, self.output, self.data_type, model_type]

              conda_exe = self.conda_wsl.getCondaExecutable()
              command = [conda_exe, "run", "-n", name_env, "python" ,"-m", f"ShapeClassificationcli"]
              for arg in args :
                    command.append("\""+arg+"\"")
              # print("The following command will be executed:\n",command)

              print()
              print()
              print('----------------------------------------')
              print("results CLI command :\n")

              # running in // to not block Slicer
              process = threading.Thread(target=self.conda_wsl.condaRunCommand, args=(command,))

              process.start()
              self.ui.applyChangesButton.setEnabled(False)
              self.ui.doneLabel.setHidden(True)
              self.ui.timeLabel.setHidden(False)
              self.ui.progressLabel.setHidden(False)
              self.ui.progressBar.setHidden(False)
              self.ui.timeLabel.setText(f"time : 0.00s")
              start_time = time.time()
              previous_time = start_time
              while process.is_alive():
                slicer.app.processEvents()
                current_time = time.time()
                gap=current_time-previous_time
                if gap>0.3:
                  previous_time = current_time
                  elapsed_time = current_time - start_time
                  self.ui.timeLabel.setText(f"time : {elapsed_time:.2f}s")
          else:
              args = [self.input_dir, self.output, self.data_type, 'severity']
              conda_exe = self.conda_wsl.getCondaExecutable()
              command = [conda_exe, "run", "-n", name_env, "python" ,"-m", f"ShapeClassificationcli"]
              for arg in args :
                    command.append("\""+arg+"\"")
              print("The following command will be executed:\n",command)


              # running in // to not block Slicer
              process = threading.Thread(target=self.conda_wsl.condaRunCommand, args=(command,))
              process.start()
              self.ui.applyChangesButton.setEnabled(False)
              self.ui.doneLabel.setHidden(True)
              self.ui.timeLabel.setHidden(False)
              self.ui.progressLabel.setHidden(False)
              self.ui.timeLabel.setText(f"time : 0.00s")
              start_time = time.time()
              previous_time = start_time
              while process.is_alive():
                slicer.app.processEvents()
                current_time = time.time()
                gap=current_time-previous_time
                if gap>0.3:
                  previous_time = current_time
                  elapsed_time = current_time - start_time
                  self.ui.timeLabel.setText(f"time : {elapsed_time:.2f}s")

          self.ui.progressLabel.setHidden(True)
          self.ui.doneLabel.setHidden(False)
          self.ui.applyChangesButton.setEnabled(True)

          # Delete csv file
          file_path = os.path.abspath(__file__)
          folder_path = os.path.dirname(file_path)
          csv_file = os.path.join(folder_path,"list_file.csv")
          if os.path.exists(csv_file):
            os.remove(csv_file)
            
      self.ui.applyChangesButton.setEnabled(True)
  
  def parall_process(self,function,arguments=[],message=""):
    process = threading.Thread(target=function, args=tuple(arguments)) #run in paralle to not block slicer
    process.start()
    start_time = time.time()
    previous_time = time.time()
    while process.is_alive():
      slicer.app.processEvents()
      current_time = time.time()
      gap=current_time-previous_time
      if gap>0.3:
        previous_time = current_time
        elapsed_time = current_time - start_time
        self.ui.timeLabel.setText(f"{message}\ntime: {elapsed_time:.1f}s")



  def onProcessStarted(self):
    self.start_time = time.time()
    self.previous_time = self.start_time  
    
    self.ui.applyChangesButton.setEnabled(False)
    self.ui.doneLabel.setHidden(True)
    self.ui.timeLabel.setHidden(False)
    self.ui.progressLabel.setHidden(False)
    self.ui.progressBar.setHidden(False)
    self.ui.timeLabel.setText(f"time : 0.00s") 


  def onProcessUpdate(self,caller,event):
    # check log file
    current_time = time.time()
    gap = current_time - self.previous_time
    if gap > 0.3:
      self.previous_time = current_time
      elapsed_time = current_time - self.start_time
      self.ui.timeLabel.setText(f"prediction in process\ntime : {elapsed_time:.2f}s")
    

    if self.logic.cliNode.GetStatus() & self.logic.cliNode.Completed:
    # process complete
      self.ui.applyChangesButton.setEnabled(True)
      self.ui.resetButton.setEnabled(True)
      self.ui.progressLabel.setHidden(False)     
      self.ui.cancelButton.setEnabled(False)
      self.ui.progressBar.setEnabled(False)
      self.ui.progressBar.setHidden(True)
      self.ui.progressLabel.setHidden(True)

      if self.logic.cliNode.GetStatus() & self.logic.cliNode.ErrorsMask:
        # error
        errorText = self.logic.cliNode.GetErrorText()
        print("CLI execution failed: \n \n" + errorText)
        msg = qt.QMessageBox()
        msg.setText(f'There was an error during the process:\n \n {errorText} ')
        msg.setWindowTitle("Error")
        msg.exec_()

      else:
        # success
        print('PROCESS DONE.')

        self.ui.progressLabel.setHidden(True)
        self.ui.doneLabel.setHidden(False)
        self.ui.applyChangesButton.setEnabled(True)
        print("Process completed successfully.")
        # self.ui.timeLabel.setText(f"time : {elapsed_time:.2f}s")
        
        print("*"*25,"Output cli","*"*25)
        print(self.logic.cliNode.GetOutputText())
        
        file_path = os.path.abspath(__file__)
        folder_path = os.path.dirname(file_path)
        csv_file = os.path.join(folder_path,"list_file.csv")
        if os.path.exists(csv_file):
          os.remove(csv_file)
      
  def onReset(self):
    self.ui.outputLineEdit.setText("")
    self.ui.mountPointLineEdit.setText("")

    self.ui.applyChangesButton.setEnabled(True)
    self.ui.progressLabel.setHidden(True)
    self.ui.progressBar.setValue(0)
    self.ui.doneLabel.setHidden(True)
    self.ui.timeLabel.setHidden(True)

    self.removeObservers()  

  def onCancel(self):
    self.logic.cliNode.Cancel()
    self.ui.applyChangesButton.setEnabled(True)
    self.ui.resetButton.setEnabled(True)
    self.ui.progressBar.setEnabled(False)
    self.ui.progressBar.setRange(0,100)
    self.ui.progressLabel.setHidden(True)
    self.ui.cancelButton.setEnabled(False)
    self.removeObservers()  
    print("Process successfully cancelled.")


  def windows_to_linux_path(self,windows_path):
    '''
    Convert a windows path to a wsl path
    '''
    windows_path = windows_path.strip()

    path = windows_path.replace('\\', '/')

    if ':' in path:
      drive, path_without_drive = path.split(':', 1)
      path = "/mnt/" + drive.lower() + path_without_drive

    return path
        
  def check_lib_wsl(self)->bool:
    '''
    Check if wsl contains the require librairies
    '''
    result1 = subprocess.run("wsl -- bash -c \"dpkg -l | grep libxrender1\"", capture_output=True, text=True)
    output1 = result1.stdout.encode('utf-16-le').decode('utf-8')
    clean_output1 = output1.replace('\x00', '')

    result2 = subprocess.run("wsl -- bash -c \"dpkg -l | grep libgl1-mesa-glx\"", capture_output=True, text=True)
    output2 = result2.stdout.encode('utf-16-le').decode('utf-8')
    clean_output2 = output2.replace('\x00', '')

    return "libxrender1" in clean_output1 and "libgl1-mesa-glx" in clean_output2


def is_ubuntu_installed(self)->bool:
    '''
    Check if wsl is install with Ubuntu
    '''
    result = subprocess.run(['wsl', '--list'], capture_output=True, text=True)
    output = result.stdout.encode('utf-16-le').decode('utf-8')
    clean_output = output.replace('\x00', '')  # Enlève tous les octets null

    return 'Ubuntu' in clean_output



#
# ShapeClassificationLogic
#


class ShapeClassificationLogic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py

    parser = argparse.ArgumentParser()
    parser.add_argument('input_csv',type = str, help='input csv file containing .vtk files')
    parser.add_argument('model',type=str, help="path to model")
    parser.add_argument('surf_column',type=str, "surface column name in csv file")
    parser.add_argument('mount_point',type=str)
    parser.add_argument('nn',type=str, help="type of neural netowrk")
    parser.add_argument('out',type=str, help="output directory")

  """

  def __init__(self, input_dir = "None", output_dir="None", data_type="None", task='severity'):
    
    """Called when the logic class is instantiated. Can be used for initializing member variables."""
    ScriptedLoadableModuleLogic.__init__(self)

    self.output_dir = output_dir
    self.input_dir = input_dir
    self.data_type = data_type
    self.task = task

  def process(self):
    """
    Run the classification algorithm.
    Can be used without GUI widget.
    """
    
    parameters = {}

    parameters ["input_dir"] = self.input_dir
    parameters ["output_dir"] = self.output_dir
    parameters ['data_type'] = self.data_type
    parameters ['task'] = self.task

    shapeaxi_process = slicer.modules.shapeclassificationcli
    self.cliNode = slicer.cli.run(shapeaxi_process,None, parameters)  
    return shapeaxi_process