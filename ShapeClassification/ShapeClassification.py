import os
import vtk, qt, slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin, pip_install
import subprocess
import platform
import sys
import time
import threading

sys.path.append('../../../ShapeAXI/')

from pathlib import Path
import re
#
# ShapeClassification
#



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

    self.log_path = os.path.normpath(os.path.join(slicer.util.tempDirectory(), 'process.log'))
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

    self.ui.cancelButton.setHidden(True)
    self.ui.doneLabel.setHidden(True)

    
    # progress bar 
    self.log_path = os.path.normpath(os.path.join(os.path.dirname(__file__), 'process.log'))
    
    if '\\' in self.log_path:
      self.log_path = self.log_path.replace('\\', '/')
    
    with open(self.log_path, mode='w') as f: pass

    self.time_log = 0
    self.cliNode = None
    self.installCliNode = None
    self.progress = 0
    self.cancel = False

    self.ui.timeLabel.setVisible(False)
    self.ui.labelBar.setVisible(False)
    self.ui.labelBar.setStyleSheet(f"""QLabel{{font-size: 12px; qproperty-alignment: AlignCenter;}}""")
    self.ui.progressLabel.setVisible(False)
    self.ui.progressLabel.setStyleSheet(f"""QLabel{{font-size: 16px; qproperty-alignment: AlignCenter;}}""")
    self.ui.progressBar.setVisible(False)
    self.ui.progressBar.setRange(0,100)
    self.ui.progressBar.setTextVisible(True)

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
      conda_exe = self.conda.getCondaExecutable()
      command = [conda_exe, "run", "-n", name_env, "python" ,"-c", f"\"import {file} as check;import os; print(os.path.isfile(check.__file__))\""]
      result = self.conda.condaRunCommand(command)
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
      conda_exe = self.conda.getCondaExecutable()
      # print("Conda_exe : ",conda_exe)
      argument = [conda_exe, 'env', 'config', 'vars', 'set', '-n', name_env, pythonpath_arg]
      results = self.conda.condaRunCommand(argument)
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
      # -------------------------------- STARTING ENV SET UP:  -----------------

      self.ui.timeLabel.setHidden(False)
      ready = True
      self.ui.timeLabel.setHidden(False)
      slicer.app.processEvents()
      
      # -------------------------------- import SlicerConda envs + check wsl if windows:  -----------------
      if platform.system() != "Windows" : #if linux system
        from CondaSetUp import CondaSetUpCall
        print("Linux")
        self.conda = CondaSetUpCall()
      else:
        from CondaSetUp import CondaSetUpCallWsl
        print("windows!!")
        self.conda = CondaSetUpCallWsl()  
        wsl = self.conda.testWslAvailable()
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
      
      # -------------------------------- check Miniconda installation -----------------

      if ready : # checking if miniconda installed
        self.ui.timeLabel.setText(f"Checking if miniconda is installed")
        if "Error" in self.conda.condaRunCommand([self.conda.getCondaExecutable(),"--version"]): # if conda is setup
          messageBox = qt.QMessageBox()
          text = "Code can't be launch. \nConda is not setup. Please go the extension CondaSetUp in SlicerConda to do it."
          ready = False
          messageBox.information(None, "Information", text)

      # ------------------------------ check if ShapeAXI exist in environment 


      if ready : # checking if environment 'shapeaxi' exist and if no ask user permission to create and install required lib in it
        self.ui.timeLabel.setText(f"Checking if environnement exist")
        if not self.conda.condaTestEnv('shapeaxi') : # check is environnement exist, if not ask user the permission to do it
          userResponse = slicer.util.confirmYesNoDisplay("The environnement to run the classification doesn't exist, do you want to create it ? ", windowTitle="Env doesn't exist")
          if userResponse :
            start_time = time.time()
            previous_time = start_time
            self.ui.timeLabel.setText(f"Creation of the new environment. This task may take a few minutes.\ntime: 0.0s")
            name_env = "shapeaxi"
            process = threading.Thread(target=self.conda.condaCreateEnv, args=(name_env,"3.9",["shapeaxi"],)) #run in paralle to not block slicer
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
            # ------------ installation of pytorch3d - differents but can be one way I think
            if platform.system() == "Windows": 
              print("installation of pytorch in windows")
              # sys.path.append("..\\CrownSegmentation\\CrownSegmentation_utils")

              result_pythonpath = self.check_pythonpath_windows(name_env,"CrownSegmentation_utils.install_pytorch")
              if not result_pythonpath : 
                self.give_pythonpath_windows(name_env)
                result_pythonpath = self.check_pythonpath_windows(name_env,"CrownSegmentation_utils.install_pytorch")
                
              if result_pythonpath : 
                conda_exe = self.conda.getCondaExecutable()
                path_pip = self.conda.getCondaPath()+f"/envs/{name_env}/bin/pip"
                command = [conda_exe, "run", "-n", name_env, "python" ,"-m", f"CrownSegmentation_utils.install_pytorch",path_pip]
                print("command : ",command)				
            else: ## I think the ../../CrownSegmentation_utils should wotk also on windows because executed by wsl in anycase
              print("installation of pytorch on linux system")
              sys.path.append("../CrownSegmentation/CrownSegmentation_utils")
          
              conda_exe = self.conda.getCondaExecutable()
              path_pip = self.conda.getCondaPath()+f"/envs/{name_env}/bin/pip"
              command = [conda_exe, "run", "-n", name_env, "python" ,"-m", f"CrownSegmentation_utils.install_pytorch",path_pip]

              # ----- then run command and update process in UI
              # results = self.conda.condaRunCommand(command)
              # print(results)
            process = threading.Thread(target=self.conda.condaRunCommand, args=(command,)) # launch install_pythorch.py with the environnement ali_ios to install pytorch3d on it
            process.start()
            
            ## create an update_timeLabelText function with text to display as input parameter
            while process.is_alive():
              slicer.app.processEvents()
              current_time = time.time()
              gap=current_time-previous_time
              if gap>0.3:
                previous_time = current_time
                elapsed_time = current_time - start_time
                self.ui.timeLabel.setText(f"Installation of pytorch into the new environnement. This task may take a few minutes.\ntime: {elapsed_time:.1f}s")							
            ready=True
          else :
            ready = False
      else:
        ready=True
        print("shapeaxi already exists!")
      name_env='shapeaxi'

      #################### DONE shape axi installation

      # ------------------------------ Access cli in environment 
          
      if ready : # if everything is ready launch script on the environnement shapeaxi
        name_env = "shapeaxi"

        #### here differents path but should be able to specify the same way for linux and windows
        if platform.system() == "Windows": 
          print("import cli in windows")
          result_pythonpath = self.check_pythonpath_windows(name_env,"ShapeClassificationcli")
          if not result_pythonpath : 
            self.give_pythonpath_windows(name_env)
            result_pythonpath = self.check_pythonpath_windows(name_env,"ShapeClassificationcli")
        else: 
          print("import cli in linux")
          sys.path.append("../ShapeClassificationcli")
      

        # ------------------------------ Create args list and send command to cli script 

        if 'Airway' in self.data_type.split(' '):
          for self.task in ['binary']:
            if not self.cancel :
              args = [self.input_dir, self.output, self.data_type, self.task, self.log_path]

              conda_exe = self.conda.getCondaExecutable()
              command = [conda_exe, "run", "-n", name_env, "python" ,"-m", f"ShapeClassificationcli"]
              for arg in args :
                    command.append("\""+arg+"\"")

              # running in // to not block Slicer
              self.process = threading.Thread(target=self.conda.condaRunCommand, args=(command,))

              self.process.start()
              self.onProcessStarted()
              self.ui.labelBar.setText(f'Loading {self.task} model...')

              self.ui.applyChangesButton.setEnabled(False)
              self.ui.doneLabel.setHidden(True)
              self.ui.timeLabel.setHidden(False)
              self.ui.progressLabel.setHidden(False)
              self.ui.progressBar.setHidden(False)
              start_time = time.time()
              previous_time = start_time
              while self.process.is_alive():
                slicer.app.processEvents()
                self.onProcessUpdate()
                current_time = time.time()
                gap=current_time-previous_time
                if gap>0.3:
                  previous_time = current_time
                  elapsed_time = current_time - start_time
                  self.ui.timeLabel.setText(f"time : {elapsed_time:.2f}s")
              self.resetProgressBar()
          self.onProcessCompleted()

        else:
            self.task = 'severity'
            args = [self.input_dir, self.output, self.data_type, self.task]
            conda_exe = self.conda.getCondaExecutable()
            command = [conda_exe, "run", "-n", name_env, "python" ,"-m", f"ShapeClassificationcli"]
            for arg in args :
                  command.append("\""+arg+"\"")
            print("The following command will be executed:\n",command)


            # running in // to not block Slicer
            self.process = threading.Thread(target=self.conda.condaRunCommand, args=(command,))
            self.process.start()
            self.onProcessStarted()
            self.ui.labelBar.setText(f'Loading {self.task} model...')

            self.ui.applyChangesButton.setEnabled(False)
            self.ui.doneLabel.setHidden(True)
            self.ui.timeLabel.setHidden(False)
            self.ui.progressLabel.setHidden(False)
            self.ui.timeLabel.setText(f"time : 0.00s")
            start_time = time.time()
            previous_time = start_time
            while self.process.is_alive():
              slicer.app.processEvents()
              self.onProcessUpdate()
              current_time = time.time()
              gap=current_time-previous_time
              if gap>0.3:
                previous_time = current_time
                elapsed_time = current_time - start_time
                self.ui.timeLabel.setText(f"time : {elapsed_time:.2f}s")
              self.onProcessCompleted()
            
      self.ui.applyChangesButton.setEnabled(True)
      self.ui.cancelButton.setHidden(True)

  def resetProgressBar(self):
    self.ui.progressBar.setValue(0)
    self.progress = 0
    self.previous_saxi_task='predict'
    self.process_completed= False

    self.ui.timeLabel.setVisible(False)
    self.ui.labelBar.setVisible(False)
    self.ui.progressLabel.setText('Prediction in progress...')
    self.ui.progressLabel.setVisible(False)
    
    self.ui.progressBar.setVisible(False)
    self.ui.progressBar.setRange(0,100)
    self.ui.progressBar.setTextVisible(True)
    self.ui.progressBar.setValue(0)
    self.ui.progressBar.setFormat("")

  def onProcessStarted(self):
    self.nbSubjects = 0
    self.nbSubjects += sum(1 for elt in os.listdir(self.input_dir) if os.path.splitext(elt)[1] == '.vtk')

    self.ui.progressBar.setValue(0)
    self.progress = 0
    self.previous_saxi_task='predict'
    self.process_completed= False
      
    self.start_time = time.time()
    self.previous_time = self.start_time  
    
    self.ui.applyChangesButton.setEnabled(False)
    self.ui.doneLabel.setHidden(True)
    self.ui.cancelButton.setHidden(False)
    self.ui.labelBar.setHidden(False)
    self.ui.timeLabel.setHidden(False)
    self.ui.progressLabel.setHidden(False)
    self.ui.progressBar.setHidden(False)
    self.ui.timeLabel.setText(f"time : 0.00s") 


  def onProcessUpdate(self):
    if os.path.isfile(self.log_path):
      time_progress = os.path.getmtime(self.log_path)

      if time_progress != self.time_log :
        with open(self.log_path, 'r') as f:
          line = f.readline()
          if line != '':
            current_saxi_task, progress, class_idx, num_classes = line.strip().split(',')
            self.progress = int(progress)

            if self.previous_saxi_task != current_saxi_task: 
              print("reset progress bar and self.progresss")
              self.progress = 0
              self.ui.progressBar.setValue(0)
              self.previous_saxi_task = current_saxi_task

            if current_saxi_task == 'explainability':
              self.ui.progressLabel.setText('Explainability in progress...')
              self.ui.labelBar.setText(f"{self.task} model\nClass {class_idx}/{int(num_classes)-1} \nNumber of processed subjects : {self.progress}/{self.nbSubjects}")
              total_progress = self.progress + int(class_idx) * self.nbSubjects
              overall_progress = total_progress / (self.nbSubjects * int(num_classes)) * 100
              progressbar_value = round(overall_progress, 2)
              if progressbar_value == 100:
                self.process_completed=True

            else:
              self.ui.labelBar.setText(f"{self.task} model\nNumber of processed subjects : {self.progress}/{self.nbSubjects}")
              progressbar_value = round((self.progress) /self.nbSubjects * 100,2)

            self.time_log = time_progress

            self.ui.progressBar.setValue(progressbar_value)
            self.ui.progressBar.setFormat(str(progressbar_value)+"%")


  def onProcessCompleted(self):
    self.ui.applyChangesButton.setEnabled(True)
    self.ui.resetButton.setEnabled(True)
    self.ui.progressLabel.setHidden(False)     
    self.ui.cancelButton.setHidden(True)
    self.resetProgressBar()
    self.ui.doneLabel.setHidden(False)
    print("Process completed successfully.")
    
    elapsed_time = round(time.time() - self.start_time,3)
    self.ui.timeLabel.setText(f"time : {elapsed_time:.2f}s")

    self.ui.doneLabel.setHidden(False)
      
  def onReset(self):
    self.ui.outputLineEdit.setText("")
    self.ui.mountPointLineEdit.setText("")

    self.ui.applyChangesButton.setEnabled(True)
    self.resetProgressBar()
    self.ui.progressLabel.setHidden(True)
    self.ui.doneLabel.setHidden(True)
    self.ui.timeLabel.setHidden(True)

    self.removeObservers()  

  def onCancel(self):
    print("cancelling processs, be patient")
    self.ui.labelBar.setText(f'Cancelling process...')
    if self.process: # windows
      self.process.join()

    else: #linux
      self.logic.cliNode.Cancel()

    self.cancel=True
    self.ui.applyChangesButton.setEnabled(True)
    self.ui.resetButton.setEnabled(True)
    self.resetProgressBar()
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
    clean_output = output.replace('\x00', '')

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

  """

  def __init__(self, input_dir = "None", output_dir="None", data_type="None", task='severity', log_path='./'):
    
    """Called when the logic class is instantiated. Can be used for initializing member variables."""
    ScriptedLoadableModuleLogic.__init__(self)

    self.output_dir = output_dir
    self.input_dir = input_dir
    self.data_type = data_type
    self.task = task
    self.log_path = log_path

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
    parameters['log_path'] = self.log_path

    shapeaxi_process = slicer.modules.shapeclassificationcli
    self.cliNode = slicer.cli.run(shapeaxi_process,None, parameters)  
    return shapeaxi_process