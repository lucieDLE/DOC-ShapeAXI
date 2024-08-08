cmake_minimum_required(VERSION 3.13.4)

project(DOCShapeAXI)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/DCBIA-OrthoLab/SlicerDentalModelSeg/blob/main/readme.md")
set(EXTENSION_CATEGORY "Segmentation")
set(EXTENSION_CONTRIBUTORS "Mathieu Leclercq (University of North Carolina), Connor Bowley (Kitware), Juan Carlos Prieto (University of North Carolina)")
set(EXTENSION_DESCRIPTION "This extension aims to provide a GUI for a deep-learning automated teeth segmentation tool that we developed at the University of North Carolina in Chapel Hill.")
set(EXTENSION_ICONURL "https://github.com/DCBIA-OrthoLab/SlicerDentalModelSeg/raw/main/examples/logo_jaw_segmentation.png")
set(EXTENSION_SCREENSHOTURLS "https://github.com/DCBIA-OrthoLab/SlicerDentalModelSeg/blob/main/examples/segmentation_example.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(DOCShapeAXI)
add_subdirectory(DOCShapeAXIcli)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
