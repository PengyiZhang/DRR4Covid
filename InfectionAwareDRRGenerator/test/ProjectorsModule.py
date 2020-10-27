"""Module for generation of Digitally Reconstructed Radiographs (DRR).

This module includes classes for generation of DRRs from either a volumetric image (CT,MRI) 
or a STL model, and a projector class factory.

Classes:
    SiddonGpu: GPU accelerated (CUDA) DRR generation from CT or MRI scan.  
    Mahfouz: binary DRR generation from CAD model in STL format.

Functions:
    projector_factory: returns a projector instance.
    
New projectors can be plugged-in and added to the projector factory
as long as they are defined as classes with the following methods:
    compute: returns a 2D image (DRR) as a numpy array.
    delete: eventually deletes the projector object (only needed to deallocate memory from GPU) 
"""

####  PYTHON MODULES
import numpy as np
import time
import sys

####  Python ITK/VTK MODULES
import itk
import cv2
import vtk
from vtk.util import numpy_support


import random





####  MY MODULES
import ReadWriteImageModule as rw
import RigidMotionModule as rm

sys.path.append('../wrapped_modules/')
from SiddonGpuPy import pySiddonGpu     # Python wrapped C library for GPU accelerated DRR generation



def projector_factory(projector_info,
                      movingImageFileName,
                      maskFileName=None,
                      PixelType = itk.F,
                      Dimension = 3,
                      ScalarType = itk.D,
                      ClassWeights=[1.0,1.0,1.0]):

    """Generates instances of the specified projectors.

    Args:
        projector_info (dict of str): includes camera intrinsic parameters and projector-specific parameters
        movingImageFileName (string): cost function returning the metric value

    Returns:
        opt: instance of the specified projector class.
    """

    p = SiddonGpu(projector_info,
                    movingImageFileName,
                    maskFileName,
                    PixelType,
                    Dimension,
                    ScalarType,
                    ClassWeights=ClassWeights)

    return p



class SiddonGpu():

    """GPU accelearated DRR generation from volumetric image (CT or MRI scan).

       This class renders a DRR from a volumetric image, with an accelerated GPU algorithm
       from a Python wrapped library (SiddonGpuPy), written in C++ and accelerated with Cuda.
       IMplementation is based both on the description found in the “Improved Algorithm” section in Jacob’s paper (1998): 
       https://www.researchgate.net/publication/2344985_A_Fast_Algorithm_to_Calculate_the_Exact_Radiological_Path_Through_a_Pixel_Or_Voxel_Space
       and on the implementation suggested in Greef et al 2009:
       https://www.ncbi.nlm.nih.gov/pubmed/19810482

       Methods:
            compute (function): returns a 2D image (DRR) as a numpy array.
            delete (function): deletes the projector object (needed to deallocate memory from GPU)
    """


    def __init__(self, projector_info,
                       movingImageFileName,
                       maskFileName,
                       PixelType,
                       Dimension,
                       ScalarType,
                       ClassWeights=[1.0,1.0,1.0]):

        """Reads the moving image and creates a siddon projector 
           based on the camera parameters provided in projector_info (dict)
        """

        # ITK: Instantiate types
        self.projector_info = projector_info
        self.Dimension = Dimension
        self.ImageType = itk.Image[PixelType, Dimension]
        self.ImageType2D = itk.Image[PixelType, 2]

        # self.MaskType = itk.Image[itk.UC, Dimension]
        # self.MaskType2D = itk.Image[itk.UC, 2]
        self.MaskType = itk.Image[PixelType, Dimension]
        self.MaskType2D = itk.Image[PixelType, 2]

        self.RegionType = itk.ImageRegion[Dimension]
        self.PhyImageType=itk.Image[itk.Vector[itk.F,Dimension],Dimension] # image of physical coordinates

        # Read moving image (CT or MRI scan)
        movImage, movImageInfo = rw.ImageReader(movingImageFileName, self.ImageType)
        movMask, movMaskInfo = rw.ImageReader(maskFileName, self.ImageType)

        self.movImageInfo = movImageInfo

        self.movDirection = movImage.GetDirection()
        self.movDirectionMatrix = itk.GetArrayFromMatrix(self.movDirection)
        print(self.movDirectionMatrix)
        print(movImageInfo['Spacing'])

        # self.projector_info['DRRspacing_x'] = movImageInfo['Spacing'][0] / 3
        # self.projector_info['DRRspacing_y'] = movImageInfo['Spacing'][2] / 3

        # self.projector_info['DRRsize_x'] = movImageInfo['Size'][0]
        # self.projector_info['DRRsize_y'] = movImageInfo['Size'][2]

        # Calculate side planes
        X0 = movImageInfo['Volume_center'][0] - movImageInfo['Spacing'][0]*movImageInfo['Size'][0]/2.0
        Y0 = movImageInfo['Volume_center'][1] - movImageInfo['Spacing'][1]*movImageInfo['Size'][1]/2.0
        Z0 = movImageInfo['Volume_center'][2] - movImageInfo['Spacing'][2]*movImageInfo['Size'][2]/2.0

        # Get 1d array for moving image
        #movImgArray_1d = np.ravel(itk.PyBuffer[self.ImageType].GetArrayFromImage(movImage), order='C') # ravel does not generate a copy of the array (it is faster than flatten)
        movImgArray_1d = np.ravel(itk.GetArrayFromImage(movImage), order='C') # ravel does not generate a copy of the array (it is faster than flatten)
        movMaskArray_1d = np.ravel(itk.GetArrayFromImage(movMask), order='C') # ravel does not generate a copy of the array (it is faster than flatten)

        # Set parameters for GPU library SiddonGpuPy
        NumThreadsPerBlock = np.array( [projector_info['threadsPerBlock_x'], projector_info['threadsPerBlock_y'], projector_info['threadsPerBlock_z'] ] , dtype=np.int32)
        DRRsize_forGpu = np.array([ projector_info['DRRsize_x'],  projector_info['DRRsize_y'], 1], dtype=np.int32)
        MovSize_forGpu = np.array([ movImageInfo['Size'][0], movImageInfo['Size'][1], movImageInfo['Size'][2] ], dtype=np.int32)
        MovSpacing_forGpu = np.array([ movImageInfo['Spacing'][0], movImageInfo['Spacing'][1], movImageInfo['Spacing'][2] ]).astype(np.float32)
        Weights = np.array(ClassWeights).astype(np.float32)

        # Define source point at its initial position (at the origin = moving image center)
        self.source = [0]*Dimension
        self.source[0] = movImageInfo['Volume_center'][0]
        self.source[1] = movImageInfo['Volume_center'][1]
        self.source[2] = movImageInfo['Volume_center'][2]

        # Set DRR image at initial position (at +focal length along the z direction)
        DRR = self.ImageType.New()
        self.DRRregion = self.RegionType()

        DRRstart = itk.Index[Dimension]()
        DRRstart.Fill(0)

        self.DRRsize = [0]*Dimension
        self.DRRsize[0] = projector_info['DRRsize_x']
        self.DRRsize[1] = 1 #projector_info['DRRsize_y']
        self.DRRsize[2] = projector_info['DRRsize_y']

        self.DRRregion.SetSize(self.DRRsize)
        self.DRRregion.SetIndex(DRRstart)

        self.DRRspacing = itk.Point[itk.F, Dimension]()
        self.DRRspacing[0] = projector_info['DRRspacing_x']
        self.DRRspacing[1] = 1.
        self.DRRspacing[2] = projector_info['DRRspacing_y']     
        self.DRR = DRR

        tGpu1 = time.time()

        # Generate projector object
        self.projector = pySiddonGpu(NumThreadsPerBlock,
                                  movImgArray_1d,
                                  movMaskArray_1d,
                                  Weights,
                                  MovSize_forGpu,
                                  MovSpacing_forGpu,
                                  X0.astype(np.float32), Y0.astype(np.float32), Z0.astype(np.float32),
                                  DRRsize_forGpu)

        tGpu2 = time.time()

        print( '\nSiddon object initialized. Time elapsed for initialization: ', tGpu2 - tGpu1, '\n')



    def compute(self, transform_parameters):


        """Generates a DRR given the transform parameters.

           Args:
               transform_parameters (list of floats): rotX, rotY,rotZ, transX, transY, transZ
 
        """

        # Get transform parameters
        rotx = transform_parameters[0]
        roty = transform_parameters[1]
        rotz = transform_parameters[2]
        tx = transform_parameters[3]
        ty = transform_parameters[4]
        tz = transform_parameters[5]

        # position movement
        
        # compute the transformation matrix and its inverse (itk always needs the inverse)
        Tr = rm.get_rigid_motion_mat_from_euler(rotz, 'z', rotx, 'x', roty, 'y', 0, 0, 0)
        
        self.movDirectionMatrix = itk.GetArrayFromMatrix(self.movDirection)
        self.movDirectionMatrix = Tr[:3,:3]
        self.movDirection = itk.GetMatrixFromArray(self.movDirectionMatrix)
        self.movDirectionMatrix01 = itk.GetArrayFromMatrix(self.movDirection)

        #tDRR1 = time.time()
        DRR = self.DRR
        movImageInfo = self.movImageInfo
        projector_info = self.projector_info
        PhyImageType = self.PhyImageType

        self.DRRorigin = itk.Point[itk.F, self.Dimension]()
        self.DRRorigin[0] = movImageInfo['Volume_center'][0] - projector_info['DRR_ppx'] - self.DRRspacing[0]*(self.DRRsize[0] - 1.) / 2. + tx
        self.DRRorigin[1] = movImageInfo['Volume_center'][1] + projector_info['focal_lenght'] + ty  #/ 2.
        self.DRRorigin[2] = movImageInfo['Volume_center'][2] - projector_info['DRR_ppy'] - self.DRRspacing[2]*(self.DRRsize[2] - 1.) / 2. + tz

        DRR.SetRegions(self.DRRregion)
        DRR.Allocate()
        DRR.SetSpacing(self.DRRspacing)
        DRR.SetOrigin(self.DRRorigin)
        #print(self.movDirection)
        DRR.SetDirection(self.movDirection)

        # Get array of physical coordinates for the DRR at the initial position 
        PhysicalPointImagefilter=itk.PhysicalPointImageSource[PhyImageType].New()
        PhysicalPointImagefilter.SetReferenceImage(DRR)
        PhysicalPointImagefilter.SetUseReferenceImage(True)
        PhysicalPointImagefilter.Update()
        sourceDRR = PhysicalPointImagefilter.GetOutput()

        #self.sourceDRR_array_to_reshape = itk.PyBuffer[PhyImageType].GetArrayFromImage(sourceDRR)[0].copy(order = 'C') # array has to be reshaped for matrix multiplication
        self.sourceDRR_array_to_reshape = itk.GetArrayFromImage(sourceDRR)#[:,0] # array has to be reshaped for matrix multiplication



        # compute the transformation matrix and its inverse (itk always needs the inverse)
        Tr = rm.get_rigid_motion_mat_from_euler(0, 'z', 0, 'x', 0, 'y', 0, projector_info['focal_lenght'], 0)
        invT = np.linalg.inv(Tr) # very important conversion to float32, otherwise the code crashes


        # Move source point with transformation matrix
        source_transformed = np.dot(invT, np.array([self.source[0],self.source[1],self.source[2], 1.]).T)[0:3]
        source_forGpu = np.array([ source_transformed[0], source_transformed[1], source_transformed[2] ], dtype=np.float32)
        
        # Instantiate new 3D DRR image at its initial position (at +focal length along the z direction)
        newDRR = self.ImageType.New()
        newMask = self.MaskType.New()
        newLung = self.MaskType.New()
        newValue = self.MaskType.New()

        newDRR.SetRegions(self.DRRregion)
        newMask.SetRegions(self.DRRregion)
        newLung.SetRegions(self.DRRregion)
        newValue.SetRegions(self.DRRregion)

        newDRR.Allocate()
        newMask.Allocate()
        newLung.Allocate()
        newValue.Allocate()

        newDRR.SetSpacing(self.DRRspacing)
        newMask.SetSpacing(self.DRRspacing)
        newLung.SetSpacing(self.DRRspacing)
        newValue.SetSpacing(self.DRRspacing)

        newDRR.SetOrigin(self.DRRorigin)
        newMask.SetOrigin(self.DRRorigin)
        newLung.SetOrigin(self.DRRorigin)
        newValue.SetOrigin(self.DRRorigin)

        self.movDirection.SetIdentity()
        newDRR.SetDirection(self.movDirection)
        newMask.SetDirection(self.movDirection)
        newLung.SetDirection(self.movDirection)
        newValue.SetDirection(self.movDirection)

        # Get 3d array for DRR (where to store the final output, in the image plane that in fact does not move)
        #newDRRArray = itk.PyBuffer[self.ImageType].GetArrayFromImage(newDRR)
        newDRRArray = itk.GetArrayViewFromImage(newDRR)
        newMaskArray = itk.GetArrayViewFromImage(newMask)
        newLungArray = itk.GetArrayViewFromImage(newLung)
        newValueArray = itk.GetArrayViewFromImage(newValue)

        #tDRR3 = time.time()

        # Get array of physical coordinates of the transformed DRR
        sourceDRR_array_reshaped = self.sourceDRR_array_to_reshape.reshape((self.DRRsize[0]*self.DRRsize[2], self.Dimension), order = 'C')

        sourceDRR_array_transformed = np.dot(invT, rm.augment_matrix_coord(sourceDRR_array_reshaped))[0:3].T # apply inverse transform to detector plane, augmentation is needed for multiplication with rigid motion matrix

        sourceDRR_array_transf_to_ravel = sourceDRR_array_transformed.reshape((self.DRRsize[0],self.DRRsize[2], self.Dimension), order = 'C')


        DRRPhy_array = np.ravel(sourceDRR_array_transf_to_ravel, order = 'C').astype(np.float32)

        # Generate DRR
        #tGpu3 = time.time()

        output, output_mask, output_mask_lung, output_mask_value = self.projector.generateDRR(source_forGpu,DRRPhy_array)
        #tGpu4 = time.time()

        # Reshape copy
        output_reshaped = np.reshape(output, (self.DRRsize[2], self.DRRsize[1], self.DRRsize[0]), order='C') # no guarantee about memory contiguity
        output_mask_reshaped = np.reshape(output_mask, (self.DRRsize[2], self.DRRsize[1], self.DRRsize[0]), order='C')
        
        output_mask_lung_reshaped = np.reshape(output_mask_lung, (self.DRRsize[2], self.DRRsize[1], self.DRRsize[0]), order='C') # no guarantee about memory contiguity
        output_mask_value_reshaped = np.reshape(output_mask_value, (self.DRRsize[2], self.DRRsize[1], self.DRRsize[0]), order='C')
                
        #output_reshaped = np.reshape(output, (self.DRRsize[2], self.DRRsize[0]), order='C') # no guarantee about memory contiguity

        # Re-copy into original image array, hence into original image (since the former is just a view of the latter)
        newDRRArray.setfield(output_reshaped, newDRRArray.dtype)
        newMaskArray.setfield(output_mask_reshaped, newMaskArray.dtype)
        newLungArray.setfield(output_mask_lung_reshaped, newLungArray.dtype)
        newValueArray.setfield(output_mask_value_reshaped, newValueArray.dtype)

        # Redim filter to convert the DRR from 3D slice to 2D image (necessary for further metric comparison)
        filterRedim = itk.ExtractImageFilter[self.ImageType, self.ImageType2D].New()
        filterRedim.InPlaceOn()
        filterRedim.SetDirectionCollapseToSubmatrix()

        filterRedim_mask = itk.ExtractImageFilter[self.MaskType, self.MaskType2D].New()
        filterRedim_mask.InPlaceOn()
        filterRedim_mask.SetDirectionCollapseToSubmatrix()

        filterRedim_lung = itk.ExtractImageFilter[self.MaskType, self.MaskType2D].New()
        filterRedim_lung.InPlaceOn()
        filterRedim_lung.SetDirectionCollapseToSubmatrix()

        filterRedim_value = itk.ExtractImageFilter[self.MaskType, self.MaskType2D].New()
        filterRedim_value.InPlaceOn()
        filterRedim_value.SetDirectionCollapseToSubmatrix()

        newDRR.UpdateOutputInformation() # important, otherwise the following filterRayCast.GetOutput().GetLargestPossibleRegion() returns an empty image
        newMask.UpdateOutputInformation()
        newLung.UpdateOutputInformation()
        newValue.UpdateOutputInformation()

        size_input = newDRR.GetLargestPossibleRegion().GetSize()
        start_input = newDRR.GetLargestPossibleRegion().GetIndex()

        size_output = [0]*self.Dimension
        size_output[0] = size_input[0]
        size_output[1] = 0
        size_output[2] = size_input[2]

        sliceNumber = 0
        start_output = [0]*self.Dimension
        start_output[0] = start_input[0]
        start_output[1] = sliceNumber
        start_output[2] = start_input[2]

        desiredRegion = self.RegionType()
        desiredRegion.SetSize( size_output )
        desiredRegion.SetIndex( start_output )

        filterRedim.SetExtractionRegion( desiredRegion )
        filterRedim_mask.SetExtractionRegion( desiredRegion )
        filterRedim_lung.SetExtractionRegion( desiredRegion )
        filterRedim_value.SetExtractionRegion( desiredRegion )

        filterRedim.SetInput(newDRR)
        filterRedim_mask.SetInput(newMask)
        filterRedim_lung.SetInput(newLung)
        filterRedim_value.SetInput(newValue)


        #tDRR2 = time.time()

        filterRedim.Update()
        filterRedim_mask.Update()
        filterRedim_lung.Update()
        filterRedim_value.Update()

        #print( '\nTime elapsed for generation of DRR: ', tDRR2 - tDRR1)

        return filterRedim.GetOutput(), filterRedim_mask.GetOutput(), filterRedim_lung.GetOutput(), filterRedim_value.GetOutput()
        


    def delete(self):
        
        """Deletes the projector object >>> GPU is freed <<<"""

        self.projector.delete()