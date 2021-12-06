import numpy as np
import os
import time
from dipy.io.image import load_nifti, save_nifti
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
import dipy.reconst.dti as dti
from dipy.data import get_sphere
from dipy.reconst.csdeconv import auto_response_ssst, mask_for_response_ssst, response_from_mask_ssst
from dipy.sims.voxel import single_tensor_odf
from dipy.data import get_fnames, default_sphere
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.direction import peaks_from_model
from dipy.reconst.odf import gfa
import matplotlib.pyplot as plt
import dipy.denoise.noise_estimate as ne
from utils import extractSubDirections, mergeStacks, mergeBvecs, mergeBvals
#from dipy.viz import window, actor
sphere = get_sphere('repulsion724')

 
#Generate b-vecs/b-val indices
inds_9 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] #b0 followed by first 9 directions
inds_16 = [0, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25] #b0 followed by last 16 directions
inds_25 = [i for i in range(26)]

root_data_path = '/media/hkebiri/Storage/Phantom_pipeline/AllStacks'
#root_data_path = '/media/hkebiri/Storage/Phantom_pipeline/HighRes_1.5iso_3Tesla/distCorr/pGT4'
#root_data_path = '/home/hkebiri/Desktop/mialsuperresolutiontoolkit/imageReconstructionDWIs/'

stacks = ['Singletons3_3/coronal_1']
distCorr = True
DTI = True
fODF = True
saveMaps = True
interactive = False
inds = inds_25 #choose which number of directions to use between: inds_9, inds_16 and inds_25
save_dir = 'coronal_1/distCorr/pGT_middleGround/maps/DistCorr_Sh_4_Min_Sep_Angle_25_fa_thr_04_bigROI'
mask_path = '/media/hkebiri/Storage/Phantom_pipeline/AllStacks/Singletons3_3/coronal_1/distCorr/pGT_middleGround/0_output_nlr_4D_mask_mg.nii.gz'
#mask_1fiber = '/media/hkebiri/Storage/Phantom_pipeline/HighRes_1.5iso_3Tesla/distCorr/pGT3/mask_1fiber.nii.gz'


save_dir_path = os.path.join(root_data_path, save_dir)
if not os.path.exists(save_dir_path):
  os.makedirs(save_dir_path)

mask, aff = load_nifti(mask_path)
#mask_1fiber, aff = load_nifti(mask_path)

print("Mask shape",mask.shape)

#fODF parameters
sh_order = 4
min_separation_angle = 25
fa_thr = 0.15 #0.25 #0.15 #value that best separates cluster 1 and 2
n_peaks = 2
relative_peak_threshold = 0.5


#Center and radius where to compute the fiber ODF
z = int(np.round(sub_data.shape[2] / 2))
y = int(np.round(sub_data.shape[1] / 2))
x = int(np.round(sub_data.shape[0] / 2))
roi_center = (x,y,z)

#if root_data_path[-1] =='1': #PGT1
roi_radii = (15,30,3)


#Data extraction
bvecs = []
bvals = []
dataPaths = []
for path, subdirs, files in os.walk(root_data_path):
    subdirs.sort()
    files.sort()
    for name in files:
        if any(elem in path for elem in stacks):# and not 'pGT' in path:
            if name.endswith(".bvec"):
                bvecs.append(os.path.join(path,name))
            if name.endswith(".bval"): 
                bvals.append(os.path.join(path,name))
            if distCorr:# and 'pGT1' in path:
                if  config in name and 'distCorr' in path: 
                    dataPaths.append(os.path.join(path,name))
            else:
                if 'BiasCorr' in path and name.endswith('denoised_biascorr.nii.gz'):
                    dataPaths.append(os.path.join(path, name))

print(dataPaths)
print(bvecs)
print(bvals)

sub_data = []
sub_bvecs = []
sub_bvals = []

for i,dataPath in enumerate(dataPaths):
    data, affine = load_nifti(dataPath)
    data = np.squeeze(data)
    bval, bvec = read_bvals_bvecs(bvals[i], bvecs[i])
    sub_data.append(extractSubDirections(data, inds))
    sub_bvecs.append(extractSubDirections(bvec, inds,b=True))
    sub_bvals.append(extractSubDirections(bval, inds,b=True))

sub_bvecs = np.asarray(sub_bvecs)
sub_bvals = np.asarray(sub_bvals)


#Merge multiple stacks in one big stack
sub_data = mergeStacks(sub_data)
sub_bvecs = mergeBvecs(sub_bvecs)
sub_bvals = mergeBvals(sub_bvals)

#sub_data[sub_data<0]=0
sub_data[np.isnan(sub_data)]=0
sub_data = sub_data.astype('float32')
print("Shape of sub_data",sub_data.shape)


if DTI:
    # Fitting the tensor
    start = time.time()
    tenmodel = dti.TensorModel(gtab)
    print(tenmodel)
    tenfit = tenmodel.fit(sub_data,mask=mask)
    print("Tensor fitting took", (time.time() - start)/60, "min")

    #Compute FA
    FA = dti.fractional_anisotropy(tenfit.evals)
    FA[np.isnan(FA)] = 0
    FA = np.clip(FA, 0, 1)

    #Compute MD
    MD = dti.mean_diffusivity(tenfit.evals)
    MD[np.isnan(MD)] = 0
    MD = np.clip(MD, 0, 1)

    #Compute MD
    RD = dti.radial_diffusivity(tenfit.evals)
    RD[np.isnan(RD)] = 0
    RD = np.clip(RD, 0, 1)

    #Compute MD
    AD = dti.axial_diffusivity(tenfit.evals)
    AD[np.isnan(AD)] = 0
    AD = np.clip(AD, 0, 1)

    RGB = dti.color_fa(FA, tenfit.evecs)
    print(np.shape(tenfit.evecs))
    print(np.shape(tenfit.evals))
    if saveMaps:
        save_nifti(os.path.join(save_dir_path, 'AD.nii.gz'), AD, affine)
        save_nifti(os.path.join(save_dir_path, 'RD.nii.gz'), RD, affine)
        save_nifti(os.path.join(save_dir_path,'MD.nii.gz'), MD, affine)
        save_nifti(os.path.join(save_dir_path,'FA.nii.gz'), FA, affine)
        save_nifti(os.path.join(save_dir_path,'main_evec.nii.gz'), tenfit.evecs[:,:,:,:,0], affine)
        save_nifti(os.path.join(save_dir_path, 'evecs.nii.gz'), tenfit.evecs, affine)
        save_nifti(os.path.join(save_dir_path, 'evals.nii.gz'), tenfit.evals, affine)
        save_nifti(os.path.join(save_dir_path,'colorFA.nii.gz'), np.array(255 * RGB, 'uint8'), affine)
        print("Maps saved")

    if interactive:

        #beg,end = ax_beg,ax_end
        ren = window.Renderer()
        evals = tenfit.evals[:,:,z:z+1]
        evecs = tenfit.evecs[:,:,z:z+1]
        RGB /= RGB.max()

        cfa = RGB[:,:,z:z+1]
        cfa /= cfa.max()

        ren.add(actor.tensor_slicer(evals, evecs, scalar_colors=cfa, sphere=sphere, scale=0.4))

        #ren.add(slice_actor)
        ren.zoom(8)
        ren.reset_clipping_range()
        if interactive:
            show_m = window.ShowManager(ren, size=(1200, 900))
            show_m.initialize()
            show_m.render()
            show_m.start()

            #window.show(ren)

if fODF:
    
    # Uncomment below is using mask of 1 or 2 fibers population
    # Reponse function estimation from 1/2 fiber population

    #mask = mask_for_response_ssst(gtab, sub_data, roi_radii=roi_radii, fa_thr=fa_thr)
    #print("Number of voxels before fiber masking", np.sum(mask))
    #mask = mask_fibers*mask

    #print("Number of voxels after fiber masking",np.sum(mask_1fiber))
    #response, ratio = response_from_mask_ssst(gtab, sub_data, mask_1fiber)

    response, ratio = auto_response_ssst(gtab, sub_data, roi_center=roi_center, roi_radii=roi_radii, fa_thr=fa_thr)
    print("Response",response)
    print("Ratio", ratio)

    # Enables/disables interactive visualization

    evals = response[0]
    evecs = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]).T

    response_odf = single_tensor_odf(default_sphere.vertices, evals, evecs)
    # transform our data from 1D to 4D
    response_odf = response_odf[None, None, None, :]
    if interactive:
        scene = window.Scene()
        response_actor = actor.odf_slicer(response_odf, sphere=default_sphere,
                                          colormap='plasma')
        scene.add(response_actor)
        print('Saving illustration as csd_response.png')
        window.record(scene, out_path='csd_response.png', size=(200, 200))
        window.show(scene)

    #fODF estimation using CSV on diff. signal and response function
    start = time.time()
    csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order = sh_order) #response contains information about the data (from single fibers)
    csd_fit = csd_model.fit(sub_data,mask=mask)
    csd_odf = csd_fit.odf(default_sphere) #csd_odf = csd_peaks.odf
    print("Fiber ODF fitting (response function estimation and CSD) took", (time.time() - start) / 60, "min")
    if interactive:
        fodf_spheres = actor.odf_slicer(csd_odf, sphere=default_sphere, scale=0.9,
                                        norm=False, colormap='plasma')

        scene.add(fodf_spheres)

        print('Saving illustration as csd_odfs.png')
        window.record(scene, out_path='csd_odfs.png', size=(600, 600))
        window.show(scene)

    #Peaks
    start = time.time()
    csd_peaks = peaks_from_model(model=csd_model,
                                 sh_basis_type='tournier07',
                                 data=sub_data,
                                 mask=mask,
                                 sphere=default_sphere,
                                 relative_peak_threshold=relative_peak_threshold,
                                 min_separation_angle=min_separation_angle,
                                 npeaks = n_peaks,
                                 sh_order = sh_order,
                                 return_odf = False,
                                 parallel=True,
                                 num_processes=10) #max 12
                     
    save_nifti(os.path.join(save_dir_path, 'Csd_peaks_shm_coeff_tournier_basis_strides_origBvecsMaskBig.nii.gz'), csd_peaks.shm_coeff, affine)
    np.save(os.path.join(save_dir_path, 'Csd_peak_directions_tournier_basis_strides'), csd_peaks.peak_dirs)
    np.save(os.path.join(save_dir_path, 'Csd_peak_values_tournier_basis_strides'), csd_peaks.peak_values)  #value of max(s) ODF extracted
    #np.save(os.path.join(save_dir_path, 'Csd_peak_indices'), csd_peaks.peak_indices) #indices where extracted ODF is max(s)
    np.save(os.path.join(save_dir_path, 'Csd_peak_GFA_coeff_tournier_basis_strides'), csd_peaks.gfa)
    #np.save(os.path.join(save_dir_path, 'Csd_peak_ODF'), csd_peaks.odf) #whole ODF (e.g. of lenght 300), from which peaks are extracted
    print("Peaks extraction took", (time.time() - start) / 60, "min")

    if interactive:
        scene.clear()
        fodf_peaks = actor.peak_slicer(csd_peaks.peak_dirs, csd_peaks.peak_values)
        scene.add(fodf_peaks)

        print('Saving illustration as csd_peaks.png')
        window.record(scene, out_path='csd_peaks.png', size=(600, 600))
        window.show(scene)

