import glob
import os
from shutil import copyfile, move
from utils import *


def preprocess_folder(in_folder_path, T1Keyword="T1", FLAIRKeyword='FLAIR'):  # receives absolute path
    listaT1 = keyword_toList(in_folder_path, T1Keyword)
    listaFLAIR = keyword_toList(in_folder_path, FLAIRKeyword)
    mni_FLAIRs = []
    mni_T1s = []
    mni_MASKSs = []
    intoT1s = []
    affines = []
    for T1, FLAIR in zip(listaT1, listaFLAIR):
        print('processing: ' + T1 + ' and ' + FLAIR)
        nativeT1_name = in_folder_path + 'native_' + T1.split('/')[-1]
        nativeFLAIR_name = in_folder_path + 'native_' + FLAIR.split('/')[-1]
        nativeT1_name = replace_extension(nativeT1_name, '.gz', '')
        nativeFLAIR_name = replace_extension(nativeFLAIR_name, '.gz', '')
        if('.gz' in FLAIR):
            copyfile(FLAIR, nativeFLAIR_name + '.gz')
            run_command('gunzip ' + stringify(nativeFLAIR_name + '.gz'))
        else:
            copyfile(FLAIR, nativeFLAIR_name)
        if('.gz' in T1):
            copyfile(T1, nativeT1_name + '.gz')
            run_command('gunzip ' + stringify(nativeT1_name + '.gz'))
        else:
            copyfile(T1, nativeT1_name)
        newT1, newFLAIR, new_mask, new_intot1, new_affine = preprocess_file(nativeT1_name, nativeFLAIR_name)
        mni_T1s.append(newT1)
        mni_FLAIRs.append(newFLAIR)
        mni_MASKSs.append(new_mask)
        intoT1s.append(new_intot1)
        affines.append(new_affine)
    return mni_T1s, mni_FLAIRs, mni_MASKSs, intoT1s, affines, listaT1, listaFLAIR


def preprocess_file(nativeT1_name, nativeFLAIR_name, output_dir):  # receives absolute path

    print('processing: ' + nativeT1_name + ' and ' + nativeFLAIR_name)
    matlabBin = './lesionBrain_v11_fullpreprocessing_exe.sh'
    #command = matlabBin + ' ' + stringify(nativeT1_name) + ' ' + stringify(nativeFLAIR_name)
    command = "{} {} {}".format(matlabBin, stringify(nativeT1_name), stringify(nativeFLAIR_name))
    print("command=", command)
    run_command(command)

    # Rename produced files

    dirname = os.path.dirname(nativeT1_name)
    assert(dirname == os.path.dirname(nativeFLAIR_name))
    T1_name = os.path.basename(nativeT1_name)
    FLAIR_name = os.path.basename(nativeFLAIR_name)

    # output filenames of MATLAB code
    #TODO: replace_prefix() !!!!
    outT1 = os.path.join(dirname, 'n_mfmni_f' + replace_extension(T1_name, '.nii', '_check.nii'))
    outFLAIR = os.path.join(dirname, 'n_mfmni_f' + replace_extension(FLAIR_name, '.nii', '_check.nii'))
    outMASK = outT1.replace('n_mfmni_f', 'mask_n_mfmni_f')
    outIntoT1 = os.path.join(dirname, 'affine_intot1_f' + replace_extension(FLAIR_name, '.nii', '_checkAffine.txt'))
    outAffine = os.path.join(dirname, 'affine_mf' + replace_extension(T1_name, '.nii', '_checkAffine.txt'))
    outCrisp = outT1.replace('n_mfmni_f', 'crisp_mfmni_f')
    outHemi = outT1.replace('n_mfmni_f', 'hemi_n_mfmni_f')
    outStructures = outT1.replace('n_mfmni_f', 'lab_n_mfmni_f')
    # new names
    newT1 = os.path.join(output_dir, 'mni_t1_' + T1_name)
    newFLAIR = os.path.join(output_dir, 'mni_flair_' + T1_name)
    newMASK = os.path.join(output_dir, 'mni_mask_' + T1_name)
    newIntoT1 = os.path.join(output_dir, 'matrix_affine_flair_to_t1_' + replace_extension(T1_name, '.nii', '.txt'))
    newAffine = os.path.join(output_dir, 'matrix_affine_native_to_mni_' + replace_extension(T1_name, '.nii', '.txt'))
    newCrisp = os.path.join(output_dir, 'mni_tissues_' + T1_name)
    newHemi = os.path.join(output_dir, 'mni_hemi_' + T1_name)  # B:TODO:useless ???
    newStructures = os.path.join(output_dir, 'mni_structures_sym_' + T1_name)  # B:TODO:useless ???
    
    assert os.path.isfile(outT1)
    assert os.path.isfile(outFLAIR)

    move(outT1, newT1)
    move(outFLAIR, newFLAIR)
    move(outMASK, newMASK)
    move(outIntoT1, newIntoT1)
    move(outAffine, newAffine)
    move(outCrisp, newCrisp)  # B:TODO: useless ???
    move(outHemi, newHemi)  # B:TODO: useless ???
    move(outStructures, newStructures)  # B:TODO: useless ???

    os.remove(replace_extension(nativeT1_name, '.nii', '_check.nii'))  # B:TODO: ???
    os.remove(replace_extension(nativeFLAIR_name, '.nii', '_check.nii'))

    os.remove(os.path.join(dirname, "log.txt")) # matlab log file

    return newT1, newFLAIR, newMASK, newIntoT1, newAffine, newCrisp, newHemi, newStructures

def preprocess_time_points(tp1_T1_name, tp1_FLAIR_name, tp2_T1_name, tp2_FLAIR_name, output_dir):  # receives absolute path

    print('processing: ' + tp1_T1_name + ' and ' + tp1_FLAIR_name)
    print(' and '+ tp2_T1_name + ' and ' + tp2_FLAIR_name)
    matlabBin = './run_DeepNewLesion_v10_fullpreprocessing_exe.sh /usr/local/MATLAB/MATLAB_Runtime/v93/'
    command = "{} {} {} {} {}".format(matlabBin, stringify(tp1_T1_name), stringify(tp1_FLAIR_name), stringify(tp2_T1_name), stringify(tp2_FLAIR_name))
    print("command=", command)
    run_command(command)

    # Rename produced files

    dirname = os.path.dirname(tp1_T1_name)
    assert(dirname == os.path.dirname(tp1_FLAIR_name))
    
    # output filenames of MATLAB code
    #TODO: replace_prefix() !!!!
    out_tp1_flair_mnitp1_nyul = os.path.join(dirname, 'mni_flair_nyul_timepoint_1_' + os.path.basename(tp1_T1_name))
    out_tp2_flair_mnitp1_nyul = os.path.join(dirname, 'mni_flair_nuyl_timepoint_2_' + os.path.basename(tp2_T1_name))
    
    out_tp1_flair__mnitp1 = os.path.join(dirname, 'mni_flair_timepoint_1_' + os.path.basename(tp1_T1_name))
    out_tp2_flair__mnitp1 = os.path.join(dirname, 'mni_flair_timepoint_2_' + os.path.basename(tp2_T1_name))
    
    out_tp1_mask_mnitp1 = os.path.join(dirname, 'mni_flair_timepoint_1_' + os.path.basename(tp1_T1_name))
    out_tp2_mask_mnitp1 = os.path.join(dirname, 'mni_flair_timepoint_2_' + os.path.basename(tp2_T1_name))

    out_tp1_t1_mni = os.path.join(dirname, 'mni_template_t1_timepoint_1_' + os.path.basename(tp1_T1_name))
    out_tp2_t1_mni = os.path.join(dirname, 'mni_template_t1_timepoint_2_' + os.path.basename(tp2_T1_name))
    
    out_tp2_flair_mni = os.path.join(dirname, 'mni_template_flair_timepoint_2_' + os.path.basename(tp1_T1_name))

    out_tp1_flair_to_t1= os.path.join(dirname, 'matrix_affine_flair_to_t1_' + replace_extension(os.path.basename(tp1_T1_name), '.nii.gz', '.txt'))
    out_tp2_flair_to_t1= os.path.join(dirname, 'matrix_affine_flair_to_t1_' + replace_extension(os.path.basename(tp2_T1_name), '.nii.gz', '.txt'))
    out_tp1_native_to_mni= os.path.join(dirname, 'matrix_affine_native_to_mni_' + replace_extension(os.path.basename(tp1_T1_name), '.nii.gz', '.txt'))
    out_tp2_native_to_mni= os.path.join(dirname, 'matrix_affine_native_to_mni_' + replace_extension(os.path.basename(tp2_T1_name), '.nii.gz', '.txt'))
    out_mniflair_to_mniflair_for_tp2= os.path.join(dirname, 'matrix_affine_mniflair_to_mniflair_timepoint_2_' + replace_extension(os.path.basename(tp2_T1_name), '.nii.gz', '.txt'))
    
    return out_tp1_flair_mnitp1_nyul, out_tp2_flair_mnitp1_nyul, out_tp1_flair__mnitp1, out_tp2_flair_mni, out_tp1_mask_mnitp1, out_tp2_mask_mnitp1, out_tp1_t1_mni, out_tp2_t1_mni, out_mniflair_to_mniflair_for_tp2

# def ground_truth_toMNI(in_folder_path, preprocessed_out_folder, SEG_keyword):
#     ants_bin = './Registration/antsApplyTransforms'
#     for seg_keyword in SEG_keyword:
#         listaSEG = keyword_toList(in_folder_path, seg_keyword)
#         for inputname in listaSEG:
#             if('.gz' in inputname):
#                 copyfile(inputname, 'tmp.nii.gz')
#                 run_command('gunzip ' + 'tmp.nii.gz')
#                 outputname = inputname.replace(seg_keyword, seg_keyword + '_MNI_')
#                 outputname = outputname.replace('.gz', '')
#                 command = ants_bin + ' -d 3 tmp.nii -r ' + reference_name + ' -o ' + outputname + ' -n MultiLabel[0.3,0] -t [' + transform_name + ', 1]'
#                 run_command(command)
#                 os.remove('tmp.nii')
#             else:
#                 outputname = inputname.replace(seg_keyword, seg_keyword + '_MNI_')
#                 command = ants_bin + ' -d 3 ' + inputname + ' -r ' + reference_name + ' -o ' + outputname + ' -n MultiLabel[0.3,0] -t [' + transform_name + ', 1]'
#                 run_command(command)
#     files_list = keyword_toList(preprocessed_out_folder, '.')
#     for file in files_list:
#         if(not ('.gz' in file)):
#             run_command('gzip -f -9 '+file)
