from prediction import *
from preprocessing import *
import argparse
import os
import shutil
from report_utils import *
from test_utils import *
from make_reports import *
from utils import *
import time
#from  assemblyNetLesionPipeline_common import 

tt0 = time.time()

parser = argparse.ArgumentParser(
    description="""DeepLesionBrain platform version""", formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('tp1_t1', type=str, help='1st time-point T1 filename')
parser.add_argument('tp1_flair', type=str, help='1st time-point FLAIR filename')
parser.add_argument('tp2_t1', type=str, help='2st time-point T1 filename')
parser.add_argument('tp2_flair', type=str, help='2st time-point FLAIR filename')
parser.add_argument('-no-pdf-report', action='store_true')
parser.add_argument('-sex', type=str, default='Unknown')
parser.add_argument('-age', type=str, default='Unknown')
args = parser.parse_args()

"/data/mni_flair_nuyl_timepoint_2_90_122039_T1.nii.gz"
native_tp1_t1_filename = args.tp1_t1
native_tp1_flair_filename = args.tp1_flair
native_tp2_t1_filename = args.tp2_t1
native_tp2_flair_filename = args.tp2_flair
output_dir = os.path.dirname(args.tp1_t1)

#DEBUG
print("native_tp1_t1_filename=", native_tp1_t1_filename)
print("native_tp1_flair_filename=", native_tp1_flair_filename)
print("native_tp2_t1_filename=", native_tp2_t1_filename)
print("native_tp2_flair_filename=", native_tp2_flair_filename)


def process_files(native_tp1_t1_filename, native_tp1_flair_filename,
                     native_tp2_t1_filename, native_tp2_flair_filename, output_dir,
                  age, sex, bound_df, no_pdf_report=False, platform=True):
                  

    #Matlab code does not support gzipped files and spaces in filenames

    pwd = "/opt/new_les" 
    
    
    native_files= [native_tp1_t1_filename, native_tp1_flair_filename, native_tp2_t1_filename, native_tp2_flair_filename]
    tmp_filename = [os.path.join(output_dir, os.path.basename(native_filename).replace(" ", "_"))   for native_filename in native_files]
    
    #DEBUG
    
    t0 = time.time()

    #be careful the masks is on tp1 space
    """
    out_tp1_flair_mnitp1_nyul, out_tp2_flair_mnitp1_nyul, out_tp1_flair_mnitp1, out_tp2_flair_mni, out_tp1_mask_mni, out_tp2_mask_mnitp1, out_tp1_t1_mni, out_tp2_t1_mni, out_mniflair_to_mniflair_for_tp2  = preprocess_time_points(*tmp_filename, output_dir)

    """
    out_tp1_flair_mnitp1_nyul, out_tp2_flair_mnitp1_nyul  = "/data/mni_flair_nyul_timepoint_1_90_99172_T1.nii.gz", "/data/mni_flair_nuyl_timepoint_2_90_122039_T1.nii.gz"
    out_tp1_flair_mnitp1, out_tp2_flair_mni= "/data/mni_flair_nyul_timepoint_1_90_99172_T1.nii.gz", "/data/mni_flair_nuyl_timepoint_2_90_122039_T1.nii.gz"
    out_tp1_mask_mni, out_tp2_mask_mnitp1= "/data/mni_mask_timepoint_1_90_99172_T1.nii.gz", "/data/mni_mask_timepoint_2_90_122039_T1.nii.gz"
    out_tp1_t1_mni, out_tp2_t1_mni= "/data/mni_template_t1_timepoint_1_90_99172_T1.nii.gz", "/data/mni_template_t1_timepoint_2_90_122039_T1.nii.gz"
    out_mniflair_to_mniflair_for_tp2="/data/matrix_affine_mniflair_to_mniflair_timepoint_2_90_122039_T1.txt"
    
    
    run_command("ls {}".format(output_dir)) #DEBUG

    """
    out_tp2_mask_mni = to_native(out_tp2_mask_mnitp1, out_mniflair_to_mniflair_for_tp2, out_tp2_t1_mni, dtype='uint8')
    """
    
    t1 = time.time()
    Weights_list = keyword_toList(path='/Weights/', keyword='.pt')
    
    
    """
    mnitp1_new_les= get_new_lesions_mni(Weights_list, out_tp1_flair_mnitp1_nyul, out_tp2_flair_mnitp1_nyul, None, strategy='decoder_with_FMs')

    Weights_list_DLB = keyword_toList(path='/Weights_DLB/', keyword='.h5')
    all_lesions_filename_tp1 = segment_image(nbNN=[5, 5, 5], ps=[96, 96, 96],
                                         Weights_list=Weights_list_DLB,
                                         T1=out_tp1_t1_mni, FLAIR=out_tp1_flair_mnitp1,
                                         FG=out_tp1_mask_mni, normalization="kde")
    all_lesions_filename_tp2 = segment_image(nbNN=[5, 5, 5], ps=[96, 96, 96],
                                         Weights_list=Weights_list_DLB,
                                         T1=out_tp2_t1_mni, FLAIR=out_tp2_flair_mni,
                                         FG=out_tp2_mask_mni, normalization="kde")

    t2 = time.time()
    """
    all_lesions_filename_tp1="/data/all_lesions_mni_template_t1_timepoint_1_90_99172_T1.nii.gz"
    all_lesions_filename_tp2="/data/all_lesions_mni_template_t1_timepoint_2_90_122039_T1.nii.gz"
    mnitp1_new_les="/data/mni_new_lesions_timepoint_2_90_122039_T1.nii.gz"

    timepoints_segmentation_consistency(all_lesions_filename_tp1, all_lesions_filename_tp2, mnitp1_new_les, method="new_les_fidelity")

    #"""
    os.chdir("Inpainting/")
    run_command(f"python3 -u doNonBlindInpainting.py {stringify(out_tp1_t1_mni)}")
    run_command(f"python3 -u doNonBlindInpainting.py {stringify(out_tp2_t1_mni)}")
    os.chdir(pwd)

    print("### after inpainting")
    print("------------------------------")
    
    #RegQCNet
    os.chdir("QualityControl/DeepQCReg/")
    run_command(f"python3 -u deepQCReg.py {stringify(out_tp1_t1_mni)}")
    run_command(f"python3 -u deepQCReg.py {stringify(out_tp2_t1_mni)}")
    os.chdir(pwd)

    print("### after RegQCNet")
    print("------------------------------")
    
    #DeepICE
    # os.chdir("Segmentation/DeepICE")
    # cmd = "python3 -u DeepICE.py {}".format(stringify(batch_tmp_t1_filenames))
    # run_command(cmd)
    # os.chdir(pwd)

           
    #DeepReg
    os.chdir("Registration/DeepREG/version2/")
    run_command(f"python3 -u DeepReg.py {stringify(out_tp1_t1_mni)}")
    run_command(f"python3 -u DeepReg.py {stringify(out_tp2_t1_mni)}")
    os.chdir(pwd)

    print("### after DeepReg")
    #run_command(f"ls -l {TMP_DIR}") #DEBUG
    print("------------------------------")

    #"""
    
    # need to rename so the input of assemblynet is n_mmni_fINPUT, and mask_n_mmni_fINPUT
    #AssemblyNet
    os.chdir("Segmentation/AssemblyNET")
    """
    run_command(f"python3 -u segment.py {stringify(out_tp1_t1_mni)}")
    run_command(f"python3 -u segment.py {stringify(out_tp2_t1_mni)}")
    """
    assemblynet_seg_tp1= "/data/Assembly_seg_1mm_n_mmni_fmni_template_t1_timepoint_1_90_99172_T1.nii.gz"
    assemblynet_seg_tp2= "/data/Assembly_seg_1mm_n_mmni_fmni_template_t1_timepoint_2_90_122039_T1.nii.gz"
    
    
    mni_lesion_filename_tp1, results_lesion_type_tp1 = get_lesion_by_regions_Assemblynet(assemblynet_seg_tp1, all_lesions_filename_tp1)
    mni_lesion_filename_tp2, results_lesion_type_tp2 = get_lesion_by_regions_Assemblynet(assemblynet_seg_tp2, all_lesions_filename_tp2)
    mni_structures_filename_tp1 = get_structures_from_Assemblynet( assemblynet_seg_tp1)
    mni_structures_filename_tp2 = get_structures_from_Assemblynet( assemblynet_seg_tp2)
    mni_tissues_tp1= get_tissues_from_Assemblynet( assemblynet_seg_tp1)
    mni_tissues_tp2= get_tissues_from_Assemblynet( assemblynet_seg_tp2)
    
        
    # mni_lesion_filename is already gzipped (as passed mni_t1 was)

    t5 = time.time()

    
    insert_lesions(mni_tissues_tp1, assemblynet_seg_tp1)
    insert_lesions(mni_tissues_tp2, assemblynet_seg_tp2)
    
    remove_lesions(mni_structures_filename_tp1, assemblynet_seg_tp1)
    remove_lesions(mni_structures_filename_tp2, assemblynet_seg_tp2)
    
    t7 = time.time()

    report(out_tp1_t1_mni, out_tp1_flair_mnitp1,out_tp1_mask_mni, mni_structures_filename_tp1,  
           out_mniflair_to_mniflair_for_tp2, mni_tissues_tp1, mni_lesion_filename_tp1, bounds_df, age, sex, no_pdf_report)
    
    report(out_tp2_t1_mni, out_tp2_flair_mni, out_tp2_mask_mnitp1, mni_structures_filename_tp2,  
           out_mniflair_to_mniflair_for_tp2, mni_tissues_tp2, mni_lesion_filename_tp2, bounds_df, age, sex, no_pdf_report)
    
    end

    # os.remove(unfiltred_t1_filename)
    os.remove(hemi_filename)
    os.remove(all_lesions_filename)
    os.remove(structures_sym_filename)

    t8 = time.time()

    # Copy README.pdf
    shutil.copyfile("README.pdf", os.path.join(os.path.dirname(mni_t1_filename), "README.pdf"))

    if platform:
        # [platform] Save previews
        save_img_preview(mni_t1_filename+'.gz')
        save_img_preview(mni_flair_filename+'.gz')
        shutil.copyfile(mni_lesion_filename, os.path.join(os.path.dirname(mni_lesion_filename), "preview_"+os.path.basename(mni_lesion_filename)))
        shutil.copyfile(mni_structures_filename, os.path.join(os.path.dirname(mni_structures_filename), "preview_"+os.path.basename(mni_structures_filename)))
        shutil.copyfile(mni_mask_filename+".gz", os.path.join(os.path.dirname(mni_mask_filename), "preview_"+os.path.basename(mni_mask_filename)+".gz"))
        shutil.copyfile(crisp_filename+".gz", os.path.join(os.path.dirname(crisp_filename), "preview_"+os.path.basename(crisp_filename)+".gz")) #mni_tissues
        # [platform] Copy original files
        shutil.copyfile(native_t1_filename, os.path.join(os.path.dirname(mni_t1_filename), os.path.basename(mni_t1_filename).replace("mni_t1_", "original_t1_")+".gz"))
        shutil.copyfile(native_flair_filename, os.path.join(os.path.dirname(mni_t1_filename), os.path.basename(mni_t1_filename).replace("mni_t1_", "original_flair_")+".gz"))
        # remove original_t1_{}.nii (not compressed) produced by preprocessing
        # orig_t1_filename = os.path.join(os.path.dirname(mni_t1_filename), os.path.basename(mni_t1_filename).replace("mni_t1_", "original_t1_"))
        # os.remove(orig_t1_filename)

    t9 = time.time()
    print("time preprocess={:.2f}s".format(t1-t0))
    print("time segment={:.2f}s".format(t2-t1))
    print("time toNative={:.2f}s".format(t3-t2))
    print("time gzip={:.2f}s".format(t4-t3))
    print("time lesions={:.2f}s".format(t5-t4))
    print("time lesion native+gzip={:.2f}s".format(t6-t5))
    print("time insert={:.2f}s".format(t7-t6))
    print("time report={:.2f}s".format(t8-t7))
    if platform:
        print("time previews={:.2f}s".format(t9-t8))


bounds_df = read_bounds(args.sex) if (args.age != "UNKNOWN" and not args.no_pdf_report) else read_bounds("")
        
process_files(native_tp1_t1_filename, native_tp1_flair_filename,
            native_tp2_t1_filename, native_tp2_flair_filename,
            output_dir, args.age, args.sex, bounds_df, args.no_pdf_report)

tt1 = time.time()
print("TOTAL processing time={:.2f}s".format(tt1-tt0))
