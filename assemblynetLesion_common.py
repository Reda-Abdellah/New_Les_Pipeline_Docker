import os
import numpy as np
# For reproducibility ################
import random as python_random
seed = 1973
np.random.seed(seed)
python_random.seed(seed)
######################################

import argparse
import shutil
import sys
import time
from pathlib import Path
import tempfile

from Segmentation.AssemblyNET.utils import run_command, replace_extensions

#B:impl: cannot import (because it imports local files)
#from Segmentation.DeepLesionBrain.preprocessing import rename_and_remove_preprocessing_output


tt0 = time.time()


UNKNOWN = 'Unknown'

def age_float_type(arg):
    """ Type function for argparse - a float within some predefined bounds """
    if arg == UNKNOWN:
        return arg
    try:
        f = float(arg)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a floating point number")
    MIN_VAL = 0
    MAX_VAL = 130
    if f < MIN_VAL or f > MAX_VAL:
        raise argparse.ArgumentTypeError(f"argument must be > {MIN_VAL} and < {MAX_VAL}")
    return f


def positive_int_type(arg):
    """ Type function for argparse - an positive int """
    try:
        f = int(arg)
    except ValueError:    
        raise argparse.ArgumentTypeError("must be an integer number")
    if f < 0:
        raise argparse.ArgumentTypeError("argument must be > 0")
    return f


def sex_type(arg):
    if arg == UNKNOWN:
        return arg
    sex = str(arg).lower()
    if sex != "female" and sex != "male":
        raise argparse.ArgumentTypeError("argument must be \"female\" or \"male\"")
    return arg



def is_nii_file(p):
    if not p.is_file():
        return False
    pr = str(p.resolve())
    if (pr.endswith(".nii") or pr.endswith(".nii.gz")):
        return True
    return False
    

def find_files(dirname, pattern, recursive):
    if recursive:
        files = sorted([str(p.resolve()) for p in Path(dirname).rglob(pattern) if is_nii_file(p)])
    else:
        files = sorted([str(p.resolve()) for p in Path(dirname).glob(pattern) if is_nii_file(p)])
    return files


def get_dirnames(t1_filenames):
    output_dirs = [os.path.dirname(f) for f in t1_filenames]
    return output_dirs


def get_input_dir_root(t1_filenames):
    return os.path.commonpath(t1_filenames)


def make_output_dirs(t1_input_dir_root, t1_filenames, output_dir_root):
    #output_dirs = [os.path.join(os.path.dirname(f).replace(t1_input_dir_root, output_dir_root), os.path.basename(f)) for f in t1_filenames]
    output_dirs = [os.path.dirname(f).replace(t1_input_dir_root, output_dir_root) for f in t1_filenames]
    for output_dir in output_dirs:
        if not os.path.exists(output_dir):
            Path(output_dir).mkdir(parents=True)  # equivalent to "mkdir -p"
        elif not os.path.isdir(output_dir):
            print("ERROR: {} already exists and is not a directory".format(output_dir))
            sys.exit(1)
        
    return output_dirs


def clean_dirname(dirname):
    if dirname.endswith('/'):
        return dirname[:-1]
    else:
        return dirname

def are_only_files(filenames):
    for filename in filenames:
        if not os.path.isfile(filename):
            return False
    return True
    
    
def get_files_and_output_dirs(filenames, patternT1, patternFLAIR, recursive):
    t1_input_dir_root = ""
    flair_input_dir_root = ""
    output_dir_root = ""
    t1_filenames = []
    flair_filenames = []
    output_dirs = []
    #print("filenames=", filenames)
    filenames = [os.path.abspath(filename) for filename in filenames]
    #print("filenames=", filenames)
    n = len(filenames)
    if (n < 1):
        print("ERROR: invalid number of filenames. It should be \"T1_filename FLAIR_FILENAME [T1_filename2 FLAIR_filename2 ...] [output_dir]\" or \"input_dir [output_dir]\" or \"T1_input_dir FLAIR_input_dir output_dir\"")
        sys.exit(1)
    elif (n == 1):
        if os.path.isdir(filenames[0]):
            t1_input_dir_root = clean_dirname(filenames[0])
            flair_input_dir_root = t1_input_dir_root
            output_dir_root = t1_input_dir_root
            t1_filenames = find_files(t1_input_dir_root, patternT1, recursive)
            flair_filenames = find_files(flair_input_dir_root, patternFLAIR, recursive)
            output_dirs = get_dirnames(t1_filenames)
        else:
            print(f"ERROR: invalid arguments: should be one directory: {filenames[0]}")
            sys.exit(1)
    elif (n == 2):
        if os.path.isdir(filenames[0]) and os.path.isdir(filenames[1]):
            t1_input_dir_root = clean_dirname(filenames[0])
            flair_input_dir_root = t1_input_dir_root
            output_dir_root = clean_dirname(filenames[1])
            t1_filenames = find_files(t1_input_dir_root, patternT1, recursive)
            flair_filenames = find_files(flair_input_dir_root, patternFLAIR, recursive) #TODO: or should we first search next to the T1 image ???? BIDS ???
            output_dirs = make_output_dirs(t1_input_dir_root, t1_filenames, output_dir_root)
        elif os.path.isfile(filenames[0]) and os.path.isfile(filenames[1]):
            t1_filenames = [filenames[0]]
            flair_filenames = [filenames[1]]
            t1_input_dir_root = get_input_dir_root(t1_filenames)
            flair_input_dir_root = get_input_dir_root(flair_filenames)
            output_dirs = get_dirnames(t1_filenames)
            output_dir_root = ""
        else:
            print(f"ERROR: invalid arguments: should be two directories or two filenames: {filenames[0]} {filenames[1]}")
            sys.exit(1)
    else:
        print("n=",n) #DEBUG !!!!!!!!!!!!!!!!!!!!!
        if (n&1 == 1): #odd
            if os.path.isdir(filenames[-1]) and are_only_files(filenames[:-1]):
                output_dir_root = clean_dirname(filenames[-1])
                output_dirs = get_dirnames(t1_filenames)
                t1_filenames = filenames[0:-1:2]
                flair_filenames = filenames[1:-1:2]
                t1_input_dir_root = get_input_dir_root(t1_filenames)
                flair_input_dir_root = get_input_dir_root(flair_filenames)
                output_dirs = make_output_dirs(t1_input_dir_root, t1_filenames, output_dir_root)
            elif (n == 3) and os.path.isdir(filenames[0]) and os.path.isdir(filenames[1]) and os.path.isdir(filenames[2]):
                t1_input_dir_root = clean_dirname(filenames[0])
                flair_input_dir_root = clean_dirname(filenames[1])
                output_dir_root = clean_dirname(filenames[2])
                t1_filenames = find_files(t1_input_dir_root, patternT1, recursive)
                flair_filenames = find_files(flair_input_dir_root, patternFLAIR, recursive) #TODO: or should we first search next to the T1 image ???? BIDS ???
                output_dirs = make_output_dirs(t1_input_dir_root, t1_filenames, output_dir_root)
            else:
                print(f"ERROR: invalid arguments: should be two directories or two filenames 1 ") #TODO
                sys.exit(1)                    
        else: #even
            if are_only_files(filenames):
                output_dir_root = ""
                output_dirs = get_dirnames(t1_filenames)
                t1_filenames = filenames[0::2]
                flair_filenames = filenames[1::2]
                t1_input_dir_root = get_input_dir_root(t1_filenames)
                flair_input_dir_root = get_input_dir_root(flair_filenames) #TODO:useless ???
                output_dirs = get_dirnames(t1_filenames)
            else:
                print(f"ERROR: invalid arguments: should be two directories or two filenames 2 ") #TODO
                sys.exit(1)

    if (len(t1_filenames) != len(flair_filenames)):
        print(f"ERROR: did not find the same number of T1 files and FLAIR files ({len(t1_filenames)} vs {len(flair_filenames)})")
        sys.exit(1)
    
    return t1_input_dir_root, output_dir_root, t1_filenames, flair_filenames, output_dirs
# def get_files_and_output_dirs(filenames, patternT1, patternFLAIR, recursive):
#     t1_input_dir_root = ""
#     flair_input_dir_root = ""
#     output_dir_root = ""
#     t1_filenames = []
#     flair_filenames = []
#     output_dirs = []
#     #print("filenames=", filenames)
#     filenames = [os.path.abspath(filename) for filename in filenames]
#     #print("filenames=", filenames)
#     n = len(filenames)
#     if (n < 1):
#         print("ERROR: invalid number of filenames. It should be \"T1_filename FLAIR_FILENAME [T1_filename2 FLAIR_filename2 ...] [output_dir]\" or \"input_dir [output_dir]\" or \"T1_input_dir FLAIR_input_dir output_dir\"")
#         sys.exit(1)
#     elif (n == 1):
#         if os.path.isdir(filenames[0]):
#             t1_input_dir_root = clean_dirname(filenames[0])
#             flair_input_dir_root = t1_input_dir_root
#             output_dir_root = t1_input_dir_root
#             t1_filenames = find_files(t1_input_dir_root, patternT1, recursive)
#             flair_filenames = find_files(flair_input_dir_root, patternFLAIR, recursive)
#             output_dirs = get_dirnames(t1_filenames)
#         else:
#             print(f"ERROR: invalid arguments: should be one directory: {filenames[0]}")
#             sys.exit(1)
#     elif (n == 2):
#         if os.path.isdir(filenames[0]) and os.path.isdir(filenames[1]):
#             t1_input_dir_root = clean_dirname(filenames[0])
#             flair_input_dir_root = t1_input_dir_root
#             output_dir_root = clean_dirname(filenames[1])
#             t1_filenames = find_files(t1_input_dir_root, patternT1, recursive)
#             flair_filenames = find_files(flair_input_dir_root, patternFLAIR, recursive) #TODO: or should we first search next to the T1 image ???? BIDS ???
#             output_dirs = make_output_dirs(t1_input_dir_root, t1_filenames, output_dir_root)
#         elif os.path.isfile(filenames[0]) and os.path.isfile(filenames[1]):
#             t1_filenames = [filenames[0]]
#             flair_filenames = [filenames[1]]
#             t1_input_dir_root = get_input_dir_root(t1_filenames)
#             flair_input_dir_root = get_input_dir_root(flair_filenames)
#             output_dirs = get_dirnames(t1_filenames)
#             output_dir_root = ""
#         else:
#             print(f"ERROR: invalid arguments: should be two directories or two filenames: {filenames[0]} {filenames[1]}")
#             sys.exit(1)
#     else:
#         print("n=",n) #DEBUG !!!!!!!!!!!!!!!!!!!!!
#         if (n&1 == 1): #odd
#             if os.path.isdir(filenames[-1]) and are_only_files(filenames[:-1]):
#                 output_dir_root = clean_dirname(filenames[-1])
#                 output_dirs = get_dirnames(t1_filenames)
#                 t1_filenames = filenames[0:-1:2]
#                 flair_filenames = filenames[1:-1:2]
#                 t1_input_dir_root = get_input_dir_root(t1_filenames)
#                 flair_input_dir_root = get_input_dir_root(flair_filenames)
#                 output_dirs = make_output_dirs(t1_input_dir_root, t1_filenames, output_dir_root)
#             elif (n == 3) and os.path.isdir(filenames[0]) and os.path.isdir(filenames[1]) and os.path.isdir(filenames[2]):
#                 t1_input_dir_root = clean_dirname(filenames[0])
#                 flair_input_dir_root = clean_dirname(filenames[1])
#                 output_dir_root = clean_dirname(filenames[2])
#                 t1_filenames = find_files(t1_input_dir_root, patternT1, recursive)
#                 flair_filenames = find_files(flair_input_dir_root, patternFLAIR, recursive) #TODO: or should we first search next to the T1 image ???? BIDS ???
#                 output_dirs = make_output_dirs(t1_input_dir_root, t1_filenames, output_dir_root)
#             else:
#                 print(f"ERROR: invalid arguments: should be two directories or two filenames 1 ") #TODO
#                 sys.exit(1)                    
#         else: #even
#             if are_only_files(filenames):
#                 output_dir_root = ""
#                 output_dirs = get_dirnames(t1_filenames)
#                 t1_filenames = filenames[0::2]
#                 flair_filenames = filenames[1::2]
#                 t1_input_dir_root = get_input_dir_root(t1_filenames)
#                 flair_input_dir_root = get_input_dir_root(flair_filenames) #TODO:useless ???
#                 output_dirs = get_dirnames(t1_filenames)
#             else:
#                 print(f"ERROR: invalid arguments: should be two directories or two filenames 2 ") #TODO
#                 sys.exit(1)
                

            
#     # elif (n == 3): #TODO: tester pair ou impair !!!
#     #     if os.path.isdir(filenames[0]) and os.path.isdir(filenames[1]) and os.path.isdir(filenames[1]):
#     #         t1_input_dir_root = clean_dirname(filenames[0])
#     #         flair_input_dir_root = t1_input_dir_root
#     #         output_dir_root = clean_dirname(filenames[1])
#     #         t1_filenames = find_files(t1_input_dir_root, patternT1, recursive)
#     #         flair_filenames = find_files(flair_input_dir_root, patternFLAIR, recursive) #TODO: or should we first search next to the T1 image ???? BIDS ???
#     #         output_dirs = make_output_dirs(t1_input_dir_root, t1_filenames, output_dir_root)
#     #     elif os.path.isfile(filenames[0]) and os.path.isfile(filenames[1]):
#     #         t1_filenames = [filenames[0]]
#     #         flair_filenames = [filenames[1]]
#     #         t1_input_dir_root = get_input_dir_root(t1_filenames)
#     #         flair_input_dir_root = get_input_dir_root(flair_filenames)
#     #         output_dirs = get_dirnames(t1_filenames)
#     #         output_dir_root = ""
#     #     else:
#     #         print(f"ERROR: invalid arguments: should be two directories or two filenames: {filenames[0]} {filenames[1]}")
#     #         sys.exit(1)
#     # elif (n > 2):
#     #     if os.path.isdir(filenames[-1]):
#     #         output_dir_root = clean_dirname(filenames[-1])
#     #         t1_filenames = filenames[0:-1]
#     #         t1_input_dir_root = get_input_dir_root(t1_filenames)
#     #         output_dirs = make_output_dirs(t1_input_dir_root, t1_filenames, output_dir_root)
#     #     else:
#     #         t1_filenames = filenames #filenames[0:-1]
#     #         output_dirs = get_dirnames(t1_filenames)
#     #         t1_input_dir_root = get_input_dir_root(t1_filenames)
#     #         output_dir_root = ""

#     if (len(t1_filenames) != len(flair_filenames)):
#         print(f"ERROR: did not find the same number of T1 files and FLAIR files ({len(t1_filenames)} vs {len(flair_filenames)})")
#         sys.exit(1)
    
#     return t1_input_dir_root, output_dir_root, t1_filenames, flair_filenames, output_dirs


#stringify list of strings
#that is transform l=['file1', 'file2', 'file3'] in s="'file1' 'file2' 'file3'"
def stringify(l):
        if not l:
                return ""
        s = "\"{}\"".format(l[0])
        for i in range(1,len(l)):
                s += " \"{}\"".format(l[i])
        return s

#stringify list of tuples of strings
def stringifyT(l):
        if not l:
                return ""
        s = "\"{}\" \"{}\" \"{}\"".format((l[0])[0], (l[0])[1], (l[0])[2])
        for i in range(1,len(l)):
                s += " \"{}\" \"{}\" \"{}\"".format((l[i])[0], (l[i])[1], (l[i])[2])
        return s


# merge two lists in one list of tuples
def merge_lists(l1, l2, l3):
    return list(map(lambda x, y, z: (x,y,z), l1, l2, l3))


def get_extension(filename):
    for e in [".nii", ".nii.gz"]:
        if (filename.endswith(e)):
            return e
    _, e = os.path.splitext(filename)
    return e


                


def printTime(timeInSeconds, num_processed_files, num_files):
        D=int(timeInSeconds//(60*60*24))
        H=int((timeInSeconds//(60*60))%24)
        M=int((timeInSeconds//60)%60)
        S=timeInSeconds%60
        timeStr=""
        if (D > 0):
                timeStr+=f"{D}d"
        if (H > 0):
                timeStr+=f"{H}h"
        if (M > 0):
                timeStr+=f"{M}m"
        timeStr+=f"{S:.2f}s"
        print("")
        if (num_files > 1):
            print(f"Total time={timeInSeconds:.2f}s => {timeStr} for {num_processed_files}/{num_files} files ({timeInSeconds/num_files:.2f}s/file)")
        else:
            print(f"Total time={timeInSeconds:.2f}s => {timeStr} for {num_processed_files}/{num_files} files")



def gunzip_if_needed(filename):
    if (filename.endswith(".gz")):
        basename = filename[:-3]  #remove ".gz"
        #tmpFilename = os.path.join(TMP_DIR, basename)
        tmpFilename = os.path.join(os.path.dirname(filename), basename)
        run_command(f"gunzip -c {filename} > {tmpFilename}")
    else:
        tmpFilename = filename
    return tmpFilename


def gzip_if_needed(filename, uncompressed_filename):
    if (filename.endswith(".gz")):
        os.remove(uncompressed_filename)
    else:
        run_command(f"gzip -9 -f {filename}")
        filename += ".gz"
    return filename



def assemblyNetLesionPipeline_process(filenames,
                                      patternT1, patternFLAIR,
                                      recursive,
                                      global_csv_filename,
                                      no_pdf_report,
                                      age_sex_csv,
                                      age,
                                      sex,
                                      batchSize,
                                      platform=False):
    
    tt0 = time.time()


    t1_input_dir_root, output_dir_root, t1_filenames, flair_filenames, output_dirs = get_files_and_output_dirs(filenames, patternT1, patternFLAIR, recursive)

    print("------------------")
    print("t1_input_dir_root=", t1_input_dir_root)
    print("output_dir_root=", output_dir_root)
    print("t1_filenames=", t1_filenames)
    print("flair_filenames=", flair_filenames)
    print("num t1_filenames=", len(t1_filenames))
    print("num flair_filenames=", len(flair_filenames))
    # print("output_dirs=", output_dirs)
    assert(len(t1_filenames) == len(output_dirs))
    assert(len(t1_filenames) == len(flair_filenames))

    options=""
    if not global_csv_filename is None:
        #global_csv_filename = os.path.join(output_dir_root, os.path.basename(global_csv_filename))
        if os.path.exists(global_csv_filename):
            print("ERROR: ouput file already exists:", global_csv_filename) 
            sys.exit(1)
        options += f"-root-dir {t1_input_dir_root} -global-csv {global_csv_filename}"
    if no_pdf_report:
        options += f" -no-pdf-report"
    #print("options=", options)
    if not age_sex_csv is None:
        #age_sex_csv_filename = os.path.join(output_dir_root, os.path.basename(age_sex_csv))
        age_sex_csv_filename = age_sex_csv
        if (not os.path.dirname(age_sex_csv_filename)):
            age_sex_csv_filename = os.path.join(output_dir_root, os.path.basename(age_sex_csv))
        if not os.path.exists(age_sex_csv_filename):
            print("ERROR: input file does not exist:", age_sex_csv_filename) 
            sys.exit(1)
        options += f" -age-sex-csv {age_sex_csv_filename}"
    else:
        #age = age
        #sex = sex
        options += f" -age {age} -sex {sex}"
    #batchSize = batchSize
    #platform = platform
    #print("------------------")

    #TMP_DIR="/tmp"
    TMP_DIR = tempfile.mkdtemp(dir="/tmp") #usefule if several "singularity" images run with the same mounted /tmp directory.
    print(f"{TMP_DIR} used as temporary directory")
    
    options += f" -tmp-dir {TMP_DIR}" 

    print("options=", options)


    # tt0 = time.time()


    # t1_input_dir_root, output_dir_root, t1_filenames, flair_filenames, output_dirs = get_files_and_output_dirs(filenames, patternT1, patternFLAIR, recursive)

    # print("------------------")
    # print("t1_input_dir_root=", t1_input_dir_root)
    # print("output_dir_root=", output_dir_root)
    # print("t1_filenames=", t1_filenames)
    # print("flair_filenames=", flair_filenames)
    # print("num t1_filenames=", len(t1_filenames))
    # print("num flair_filenames=", len(flair_filenames))
    # # print("output_dirs=", output_dirs)
    # assert(len(t1_filenames) == len(output_dirs))
    # assert(len(t1_filenames) == len(flair_filenames))

    
    # options=""
    # if not global_csv_filename is None:
    #     #global_csv_filename = os.path.join(output_dir_root, os.path.basename(global_csv_filename))
    #     if os.path.exists(global_csv_filename):
    #         print("ERROR: ouput file already exists:", global_csv_filename) 
    #         sys.exit(1)
    #     options += f"-root-dir {t1_input_dir_root} -global-csv {global_csv_filename}"
    # if no_pdf_report:
    #     options += f" -no-pdf-report"
    # #print("options=", options)
    # if not age_sex_csv is None:
    #     #age_sex_csv_filename = os.path.join(output_dir_root, os.path.basename(age_sex_csv))
    #     age_sex_csv_filename = age_sex_csv
    #     if (not os.path.dirname(age_sex_csv_filename)):
    #         age_sex_csv_filename = os.path.join(output_dir_root, os.path.basename(age_sex_csv))
    #     if not os.path.exists(age_sex_csv_filename):
    #         print("ERROR: input file does not exist:", age_sex_csv_filename) 
    #         sys.exit(1)
    #     options += f" -age-sex-csv {age_sex_csv_filename}"
    # else:
    #     #age = age
    #     #sex = sex
    #     options += f" -age {age} -sex {sex}"
    # #batchSize = batchSize
    # #platform = platform
    # #print("------------------")

    
    # #TMP_DIR="/tmp"
    # TMP_DIR = tempfile.mkdtemp(dir="/tmp") #useful if several "singularity" images run with the same mounted /tmp directory.
    # print(f"{TMP_DIR} used as temporary directory")
    
    # options += f" -tmp-dir {TMP_DIR}" 

    pwd = "/opt/assemblyNetLesion" # os.getcwd() # singularity needs an absolute path!
    os.chdir(pwd)
    run_command(f"pwd") #DEBUG
    run_command(f"ls -l .") #DEBUG
    print("------------------------------")
    num_files = len(t1_filenames)
    num_processed_files = 0
    for i in range(0, num_files, batchSize):

            iend = min(num_files, i+batchSize)

            batch_output_dirs = output_dirs[i:iend]
            batch_original_t1_filenames = []
            batch_tmp_t1_filenames = []
            batch_original_flair_filenames = []
            batch_tmp_flair_filenames = []
            for j in range(i, iend):
                t1_filename = t1_filenames[j]
                flair_filename = flair_filenames[j]
                t1_ext = get_extension(t1_filename)
                flair_ext = get_extension(flair_filename)
                tmp_t1_filename = os.path.join(TMP_DIR, "t1_"+str(j)+t1_ext)
                tmp_flair_filename = os.path.join(TMP_DIR, "flair_"+str(j)+flair_ext)
                batch_tmp_t1_filenames.append(tmp_t1_filename)
                shutil.copyfile(t1_filename, tmp_t1_filename)
                batch_original_t1_filenames.append(t1_filename)
                batch_tmp_flair_filenames.append(tmp_flair_filename)
                shutil.copyfile(flair_filename, tmp_flair_filename)
                batch_original_flair_filenames.append(flair_filename)

            assert(len(batch_original_t1_filenames) == len(batch_tmp_t1_filenames))
            assert(len(batch_original_t1_filenames) == len(batch_output_dirs))
            assert(len(batch_original_flair_filenames) == len(batch_tmp_flair_filenames))
            assert(len(batch_original_flair_filenames) == len(batch_original_t1_filenames))


            #Impl: we don't let DeepLesionBrain code do the preprocessing
            # as it is not in the same directory anymore

            #TODO: check all files have been correctly processed !!!
            
            #Preprocessing
            os.chdir("Compilation_lesionBrain_v11_fullpreprocessing")
            for j in range(i, iend):
                t1_filename = batch_tmp_t1_filenames[j-i]
                tmp_t1_filename = gunzip_if_needed(t1_filename)
                flair_filename = batch_tmp_flair_filenames[j-i]
                tmp_flair_filename = gunzip_if_needed(flair_filename)
                
                run_command(f"./lesionBrain_v11_fullpreprocessing_exe.sh {tmp_t1_filename} {tmp_flair_filename}")

                t1_filename = gzip_if_needed(t1_filename, tmp_t1_filename)
                batch_tmp_t1_filenames[j-i] = t1_filename
                flair_filename = gzip_if_needed(flair_filename, tmp_flair_filename)
                batch_tmp_flair_filenames[j-i] = flair_filename
            os.chdir(pwd)

            #TODO: compress produced files that we need
            # and remove remaining useless files 
            run_command(f"ls -l {TMP_DIR}") #DEBUG
            print("------------------------------")
            
            os.chdir("Segmentation/DeepLesionBrain/")
            for j in range(i, iend):
                #To do only of the output files were correctly generated !
                t1_filename = batch_tmp_t1_filenames[j-i]
                flair_filename = batch_tmp_flair_filenames[j-i]
                run_command(f"python3 -u end_to_end_pipeline_file.py {t1_filename} {flair_filename}")
            os.chdir(pwd)
            #TODO: we should process the whole batch in one call to python3 !!!!
            #TODO: rename end_to_end_pipeline en segment_lesions.py 

            print("### after lesion segmentation")
            run_command(f"ls -l {TMP_DIR}") #DEBUG
            print("------------------------------")

            
            # ????
            #From now on, all manipulated files are .nii.gz


            #TODO
            # # check that preprocessing produced some files
            # btf = []
            # bod = []
            # bof = []
            # for i in range(len(batch_tmp_filenames)):
            #     filename = os.path.join(os.path.dirname(batch_tmp_filenames[i]),
            #                             "n_mmni_f"+os.path.basename(batch_tmp_filenames[i]))
            #     if os.path.exists(filename):
            #         btf.append(batch_tmp_filenames[i])
            #         bof.append(batch_original_filenames[i])
            #         bod.append(batch_output_dirs[i])
            # if not btf:
            #     #no file to process for this batch
            #     continue
            # batch_tmp_filenames = btf
            # batch_original_filenames = bof
            # batch_output_dirs = bod
            # assert(len(batch_original_filenames) == len(batch_tmp_filenames))
            # assert(len(batch_original_filenames) == len(batch_output_dirs))
            
            
            #files_and_dirs = stringifyT(merge_lists(t1_filenames[i:iend], output_dirs[i:iend]))

            os.chdir("Inpainting/")
            run_command(f"python3 -u doNonBlindInpainting.py {stringify(batch_tmp_t1_filenames)}")
            os.chdir(pwd)

            print("### after inpainting")
            run_command(f"ls -l {TMP_DIR}") #DEBUG
            print("------------------------------")
            
            #RegQCNet
            os.chdir("QualityControl/DeepQCReg/")
            run_command(f"python3 -u deepQCReg.py {stringify(batch_tmp_t1_filenames)}")
            os.chdir(pwd)

            print("### after RegQCNet")
            run_command(f"ls -l {TMP_DIR}") #DEBUG
            print("------------------------------")
            
            #DeepICE
            # os.chdir("Segmentation/DeepICE")
            # cmd = "python3 -u DeepICE.py {}".format(stringify(batch_tmp_t1_filenames))
            # run_command(cmd)
            # os.chdir(pwd)

            for filename in batch_tmp_t1_filenames:
                dirname = os.path.dirname(filename)
                basename = os.path.basename(filename)
                os.replace(os.path.join(dirname, "mni_t1_"+basename), os.path.join(dirname, "orig_mni_t1_"+basename)) #DEBUG only?
                os.replace(os.path.join(dirname, "syn_mni_t1_"+basename), os.path.join(dirname, "mni_t1_"+basename))
            
            
            #DeepReg
            os.chdir("Registration/DeepREG/version2/")
            run_command(f"python3 -u DeepReg.py {stringify(batch_tmp_t1_filenames)}")
            os.chdir(pwd)

            print("### after DeepReg")
            run_command(f"ls -l {TMP_DIR}") #DEBUG
            print("------------------------------")
            

            #AssemblyNet
            os.chdir("Segmentation/AssemblyNET")
            run_command(f"python3 -u segment.py {stringify(batch_tmp_t1_filenames)}")

            print("### after AssemblyNet")
            run_command(f"ls -l {TMP_DIR}") #DEBUG
            print("------------------------------")
            print("batch_tmp_t1_filenames=", batch_tmp_t1_filenames)
            print("batch_tmp_flair_filenames=", batch_tmp_flair_filenames)
            print("batch_original_t1_filenames=", batch_original_t1_filenames)
            
            run_command(f"python3 -u make_reports.py {options} {stringifyT(merge_lists(batch_tmp_t1_filenames, batch_tmp_flair_filenames, batch_original_t1_filenames))}")
            
            #TODO: report
            #TODO: preview if platform
            #TODO: clean useless files
            
            # cmd = "python3 -u make_reports.py {} {}".format(options, stringifyT(merge_lists(batch_tmp_filenames, batch_original_filenames)))
            # print("cmd=", cmd)
            # run_command(cmd)

            # #TODO: transform to native space (has to be done where antsApplyTransforms is available !!!)

            # if platform:
            #     cmd = "python3 -u save_preview.py {}".format(stringify(batch_tmp_filenames))
            #     run_command(cmd)            

            os.chdir(pwd)

            # move/rename/copy/remove to get final files with correct names
            
            # prefixes = ["mni_lobes_", "mni_macrostructures_", "mni_structures_", "mni_tissues_", "mni_mask_", "mni_t1_", "native_lobes_", "native_macrostructures_", "native_structures_", "native_tissues_", "native_mask_", "native_t1_"]
            # if platform:
            #     prefixes.append("preview_mni_t1_")

            # prefixes_suffixes = [("matrix_affine_native_to_mni_", ".txt")]
            # if not no_pdf_report:
            #     prefixes_suffixes.append(("report_", ".pdf"))
            # if global_csv_filename is None:
            #     prefixes_suffixes.append(("report_", ".csv"))

            # for i in range(len(batch_tmp_filenames)):
            #     tmp_filename = batch_tmp_filenames[i]
            #     original_filename = replace_extensions(batch_original_filenames[i], [".nii"], ".nii.gz")
            #     output_dir = batch_output_dirs[i]

            #     d1 = os.path.dirname(tmp_filename)
            #     b1 = os.path.basename(tmp_filename)
            #     b2 = os.path.basename(original_filename)

            #     if platform:
            #         #copy
            #         src = os.path.join(d1, "mni_structures_"+b1)
            #         assert(os.path.isfile(src))
            #         dst = os.path.join(output_dir, "preview_mni_structures_"+b2)
            #         shutil.copyfile(src, dst)
            #         #copy
            #         src = tmp_filename
            #         assert(os.path.isfile(src))
            #         dst = os.path.join(output_dir, "original_t1_"+b2)
            #         shutil.copyfile(src, dst)                    
                
            #     for prefix in prefixes:
            #         src = os.path.join(d1, prefix+b1)
            #         assert(os.path.isfile(src))
            #         dst = os.path.join(output_dir, prefix+b2)
            #         #shutil.copyfile(src, dst)
            #         shutil.move(src, dst)

            #     b1we = replace_extensions(b1, [".nii", ".nii.gz"], "")
            #     b2we = replace_extensions(b2, [".nii", ".nii.gz"], "")
            #     for prefix, suffix in prefixes_suffixes:
            #         src = os.path.join(d1, prefix+b1we+suffix)
            #         assert(os.path.isfile(src))
            #         dst = os.path.join(output_dir, prefix+b2we+suffix)
            #         #shutil.copyfile(src, dst)
            #         shutil.move(src, dst)

            #     #remove tmp files 
            #     os.remove(tmp_filename)

            num_processed_files += len(batch_tmp_t1_filenames)

            
    # TODO: copy README 
                
    # if (num_processed_files > 0):
    #     shutil.copyfile("Segmentation/AssemblyNET/README.pdf", os.path.join(output_dir_root, "README.pdf"))

    tt1 = time.time()

    printTime(tt1-tt0, num_processed_files, num_files)

    shutil.rmtree(TMP_DIR)
