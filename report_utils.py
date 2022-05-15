import os
# import sys
import datetime
import csv
import numpy as np
import nibabel as nii
from string import Template
from PIL import Image
from skimage import filters
from scipy.ndimage.morphology import binary_fill_holes, binary_dilation, binary_erosion
from scipy.ndimage.measurements import center_of_mass
from skimage.measure import label
from matplotlib import pyplot as plt
import numpy as np
import pickle
from utils import run_command, stringify, replace_extension, replace_extensions
import pandas as pd
import math

OUT_HEIGHT = 217
DEFAULT_ALPHA = 0.5


nii.Nifti1Header.quaternion_threshold = -8e-07
version = '1.0'
release_date = datetime.datetime.strptime("30-07-2021", "%d-%m-%Y").strftime("%d-%b-%Y")


# # RGB
# colormap = {}
# colormap[0] = [0, 0, 0]
# colormap[1] = [255, 0, 0]
# colormap[2] = [0, 255, 0]
# colormap[3] = [0, 0, 255]


# def save_obj(obj, name):
#     with open(name + '.pkl', 'wb') as f:
#         pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


# def load_obj(name):
#     with open(name + '.pkl', 'rb') as f:
#         return pickle.load(f)

def get_structures(hemi_filename, structures_sym_filename):
    hemi = nii.load(hemi_filename).get_data()
    structures_sym = nii.load(structures_sym_filename).get_data() 
    structures = np.zeros(hemi.shape)
    
    structures[np.logical_and((hemi==1) , (structures_sym==9))]=15
    structures[np.logical_and((hemi==1) , (structures_sym==8))]=13
    structures[np.logical_and((hemi==1) , (structures_sym==7))]=11
    structures[np.logical_and((hemi==1) , (structures_sym==6))]=9
    structures[np.logical_and((hemi==1) , (structures_sym==5))]=7
    structures[np.logical_and((hemi==1) , (structures_sym==4))]=5
    structures[np.logical_and((hemi==1) , (structures_sym==3))]=3
    structures[(structures_sym==1)]=1
    
    structures[np.logical_and((hemi==2) , (structures_sym==9))]=16
    structures[np.logical_and((hemi==2) , (structures_sym==8))]=14
    structures[np.logical_and((hemi==2) , (structures_sym==7))]=12
    structures[np.logical_and((hemi==2) , (structures_sym==6))]=10
    structures[np.logical_and((hemi==2) , (structures_sym==5))]=8
    structures[np.logical_and((hemi==2) , (structures_sym==4))]=6
    structures[np.logical_and((hemi==2) , (structures_sym==3))]=4
    structures[(structures_sym==2)]=2

    structures_filename = structures_sym_filename.replace('_sym', '').replace('.nii', '.nii.gz')
    array_img = nii.Nifti1Image(structures.astype('uint8'), nii.load(hemi_filename).affine)
    array_img.to_filename(structures_filename)
    return structures_filename


def get_lesion_by_regions(fname, fname_crisp, fname_hemi, fname_lab, fname_lesion):

    juxtacortical_idx = 3
    deepwhite_idx = 2
    periventricular_idx = 1
    cerebelar_idx = 4
    medular_idx = 5

    T1_img = nii.load(fname)
    crisp = nii.load(fname_crisp).get_data()
    hemi = nii.load(fname_hemi).get_data()
    lab = nii.load(fname_lab).get_data()
    lesion = nii.load(fname_lesion).get_data()

    ventricles = (lab == 1) + (lab == 2)
    cond1 = (crisp == 3)
    cond2 = (lesion > 0)
    structure = np.ones([5, 5, 5])
    cond3 = binary_dilation((lab > 0), structure)
    wm_filled = binary_fill_holes((cond1.astype('int') + cond2.astype('int') + cond3.astype('int')) > 0).astype('int')*6

    wm = (((crisp == 3) + (lesion > 0)) > 0).astype('int')

    SE = np.zeros([5, 5, 5])  # 3 mm distance
    for i in range(5):
        for j in range(5):
            for k in range(5):
                if(((((i-3)**2)+((j-3)**2)+((k-3)**2))**0.5) < 3):
                    SE[i, j, k] = 1

    periventricular = wm_filled*binary_dilation(ventricles, SE)
    yuxtacortical = wm_filled-binary_erosion(wm_filled, SE)
    deep = abs(wm_filled-periventricular-yuxtacortical) > 0

    cerebrum = (hemi == 1) + (hemi == 2)
    medular = (hemi == 5)
    cerebelar = ((hemi == 4) + (hemi == 3)) * (crisp == 3)
    # infratenttorial = medular+cerebelar

    regions = np.zeros(crisp.shape, dtype=np.uint8)
    ind = (cerebrum * yuxtacortical) > 0
    regions[ind] = 3
    ind = (cerebrum * deep) > 0
    regions[ind] = 2
    ind = (cerebrum * periventricular) > 0
    regions[ind] = 1
    ind = (cerebelar) > 0
    regions[ind] = 4
    ind = (medular) > 0
    regions[ind] = 5

    # #result
    # region_name = fname_crisp.replace('tissues', 'regions')
    # wm_name = fname_crisp.replace('tissues', 'wmmap')
    # nii.Nifti1Image(regions, T1_img.affine).to_filename(region_name)
    # nii.Nifti1Image(wm, T1_img.affine).to_filename(wm_name)

    # clasification
    seg_labels, seg_num = label(lesion, return_num=True, connectivity=2)

    # Lesion analysis
    lesion2 = np.zeros(lesion.shape, dtype=np.uint8)
    for i in range(1, seg_num+1):
        # Clasification
        ind = (seg_labels == i)
        votes = regions[ind]
        # periventicular
        if((votes == periventricular_idx).sum() > 0):
            lesion2[ind] = periventricular_idx

        # yuxtacortical
        elif((votes == juxtacortical_idx).sum() > 0):
            lesion2[ind] = juxtacortical_idx

        # cerebelar
        elif((votes == cerebelar_idx).sum() > 0):
            lesion2[ind] = cerebelar_idx

        # medular
        elif((votes == medular_idx).sum() > 0):
            lesion2[ind] = medular_idx

        # deep
        else:
            lesion2[ind] = deepwhite_idx

    classified_name = fname.replace('t1', 'lesions')
    array_img = nii.Nifti1Image(lesion2, T1_img.affine)
    array_img.set_data_dtype(lesion2.dtype)
    array_img.to_filename(classified_name)
    return classified_name  # , region_name, wm_name   #B:TODO: return lesion2 ???? to avoid to reload it !!!


def compute_volumes(im, labels, scale):
    assert(type(labels) is list)
    vols = []
    for ll in labels:
        v = 0
        if not type(ll) is list:
            ll = [ll]
        for l in ll:
            a = (im == l)
            vl = np.sum(a[:])
            # print("l=", l, " -> volume=", vl)
            v += vl
        # print("==> ll=", ll, " -> total volume=", v)
        vols.extend([(v*scale)/1000])
        #vols.extend([(v*scale)])
    assert(len(vols) == len(labels))
    return vols


def read_info_file(information_filename):
    with open(information_filename, 'r') as f:
        list_file = f.readlines()
        assert(len(list_file)==2)
        assert(list_file[0]=="my_snr_1,my_snr_2,scalet1,orientation_report\n")
        list2 = list_file[1].split(",")
        assert(len(list2) == 4)
        snr = float(list2[0])
        scale = float(list2[2])
        orientation_report = capitalize(list2[3])
        return snr, scale, orientation_report


def save_tissue_plots(age, bounds_all_df, tissue_vol, vol_ice):
    assert(not bounds_all_df.empty)
    structure = ['White matter', 'Grey matter', 'Cerebrospinal fluid']
    filenames = ['WM.png', 'GM.png', 'CSF.png']
    tissue_vol_indices = [2, 1, 0]  # in tissue_vol, order is CSF, GM, WM
    csv_tissue_names = ['Tissue WM cm3', 'Tissue GM cm3', 'Tissue CSF cm3'] #'Tissue IC cm3'
    
    plt.rcParams.update({'font.size': 40})
    for i in range(len(structure)):
        name_csv = csv_tissue_names[i] # get_tissue_name_in_bounds_csv(structure[i])
        bounds = (bounds_all_df[bounds_all_df["structure"] == name_csv]).reset_index()
        lb = (bounds.iloc[:, 2::2].to_numpy(dtype=np.float64))[0]
        ub = (bounds.iloc[:, 3::2].to_numpy(dtype=np.float64))[0]
        plt.figure(figsize=(20, 13))
        plt.fill_between(np.arange(101), ub, lb, color=['lightgreen'])
        plt.plot(np.arange(101), (ub + lb)/2, 'b--', linewidth=3)
        plt.title(structure[i], fontweight='bold')
        plt.xlabel('Age (years)')
        plt.ylabel('Volume (%)')
        if(not str(age).lower() == 'unknown'):
            plt.scatter([int(age)], [int(100*tissue_vol[tissue_vol_indices[i]]/vol_ice)], s=300, c='red')
        plt.savefig(filenames[i], dpi=300, bbox_inches = 'tight', pad_inches=0.1)
        plt.clf()
    return filenames


# def get_expected_volumes(age, bounds_df):
#     structure = ['White matter', 'Grey matter', 'Cerebrospinal fluid']
#     tissue_vol_indices = [2, 1, 0]  # in tissue_vol, order is CSF, GM, WM
#     normal_vol = []
#     if(not age == 'unknown'):
#         for i in range(len(structure)):
#             TODO: use bounds_df   => A FAIRE dans get_tissue_seg() !!!
#             age_inf = math.floor(age)
#             age_sup = math.ceil(age)
#             lower_bound = np.interp(age, [age_inf, age_sup], [y2[age_inf], y2[age_sup]])
#             upper_bound = np.interp(age, [age_inf, age_sup], [y1[age_inf], y1[age_sup]])
#             normal_vol.append([lower_bound, upper_bound])
#     return normal_vol
    
#def get_expected_volumes(age, sex, tissue_vol, vol_ice):
    # plt.rcParams.update({'font.size': 40})
    # if(sex == 'f' or sex == 'femme' or sex == 'woman'):
    #     sex = 'female'
    # if(sex == 'm' or sex == 'homme' or sex == 'man'):
    #     sex = 'male'

    # structure = ['White matter', 'Grey matter', 'Cerebrospinal fluid']
    # filenames = ['WM.png', 'GM.png', 'CSF.png']
    # tissue_vol_indices = [2, 1, 0]  # in tissue_vol, order is CSF, GM, WM
    
    # dataset = load_obj('normal_crisp_volume_by_age')
    # normal_vol = []
    # for i in range(3):
    #     if(sex == 'unknown'):
    #         y1 = (dataset['male'][i]['up'] + dataset['female'][i]['up'])/2
    #         y2 = (dataset['male'][i]['down'] + dataset['female'][i]['down'])/2
    #     else:
    #         y1 = dataset[sex][i]['up']
    #         y2 = dataset[sex][i]['down']
    #     plt.figure(figsize=(20, 13))
    #     plt.fill_between(np.arange(101), y1, y2, color=['lightgreen'])
    #     plt.plot(np.arange(101), (y1 + y2)/2, 'b--', linewidth=3)
    #     plt.title(structure[i], fontweight='bold')
    #     plt.xlabel('Age (years)')
    #     plt.ylabel('Volume (%)')
    #     if(not age == 'unknown'):
    #         plt.scatter([int(age)], [int(100*tissue_vol[tissue_vol_indices[i]]/vol_ice)], s=300, c='red')
    #         #normal_vol.append([y2[int(age)], y1[int(age)]])
    #         age_inf = math.floor(age)
    #         age_sup = math.ceil(age)
    #         lower_bound = np.interp(age, [age_inf, age_sup], [y2[age_inf], y2[age_sup]])
    #         upper_bound = np.interp(age, [age_inf, age_sup], [y1[age_inf], y1[age_sup]])
    #         normal_vol.append([lower_bound, upper_bound])
    #         print(structure[i], "[", lower_bound, ",", upper_bound, "] instead of [", y2[int(age)], ",", y1[int(age)],"]")

    #     plt.savefig(filenames[i], dpi=300, bbox_inches = 'tight', pad_inches=0.1)
    #     plt.clf()
    # return filenames, normal_vol


def save_seg_nii(img, affine, input_filename, prefix):
    output_filename = get_filename(input_filename, prefix)
    OUT_TYPE = np.uint8
    assert(np.max(img) < np.iinfo(OUT_TYPE).max)
    OUT = img.astype(OUT_TYPE)
    array_img = nii.Nifti1Image(OUT, affine)
    array_img.set_data_dtype(OUT_TYPE)
    array_img.to_filename(output_filename)


def make_centered(im, width=256, height=256):
    assert(im.ndim == 3)
    if ((im.shape[0] > width) or (im.shape[1] > height)):
        rw = im.shape[1] / width
        rh = im.shape[0] / height
        newWidth = width
        newHeight = height
        if (rw > rh):
            newWidth = width
            newHeight = int(im.shape[0] / rw)
        else:
            newWidth = int(im.shape[1] / rh)
            newHeight = height
        im = np.array(Image.fromarray(im).resize((newWidth, newHeight), Image.ANTIALIAS))
        
    assert(im.shape[1] <= width)
    assert(im.shape[0] <= height)
    y0 = int(height/2 - im.shape[0]/2)
    x0 = int(width/2 - im.shape[1]/2)
    assert(x0 >= 0 and x0 <= width)
    assert(y0 >= 0 and y0 <= height)
    out = np.zeros((height, width, 3), im.dtype)
    out[y0:y0+im.shape[0], x0:x0+im.shape[1], :] = im
    return out


def make_slice_image(T1_slice):
    assert T1_slice.ndim == 2

    # Put values in [0; 255]
    im = T1_slice * 255.0

    # Add a channels dimension
    im = np.expand_dims(im, axis=-1)

    # Repeat value to have three-channel image
    im = np.tile(im, (1, 1, 3))

    out_im = im.astype(np.uint8)
    return out_im


def make_slice_with_seg_image_with_alpha_blending(T1_slice, LAB_slice, colors, alpha=0.8):
    assert T1_slice.ndim == 2
    assert T1_slice.shape == LAB_slice.shape

    labels = list(np.unique(LAB_slice).astype(int))
    labels.remove(0)  # remove background label

    # Put values in [0; 255]
    im = T1_slice * 255.0

    # image premultiplied by 1-alpha
    aim = im * (1-alpha)

    # Add a channels dimension
    im = np.expand_dims(im, axis=-1)
    aim = np.expand_dims(aim, axis=-1)

    # Repeat value to have three-channel image
    im = np.tile(im, (1, 1, 3))
    aim = np.tile(aim, (1, 1, 3))

    acolors = colors * alpha

    for l in labels:
        im[LAB_slice == l] = aim[LAB_slice == l] + acolors[l, :]

    out_im = im.astype(np.uint8)
    return out_im


def get_patient_id(input_file):
    idStr = replace_extension(input_file, ".pdf", "")  # get_filename(input_file, "", "")
    if len(idStr) > 20:
        idStr = idStr[0:14]+"..."
    return idStr


def getRowColor(i):
    if (i % 2 == 0):
        return "\\rowcolor{white}"
    else:
        return "\\rowcolor{gray!15}"


def compute_lesion_measures(lesion_type, scale, WM_vol):
    lesion_type = lesion_type.astype('int')
    seg_labels, seg_num = label(lesion_type, return_num=True, connectivity=2)
    vol = (compute_volumes(lesion_type, [[1]], scale))[0]
    lesion_burden = (100 * vol) / WM_vol
    return seg_labels, seg_num, vol, lesion_burden


def write_lesions_details(out, name, seg_labels, seg_num, scale, vol_ice, lesion_number):
        if(seg_num > 0):
            # sort lesions by descending volume
            v = []
            for j in range(1, seg_num+1):
                lesion = (seg_labels == j).astype('int')
                pos = center_of_mass(lesion)
                pos = (int(pos[0]), int(pos[1]), int(pos[2]))
                vol = (compute_volumes(lesion, [[1]], scale))[0]
                v.append([vol, pos])
            v = sorted(v, key=lambda vals: vals[0], reverse=True)

            out.write(Template('\n').safe_substitute())
            out.write(Template('\\begin{tabularx}{0.9\\textwidth}{X c c c}\n').safe_substitute())
            out.write(Template(' \\rowcolor{volbrain_blue} {\\bfseries \\textcolor{text_white}{$t Lesions}} & {\\bfseries \\textcolor{text_white}{Absolute vol. ($cm^{3}$)}} & {\\bfseries \\textcolor{text_white}{Normalized vol. (\%)}} & {\\bfseries \\textcolor{text_white}{Position (MNI coord.)}}  \\\\\n').safe_substitute(t=name))
            for j in range(seg_num):
                row_color = getRowColor(j)
                vol, pos = v[j]
                out.write(Template(row_color+'Lesion $p & $g & $a & $d\\\\ \n').safe_substitute(p=lesion_number, g="{:5.4f}".format(vol), a="{:5.4f}".format(vol*100/vol_ice), d=pos))
                lesion_number = lesion_number + 1

            out.write(Template('\\end{tabularx}\n').safe_substitute())
            out.write(Template('\n').safe_substitute())
            out.write(Template('\\vspace*{10pt}\n').safe_substitute())


def write_lesions(out, lesion_types_filename, scale, vol_ice, WM_vol):
    types = ['healthy', 'Periventricular', 'Deepwhite', 'Juxtacortical', 'Cerebellar', 'Medular']
    lesion_mask = nii.load(lesion_types_filename).get_data()

    lesion_number = 1
    all_lesions = []
    _, seg_num, vol_tot, lesion_burden = compute_lesion_measures((lesion_mask > 0), scale, WM_vol)
    all_lesions.append({'count': seg_num, 'volume_abs': vol_tot, 'volume_rel': vol_tot*100/vol_ice, 'burden': lesion_burden})
    for i in range(1, 4):
        lesion_type = (lesion_mask == i).astype('int')
        seg_labels, seg_num, vol, lesion_burden = compute_lesion_measures((lesion_mask == i), scale, WM_vol)
        write_lesions_details(out, types[i], seg_labels, seg_num, scale, vol_ice, lesion_number)
        lesion_number += seg_num
        all_lesions_type = {'count': seg_num, 'volume_abs': vol, 'volume_rel': vol*100/vol_ice, 'burden': lesion_burden}
        all_lesions.append(all_lesions_type)

    seg_labelsC, seg_numC, volC, lesion_burdenC = compute_lesion_measures((lesion_mask == 4), scale, WM_vol)  # Cerebellar
    seg_labelsM, seg_numM, volM, lesion_burdenM = compute_lesion_measures((lesion_mask == 5), scale, WM_vol)  # Medular
    write_lesions_details(out, types[4], seg_labelsC, seg_numC, scale, vol_ice, lesion_number)
    lesion_number += seg_numC
    write_lesions_details(out, types[5], seg_labelsM, seg_numM, scale, vol_ice, lesion_number)
    lesion_number += seg_numM
    # infratentorial
    all_lesions_type = {'count': seg_numC+seg_numM, 'volume_abs': volC+volM, 'volume_rel': (volC+volM)*100/vol_ice, 'burden': lesion_burdenC+lesion_burdenM}
    all_lesions.append(all_lesions_type)
    all_lesions_type = {'count': seg_numC, 'volume_abs': volC, 'volume_rel': volC*100/vol_ice, 'burden': lesion_burdenC}
    all_lesions.append(all_lesions_type)
    all_lesions_type = {'count': seg_numM, 'volume_abs': volM, 'volume_rel': volM*100/vol_ice, 'burden': lesion_burdenM}
    all_lesions.append(all_lesions_type)

    out.write(Template('\n').safe_substitute())
    # out.write(Template('\\vspace*{10pt}\n').safe_substitute())
    return all_lesions


def write_lesion_table(out, lesion_types_filename, colors_lesions, scale, vol_ice, WM_vol):
    types = ['healthy', 'Periventricular', 'Deepwhite', 'Juxtacortical', 'Cerebellar', 'Medular']
    lesion_mask = nii.load(lesion_types_filename).get_data()
    out.write(Template('\n').safe_substitute())
    out.write(Template('\\begin{tabularx}{0.9\\textwidth}{X c c c c}\n').safe_substitute())
    out.write(Template(' \\rowcolor{volbrain_blue} {\\bfseries \\textcolor{text_white}{Lesion} } & {\\bfseries \\textcolor{text_white}{Count}} & {\\bfseries \\textcolor{text_white}{Absolute vol. ($cm^{3}$)} } & {\\bfseries \\textcolor{text_white}{Normalized vol. (\%)} } & {\\bfseries \\textcolor{text_white}{Lesion burden} }\\\\\n').safe_substitute())
    _, seg_num_tot, vol_tot, lesion_burden = compute_lesion_measures((lesion_mask > 0), scale, WM_vol)
    out.write(Template(getRowColor(0)+'Total Lesions & $g & $a & $d & $b\\\\ \n').safe_substitute(g=seg_num_tot, a="{:5.4f}".format(vol_tot), d="{:5.4f}".format(vol_tot*100/vol_ice), b="{:5.4f}".format(lesion_burden)))
    out.write(Template('\\hline\n').safe_substitute())
    for i in range(1, 4):
        _, seg_num, vol, lesion_burden = compute_lesion_measures((lesion_mask == i), scale, WM_vol)
        out.write(Template(getRowColor(i)+'$p & $g & $a & $d & $b\\\\ \n').safe_substitute(p=types[i], g=seg_num, a="{:5.4f}".format(vol), d="{:5.4f}".format(vol*100/vol_ice), b="{:5.4f}".format(lesion_burden)))
    _, seg_numC, volC, lesion_burdenC = compute_lesion_measures((lesion_mask == 4), scale, WM_vol)  # Cerebellar
    _, seg_numM, volM, lesion_burdenM = compute_lesion_measures((lesion_mask == 5), scale, WM_vol)  # Medular 
    # Infratentorial = Cerebellar + Medular
    seg_numI = seg_numC + seg_numM
    volI = volC + volM
    lesion_burdenI = lesion_burdenC + lesion_burdenM
    out.write(Template(getRowColor(4)+'$p & $g & $a & $d & $b\\\\\n').safe_substitute(p='Infratentorial', g=seg_numI, a="{:5.4f}".format(volI), d="{:5.4f}".format(volI*100/vol_ice), b="{:5.4f}".format(lesion_burdenI)))
    if seg_numI > 0:
        out.write(Template('\\hline\n').safe_substitute())
        out.write(Template(getRowColor(5)+'\quad $p & $g & $a & $d & $b\\\\ \n').safe_substitute(p=types[4], g=seg_numC, a="{:5.4f}".format(volC), d="{:5.4f}".format(volC*100/vol_ice), b="{:5.4f}".format(lesion_burdenC)))
        out.write(Template(getRowColor(6)+'\quad $p & $g & $a & $d & $b\\\\ \n').safe_substitute(p=types[5], g=seg_numM, a="{:5.4f}".format(volM), d="{:5.4f}".format(volM*100/vol_ice), b="{:5.4f}".format(lesion_burdenM)))

    out.write(Template('\\end{tabularx}\n').safe_substitute())
    out.write(Template('\n').safe_substitute())
# B:TODO: Are these values in the CSV ????


def load_latex_packages(out):
    out.write(Template('\\documentclass[10pt,a4paper,oneside,notitlepage]{article}\n').safe_substitute())
    out.write(Template('\\usepackage{color}\n').safe_substitute())
    out.write(Template('\\usepackage[table,usenames,dvipsnames]{xcolor}\n').safe_substitute())
    out.write(Template('\\usepackage{mathptmx}\n').safe_substitute())
    out.write(Template('\\usepackage[T1]{fontenc}\n').safe_substitute())
    out.write(Template('\\usepackage[english]{babel}\n').safe_substitute())
    out.write(Template('\\usepackage{graphicx}\n').safe_substitute())
    out.write(Template('\\usepackage[cm]{fullpage}\n').safe_substitute())
    out.write(Template('\\usepackage{tabularx}\n').safe_substitute())
    out.write(Template('\\usepackage{array}\n').safe_substitute())
    out.write(Template('\\usepackage{multirow}\n').safe_substitute())
    out.write(Template('\\usepackage{subfig}\n').safe_substitute())
    out.write(Template('\\usepackage{tikz}\n').safe_substitute())
    out.write(Template('\\usepackage{hyperref}\n').safe_substitute())
    # out.write(Template('\newcolumntype{Y}{>{\centering\arraybackslash}X}').safe_substitute())
    out.write(Template('\n').safe_substitute())


def capitalize(str):
    return str[0].upper() + str[1:]


def get_patient_info(out, basename, gender, age):
    out.write(Template('\n').safe_substitute())
    out.write(Template('\\begin{tabularx}{0.9\\textwidth}{X X X X}\n').safe_substitute())
    out.write(Template('\\rowcolor{volbrain_blue} {\\bfseries \\textcolor{text_white}{Patient ID}} & {\\bfseries \\textcolor{text_white}{Sex}} & {\\bfseries \\textcolor{text_white}{Age}} & {\\bfseries \\textcolor{text_white}{Report Date}} \\\\\n').safe_substitute())
    date = datetime.datetime.now().strftime("%d-%b-%Y")
    ageStr = str(age).upper()
    genderStr = capitalize(gender)
    if genderStr[0] == "U":
        genderStr = "UNKNOWN"
    out.write(Template('$p & $g & $a & $d\\\\ \n').safe_substitute(p=basename, g=genderStr, a=ageStr, d=date))
    out.write(Template('\\end{tabularx}\n').safe_substitute())
    out.write(Template('\n').safe_substitute())
    out.write(Template('\\vspace*{10pt}\n').safe_substitute())


def get_image_info(out, orientation_report, scale, snr):
    out.write(Template('\n').safe_substitute())
    out.write(Template('\\begin{tabularx}{0.9\\textwidth}{X X X X}\n').safe_substitute())
    out.write(Template('\\rowcolor{volbrain_blue} {\\bfseries \\textcolor{text_white}{Image information}} & {\\bfseries \\textcolor{text_white}{Orientation}} & {\\bfseries \\textcolor{text_white}{Scale factor}} & {\\bfseries \\textcolor{text_white}{SNR}} \\\\\n').safe_substitute())
    out.write(Template(' & $o & $sf & $snr\\\\ \n').safe_substitute(o=capitalize(orientation_report), sf="{:5.2f}".format(scale), snr="{:5.2f}".format(snr)))
    out.write(Template('\\end{tabularx}\n').safe_substitute())
    out.write(Template('\n').safe_substitute())
    out.write(Template('\\vspace*{10pt}\n').safe_substitute())


def get_tissue_seg(out, vols_tissue, vol_ice, colors_ice, colors_tissue, age, bounds_df):
    out.write(Template('\\begin{tabularx}{0.9\\textwidth}{X c c}\n').safe_substitute())
    out.write(Template('\\rowcolor{volbrain_blue} {\\bfseries \\textcolor{text_white}{Tissues}} & {\\bfseries \\textcolor{text_white}{Absolute vol. ($cm^{3}$)}} & {\\bfseries \\textcolor{text_white}{Normalized vol. (\%)}}  \\\\\n').safe_substitute())

    tissues_names = np.array(['White matter (including lesions)', 'Grey matter', 'Cerebrospinal fluid'])
    tissue_vol_indices = [2, 1, 0]  # in tissue_vol, order is CSF, GM, WM

    csv_tissue_names = ['Tissue WM cm3', 'Tissue GM cm3', 'Tissue CSF cm3'] #'Tissue IC cm3'

    n = "Intracranial Cavity (IC)"
    v = vol_ice
    p = 100*v/vol_ice
    out.write(Template(getRowColor(0)+'$n & $v & $p \\\\\n').safe_substitute(n=n, v="{:5.2f}".format(v), p="{:5.3f}".format(p)))
    for i in range(len(tissues_names)):
        row_color = getRowColor(i+1)
        n = tissues_names[i]
        v = vols_tissue[tissue_vol_indices[i]]
        p = 100*v/vol_ice
        if(bounds_df.empty):
            out.write(Template(row_color+'$n & $v & $p \\\\\n').safe_substitute(n=n, v="{:5.2f}".format(v), p="{:5.3f}".format(p)))
        else:
            name_csv = csv_tissue_names[i]
            lb = bounds_df[bounds_df["structure"] == name_csv].lower_bound.iloc[0]
            ub = bounds_df[bounds_df["structure"] == name_csv].upper_bound.iloc[0]
            p_c = getColorByBound(p, lb, ub)
            out.write(Template(row_color+'$n & $p_c{$v} & $p_c{$p} \\scriptsize $p_c{[$a, $b]} \\\\\n').safe_substitute(n=n, v="{:5.2f}".format(v), p_c=p_c, p="{:5.3f}".format(p), a="{:5.3f}".format(lb), b="{:5.3f}".format(ub)))

    out.write(Template('\\end{tabularx}\n').safe_substitute())
    out.write(Template('\n').safe_substitute())
    #out.write(Template('\n').safe_substitute())
    #out.write(Template('\n').safe_substitute())


AGE_MIN=1
AGE_MAX=101

# To call only once when processing several files 
def read_bounds(sex):
    sex = sex.lower()
    if (sex == "female"):
        return pd.read_pickle("female_vb_bounds.pkl")
    elif (sex == "male"):
         return pd.read_pickle("male_vb_bounds.pkl")
    elif (sex == "unknown"):
         return pd.read_pickle("average_vb_bounds.pkl")
    else:
        return pd.DataFrame({'A' : []}) # empty DataFrame


#TODO: for each patient with age & sex
structures_names = np.array(['Lateral ventricles', 'Caudate', 'Putamen', 'Thalamus', 'Globus pallidus', 'Hippocampus', 'Amygdala', 'Accumbens'])
tissues_names = np.array(['Intracranial Cavity (IC)', 'White matter (including lesions)', 'Grey matter', 'Cerebrospinal fluid'])

def get_tissue_names_in_bound_csv(name):
    if (name == 'Intracranial Cavity (IC)'):
        name = 'Tissue IC cm3'
    elif (name == 'White matter (including lesions)'):
        name = 'Tissue WM cm3'
    elif (name == 'Grey matter'):
        name = 'Tissue GM cm3'
    elif (name == 'Cerebrospinal fluid'):
        name = 'Tissue CSF cm3'
    return name
    
def get_structure_names_in_bound_csv(name):
    return [f"{name} Total cm3", f"{name} Right cm3", f"{name} Left cm3", f"{name} Asymmetry"]
#TODO:BORIS: ce n'est pas des "cm3" !!!!!! Supprimer ca du code matlab !!!

def compute_vol_bounds(age, bounds_df):
    def get_bounds(name, age, df):
        assert isinstance(age, int)
        filtered_data = df[df.structure == name]
        #assert len(filtered_data) == 1, name_in_df
        return filtered_data[[f"age_{age}_lower", f"age_{age}_upper"]].iloc[0]

    # def clamp(row):
    #     if not "Asymmetry" in row["structure"]:
    #         return max(0, row["lower_bound"])
    #     else:
    #         return row["lower_bound"]
    
    def compute_bounds(age, bounds_df):
        out = pd.DataFrame(bounds_df["structure"])
        #print("out\n", out)
    
        age = max(min(age, AGE_MAX), AGE_MIN)
        age_inf = math.floor(age)
        age_sup = math.ceil(age)
        out[[f"age_{age_inf}_lower", f"age_{age_inf}_upper"]] = out.apply(lambda x: get_bounds(x.structure, age_inf, bounds_df), axis=1) #
        if age_inf < age_sup:
            out[[f"age_{age_sup}_lower", f"age_{age_sup}_upper"]] = out.apply(lambda x: get_bounds(x.structure, age_sup, bounds_df), axis=1) #, axis=1
        
        out["lower_bound"] = out.apply(lambda x: np.interp(age, [age_inf, age_sup], [x[f"age_{age_inf}_lower"], x[f"age_{age_sup}_lower"]]), axis=1)
        out["upper_bound"] = out.apply(lambda x: np.interp(age, [age_inf, age_sup], [x[f"age_{age_inf}_upper"], x[f"age_{age_sup}_upper"]]), axis=1)

        out = out[["structure", "lower_bound", "upper_bound"]]
        # print("out=\n", out)
        # out["lower_bound"] = out.apply(lambda row: clamp(row), axis=1)  #already done in csv
        # print("out=\n", out)
        
        return out

    return compute_bounds(age, bounds_df)


def compute_all(right_vol, left_vol, vol_ice):
    assert(vol_ice > 0)
    vol_total = right_vol+left_vol
    right_percentage = 100*right_vol/vol_ice
    left_percentage = 100*left_vol/vol_ice
    total_percentage = right_percentage+left_percentage  # 100*vol_total/vol_ice
    asymmetry = 100*(right_vol - left_vol) / ((right_vol + left_vol)*0.5)   #TODO: wrong formula ??? should be: 100*(right_vol - left_vol) / (right_vol + left_vol)
    return vol_total, right_percentage, left_percentage, total_percentage, asymmetry


def getColorByBound(p, low, up):
    assert(low <= up)
    if (p < low or p > up):
        return "\\textcolor{Maroon}"
    else:
        return "\\textcolor{black}"


def getBoundsSym(name, tp, rp, lp, a, vol_ice, bounds_df):
    names_csv = get_structure_names_in_bound_csv(name)
    # print("name=", name)
    # print("names_csv[0]=", names_csv[0])
    # print("names_csv[1]=", names_csv[1])
    # print("names_csv[2]=", names_csv[2])
    # print("names_csv[3]=", names_csv[3])
    # print("{}".format(bounds_df[bounds_df["structure"] == names_csv[0]]))
    # print("{}".format(bounds_df[bounds_df["structure"] == names_csv[1]]))
    # print("{}".format(bounds_df[bounds_df["structure"] == names_csv[2]]))
    # print("{}".format(bounds_df[bounds_df["structure"] == names_csv[3]]))
    tp_low = (bounds_df[bounds_df["structure"] == names_csv[0]]).lower_bound.iloc[0]
    tp_up = (bounds_df[bounds_df["structure"] == names_csv[0]]).upper_bound.iloc[0]
    rp_low = (bounds_df[bounds_df["structure"] == names_csv[1]]).lower_bound.iloc[0]
    rp_up = (bounds_df[bounds_df["structure"] == names_csv[1]]).upper_bound.iloc[0]
    lp_low = (bounds_df[bounds_df["structure"] == names_csv[2]]).lower_bound.iloc[0]
    lp_up = (bounds_df[bounds_df["structure"] == names_csv[2]]).upper_bound.iloc[0]
    a_low = (bounds_df[bounds_df["structure"] == names_csv[3]]).lower_bound.iloc[0]
    a_up = (bounds_df[bounds_df["structure"] == names_csv[3]]).upper_bound.iloc[0]
    #asymmetry bounds are normalized (like the rest of the measures) with 100/vol_ice in the Matlab code
    #We have to de-normalize them.
    #a_low *= vol_ice/100
    #a_up *= vol_ice/100
    tp_color = getColorByBound(tp, tp_low, tp_up)
    rp_color = getColorByBound(rp, rp_low, rp_up)
    lp_color = getColorByBound(lp, lp_low, lp_up)
    a_color =  getColorByBound(a, a_low, a_up)
    return tp_low, tp_up, tp_color, rp_low, rp_up, rp_color, lp_low, lp_up, lp_color, a_low, a_up, a_color


def get_structures_seg(out, vols_structures, vol_ice, colors_ice, colors_tissue, bounds_df):
    out.write(Template('\\begin{tabularx}{0.9\\textwidth}{X c c c c}\n').safe_substitute())
    out.write(Template('\\rowcolor{volbrain_blue} {\\bfseries \\textcolor{text_white}{Structure}} & {\\bfseries \\textcolor{text_white}{Total ($cm^{3}$ / \\%)}} & {\\bfseries \\textcolor{text_white}{Right ($cm^{3}$ / \\%)}} & {\\bfseries \\textcolor{text_white}{Left ($cm^{3}$ / \\%)}} & {\\bfseries \\textcolor{text_white}{Asymmetry (\\%)}}  \\\\\n').safe_substitute())
    """
    structures_names = np.array(['Left ventricle', 'Right ventricle', 'Left caudate','Right caudate',
                                    'Left putamen','Right putamen', 'Left thalamus','Right thalamus',
                                    'Left globus pallidus','Right globus pallidus', 'Left hippocampus','Right hippocampus',
                                'Left amygdala','Right amygdala', 'Left accumbens', 'Right accumbens'
                                    ])
    """
    structures_names = np.array(['Lateral ventricles', 'Caudate', 'Putamen', 'Thalamus', 'Globus pallidus', 'Hippocampus', 'Amygdala', 'Accumbens'])

    
    for i in range(len(structures_names)):
        rowColor = getRowColor(i)
        n = structures_names[i]
        l = vols_structures[i*2]
        r = vols_structures[i*2+1]
        t, rp, lp, tp, a = compute_all(r, l, vol_ice)

        if (bounds_df.empty):
            out.write(Template('$rc $n & $t / $tp & $r / $rp & $l / $lp & $a \\\\\n').safe_substitute(rc=rowColor, n=n, t="{:5.2f}".format(t), tp="{:5.3f}".format(tp), r="{:5.2f}".format(r), rp="{:5.3f}".format(rp), l="{:5.2f}".format(l), lp="{:5.3f}".format(lp), a="{:5.4f}".format(a)))
        else:
            tp_low, tp_up, tp_color, rp_low, rp_up, rp_color, lp_low, lp_up, lp_color, a_low, a_up, a_color = getBoundsSym(n, tp, rp, lp, a, vol_ice, bounds_df)
            out.write(Template('$rc $n & $tp_c{$t / $tp} & $rp_c{$r / $rp} & $lp_c{$l / $lp} & $a_c{$a} \\\\\n').safe_substitute(rc=rowColor, n=n, tp_c=tp_color, t="{:5.2f}".format(t), tp="{:5.3f}".format(tp), rp_c=rp_color, r="{:5.2f}".format(r), rp="{:5.3f}".format(rp), lp_c=lp_color, l="{:5.2f}".format(l), lp="{:5.3f}".format(lp), a_c=a_color, a="{:5.4f}".format(a)))
            out.write(Template('$rc & \\scriptsize $tp_c{[$tp_low, $tp_up]} & \\scriptsize $rp_c{[$rp_low, $rp_up]} & \\scriptsize $lp_c{[$lp_low, $lp_up]} & \\scriptsize $a_c{[$a_low, $a_up]} \\\\\n').safe_substitute(rc=rowColor, tp_c=tp_color, tp_low="{:5.3f}".format(tp_low), tp_up="{:5.3f}".format(tp_up), rp_c=rp_color, rp_low="{:5.3f}".format(rp_low), rp_up="{:5.3f}".format(rp_up), lp_c=lp_color, lp_low="{:5.3f}".format(lp_low), lp_up="{:5.3f}".format(lp_up), a_c=a_color, a_low="{:5.3f}".format(a_low), a_up="{:5.3f}".format(a_up)))

        # t = l+r
        # left = 100*vols_structures[i*2]/v
        # p = 100*v/vol_ice
        # out.write(Template(row_color+'$n & $v & $p & $left ($right) \\\\\n').safe_substitute(n=n, v="{:5.2f}".format(v), p="{:5.2f}".format(p), left="{:5.2f}".format(left), right="{:5.2f}".format(100-left)  ) )
        
    out.write(Template('\\end{tabularx}\n').safe_substitute())
    out.write(Template('\n').safe_substitute())


def plot_img(out, plot_images_filenames):
    #titles = ['FLAIR', 'Intracranial cavity segmentation', 'Tissue segmentation', 'Lesion segmentation']
    titles = ['FLAIR', 'Structure segmentation', 'Tissue segmentation', 'Lesion segmentation']
    
    #for i in [1, 2, 0, 3]:
    for i in [2, 1, 0, 3]:
        out.write(Template('\\begin{tabularx}{0.9\\textwidth}{X}\n').safe_substitute())
        out.write(Template('\\rowcolor{volbrain_blue} {\\bfseries \\textcolor{text_white}{$v}} \\\\\n').safe_substitute(v=titles[i]))
        out.write(Template('\\end{tabularx}\n').safe_substitute())
        out.write(Template('\\begin{tabularx}{0.8\\textwidth}{X}\n').safe_substitute())
        out.write(Template('\\centering \\includegraphics[width=0.25\\textwidth]{$f0} \\includegraphics[width=0.25\\textwidth]{$f1} \\includegraphics[width=0.25\\textwidth]{$f2}\\\\\n').safe_substitute(f0=plot_images_filenames[0, i], f1=plot_images_filenames[1, i], f2=plot_images_filenames[2, i]))
        out.write(Template('\\end{tabularx}\n').safe_substitute())
    out.write(Template('\\pagebreak\n').safe_substitute())
    out.write(Template('\n').safe_substitute())
    # out.write(Template('\\vspace*{30pt}\n').safe_substitute())


def get_tissue_plot(out, filenames_normal_tissue):
    #out.write(Template('\\begin{tabularx}{0.9\\textwidth}{X}\n').safe_substitute())
    # out.write(Template('\\rowcolor{volbrain_blue} {\\bfseries \\textcolor{text_white}{$v}} \\\\\n').safe_substitute(v='Tissue expected volumes'))
    #out.write(Template('\\end{tabularx}\n').safe_substitute())
    #out.write(Template('\n').safe_substitute())
    #out.write(Template('\\vspace*{2pt}\n').safe_substitute())
    #out.write(Template('\n').safe_substitute())
    out.write(Template('\\begin{tabularx}{0.8\\textwidth}{X}\n').safe_substitute())
    out.write(Template('\\centering \\includegraphics[width=0.25\\textwidth]{$f0} \\includegraphics[width=0.25\\textwidth]{$f1} \\includegraphics[width=0.25\\textwidth]{$f2}\\\\\n').safe_substitute(f0=filenames_normal_tissue[0], f1=filenames_normal_tissue[1], f2=filenames_normal_tissue[2]))
    out.write(Template('\\end{tabularx}\n').safe_substitute())
    out.write(Template('\n').safe_substitute())
    #out.write(Template('\\vspace*{1pt}\n').safe_substitute())




def write_footnotes(out, display_bounds=False):
        out.write(Template('\\begin{tabularx}{0.9\\textwidth}{X}\n').safe_substitute())
        out.write(Template('\\textcolor{text_gray}{\\footnotesize \\itshape All the volumes are presented in absolute value (measured in $cm^{3}$) and in relative value (measured in relation to the IC volume).}\\\\*\n').safe_substitute())
        if(display_bounds):
            out.write(Template('\\textcolor{text_gray}{\\footnotesize \\itshape Values between brackets show expected limits (95\%) of normalized volume in function of sex and age for each measure for reference purpose.}\\\\*\n').safe_substitute())
        out.write(Template('\\textcolor{text_gray}{\\footnotesize \\itshape Position provides the $x$, $y$ and $z$ coordinates of the lesion center of mass.}\\\\*\n').safe_substitute())
        out.write(Template('\\textcolor{text_gray}{\\footnotesize \\itshape Lesion burden is calculated as the lesion volume divided by the white matter volume.}\\\\*\n').safe_substitute())
        out.write(Template('\\textcolor{text_gray}{\\footnotesize \\itshape All the result images are located in the MNI space (neurological orientation).}\\\\*\n').safe_substitute())
        out.write(Template('\\end{tabularx}\n').safe_substitute())
        

def write_ref(out):
    out.write(Template('\\begin{tabularx}{0.9\\textwidth}{X}\n').safe_substitute())
    title="DeepLesionBrain: Towards a broader deep-learning generalization for multiple sclerosis lesion segmentation"
    authors="Reda Abdellah Kamraoui, Vinh-Thong Ta, Thomas Tourdias, Boris Mansencal, José V Manjon, Pierrick Coupé"
    where="Medical Image Analysis, volume 76, February 2022.~\\href{https://www.sciencedirect.com/science/article/pii/S1361841521003571}{PDF}"
    out.write(Template('{[}$r{]} $a, \\textit{$t}, $w\\\\\n').safe_substitute(r="1", a=authors, t=title, w=where))
    out.write(Template('\\end{tabularx}\n').safe_substitute())

        

def write_colors(out):
    out.write(Template('\\pagestyle{plain}\n').safe_substitute())
    out.write(Template('\n').safe_substitute())
    out.write(Template('\\definecolor{volbrain_blue}{RGB}{40,70,96}\n').safe_substitute())
    out.write(Template('\\definecolor{volbrain_clear_blue}{RGB}{161,185,205}\n').safe_substitute())
    out.write(Template('\\definecolor{header_blue}{RGB}{73,134,202}\n').safe_substitute())
    out.write(Template('\\definecolor{header_clear_blue}{RGB}{133,194,255}\n').safe_substitute())
    out.write(Template('\\definecolor{header_gray}{RGB}{200,200,200}\n').safe_substitute())
    # out.write(Template('\\definecolor{row_gray}{RGB}{230,230,230}\n').safe_substitute())
    out.write(Template('\\definecolor{text_white}{RGB}{255,255,255}\n').safe_substitute())
    out.write(Template('\\definecolor{text_gray}{RGB}{190,190,190}\n').safe_substitute())
    out.write(Template('\\hypersetup{colorlinks=true, urlcolor=volbrain_blue, linkcolor=volbrain_blue, filecolor=volbrain_blue}').safe_substitute())  # allow to not have a frame around the image
    out.write(Template('\n').safe_substitute())


def write_banner(out):
    #out.write(Template('\\hypersetup{colorlinks=true, urlcolor=magenta}').safe_substitute())  # allow to to not have a frame around the image
    out.write(Template('\n').safe_substitute())
    out.write(Template('\n').safe_substitute())
    out.write(Template('\\begin{document}\n').safe_substitute())
    out.write(Template('\\begin{center}\n').safe_substitute())
    filename_header = "header.png"
    out.write(Template('\\href{https://www.volbrain.net}{\\XeTeXLinkBox{ \\includegraphics[width=0.9\\textwidth]{$f}}}\\\\\n').safe_substitute(f=filename_header))
    # out.write(Template('\\begin{tabularx}{0.9\\textwidth}{X X X X}\n').safe_substitute())
    # out.write(Template('\\footnotesize \\itshape \\textcolor{NavyBlue}{version $v release $d}\n').safe_substitute(v=version, d=release_date))
    out.write(Template('{\\footnotesize \\itshape version $v release $d}\n').safe_substitute(v=version, d=release_date))
    # out.write(Template('\\end{tabularx}\n').safe_substitute())
    out.write(Template('\\vspace*{10pt}\n').safe_substitute())


def save_pdf(input_file, age, gender, snr, orientation_report, scale,
             bounds_all_df,
             vols_tissue, vol_ice, vols_structures,
             colors_ice, colors_lesions, colors_tissue, colors_structures,
             lesion_types_filename, plot_images_filenames,
             no_pdf_report):

    basename = replace_extension(os.path.basename(input_file).replace("mni_", "").replace("t1_", ""), ".nii.gz", "")
    # output_tex_filename = input_file.replace(".nii.gz", ".nii").replace(".nii", ".tex").replace("mni", "report")
    output_tex_filename = os.path.join(os.path.dirname(input_file), replace_extensions(os.path.basename(input_file).replace("mni_t1_", "report_"), [".nii.gz", ".nii"], ".tex"))
    print("output_tex_filename=", output_tex_filename)

    with open(output_tex_filename, 'w', newline='') as out:

        if not no_pdf_report:
            load_latex_packages(out)
            write_colors(out)
            write_banner(out)

            if not bounds_all_df.empty and not str(age).lower() == "unknown":
                bounds_df = compute_vol_bounds(age, bounds_all_df)
            else:
                bounds_df = pd.DataFrame({'A' : []}) # empty DataFrame

            display_bounds = (not bounds_df.empty)

            # Patient information
            get_patient_info(out, basename, gender, age)

            # Image information
            get_image_info(out, orientation_report, scale, snr)

            # Tissues Segmentation
            get_tissue_seg(out, vols_tissue, vol_ice, colors_ice, colors_tissue, age, bounds_df)

            out.write(Template('\\vspace*{3pt}\n').safe_substitute())
            
            # Tissue expected volumes
            
            if not bounds_all_df.empty:
                tissue_plots_filenames = save_tissue_plots(age, bounds_all_df, vols_tissue, vol_ice)
                get_tissue_plot(out, tissue_plots_filenames)

            if (not display_bounds):
                out.write(Template('\\vspace*{5pt}\n').safe_substitute())
                
            # structure Segmentation
            get_structures_seg(out, vols_structures, vol_ice, colors_ice, colors_structures, bounds_df)

            if (not display_bounds):
                out.write(Template('\\vspace*{20pt}\n').safe_substitute())
            else:
                out.write(Template('\\vspace*{5pt}\n').safe_substitute())

            # Lesion tables
            write_lesion_table(out, lesion_types_filename, colors_lesions, scale, vol_ice, WM_vol=vols_tissue[2])

            if (not display_bounds):
                out.write(Template('\\vspace*{50pt}\n').safe_substitute())
            else:
                out.write(Template('\\vspace*{6pt}\n').safe_substitute())
                
            # Footnotes
            write_footnotes(out, display_bounds=display_bounds)
            if (not display_bounds):
                out.write(Template('\\vspace*{10pt}\n').safe_substitute())
            else:
                out.write(Template('\\vspace*{1pt}\n').safe_substitute())
            out.write(Template('\n').safe_substitute())
            write_ref(out)
            out.write(Template('\n').safe_substitute())

            # out.write(Template('\\vspace*{5pt}\n').safe_substitute())
            # out.write(Template('\n').safe_substitute())

            out.write(Template('\\pagebreak\n').safe_substitute())

            # plot images
            plot_img(out, plot_images_filenames)

        # Lesion type tables
        all_lesions = write_lesions(out, lesion_types_filename, scale, vol_ice, vols_tissue[2])

        if not no_pdf_report:
            out.write(Template('\\end{center}\n').safe_substitute())
            out.write(Template('\\end{document}\n').safe_substitute())
            out.close()

            output_tex_basename = os.path.basename(output_tex_filename)
            output_tex_dirname = os.path.dirname(output_tex_filename)
            if not output_tex_dirname:
                output_tex_dirname = os.getcwd()
            #command = "xelatex -interaction=nonstopmode -output-directory={} {}".format(output_tex_dirname, output_tex_basename)
            command = "xelatex -interaction=batchmode -halt-on-error -output-directory={} {}".format(stringify(output_tex_dirname), stringify(output_tex_basename))
            print(command)
            run_command(command)

            # os.remove(output_tex_filename)  # comment out to debug LaTeX
            # os.remove(replace_extension(output_tex_filename, 'tex', 'log'))
            # os.remove(replace_extension(output_tex_filename, 'tex', 'aux'))
            # os.remove(replace_extension(output_tex_filename, 'tex', 'out'))

        return all_lesions


def save_csv(input_file, age, gender, all_lesions, vol_ice, snr, scale):
    basename = replace_extension(os.path.basename(input_file).replace("mni_", "").replace("t1_", ""), ".nii.gz", "")
    # output_csv_filename = input_file.replace(".nii.gz", ".nii").replace(".nii", ".csv").replace("mni", "report")
    output_csv_filename = os.path.join(os.path.dirname(input_file), replace_extensions(os.path.basename(input_file).replace("mni_t1_", "report_"), [".nii.gz", ".nii"], ".csv"))
    first_row = ['Patient ID', 'Sex', 'Age', 'Report Date', 'Scale factor', 'SNR',  # 'mSNR',
    	         'ICV cm3',
                 'Total lesion count', 'Total lesion volume (absolute) cm3', 'Total lesion volume (normalized) %', 'Total lesion burden',
                 'Periventricular lesion count', 'Periventricular lesion volume (absolute) cm3', 'Periventricular lesion volume (normalized) %', 'Periventricular lesion burden',
                 'Deep white lesion count', 'Deep white lesion volume (absolute) cm3', 'Deep white lesion volume (normalized) %', 'Deep white lesion burden',
                 'Juxtacortical lesion count', 'Juxtacortical lesion volume (absolute) cm3', 'Juxtacortical lesion volume (normalized) %', 'Juxtacortical lesion burden',
                 'Infratentorial lesion count', 'Infratentorial lesion volume (absolute) cm3', 'Infratentorial lesion volume (normalized) %', 'Infratentorial lesion burden',
                 'Cerebellar lesion count', 'Cerebellar lesion volume (absolute) cm3', 'Cerebellar lesion volume (normalized) %', 'Cerebellar lesion burden',
                 'Medular lesion count', 'Medular lesion volume (absolute) cm3', 'Medular lesion volume (normalized) %', 'Medular lesion burden']

    with open(output_csv_filename, mode='w') as output_file:
        csv_writer = csv.writer(output_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # write labels
        row = []
        csv_writer.writerow(first_row)
        # write labels names
        row.append(basename)
        row.append(str(gender))
        row.append(str(age))
        date = datetime.datetime.now().strftime("%d-%b-%Y")
        row.append(str(date))
        row.append(str(scale))
        row.append(str(snr))
        # row.extend(str (snr))
        # DO mSNR
        # verify order of lesion type
        # {'count':seg_num-1, 'volume_abs':vol_tot, 'volume_rel':vol_tot*100/vol_tot, 'burden': lesion_type.sum()}
        # juxtacortical_idx=3     deepwhite_idx=2     periventricular_idx=1     cerebelar_idx=4     medular_idx=5

        row.append(str(vol_ice))

        for lesion_info in all_lesions:
            # print(lesion_info)
            row.append(str(lesion_info['count']))
            row.append(str(lesion_info['volume_abs']))
            row.append(str(lesion_info['volume_rel']))
            row.append(str(lesion_info['burden']))

        csv_writer.writerow(row)


def save_images(suffixe,
                T1_slice, FLAIR_slice, CRISP_slice,
                LAB_slice, MASK_slice,
                colors_ice,
                colors_lesions, colors_tissue,
                out_height=OUT_HEIGHT, alpha=DEFAULT_ALPHA):

    T1_slice = np.rot90(T1_slice)
    FLAIR_slice = np.rot90(FLAIR_slice)
    LAB_slice = np.rot90(LAB_slice)
    MASK_slice = np.rot90(MASK_slice)
    CRISP_slice = np.rot90(CRISP_slice)

    out_im = make_slice_with_seg_image_with_alpha_blending(FLAIR_slice, LAB_slice, alpha=alpha, colors=colors_lesions)
    out_im = make_centered(out_im, out_height, out_height)
    filename_seg = "seg_{}.png".format(suffixe)
    Image.fromarray(out_im, 'RGB').save(filename_seg)

    out_im = make_slice_with_seg_image_with_alpha_blending(FLAIR_slice, LAB_slice, alpha=0, colors=colors_lesions)
    out_im = make_centered(out_im, out_height, out_height)
    filename_flair = "flair_{}.png".format(suffixe)
    Image.fromarray(out_im, 'RGB').save(filename_flair)

    out_im = make_slice_with_seg_image_with_alpha_blending(T1_slice, MASK_slice, alpha=alpha, colors=colors_ice)
    out_im = make_centered(out_im, out_height, out_height)
    filename_ice = "ice_{}.png".format(suffixe)
    Image.fromarray(out_im, 'RGB').save(filename_ice)

    out_im = make_slice_with_seg_image_with_alpha_blending(T1_slice, CRISP_slice, alpha=alpha, colors=colors_tissue)
    out_im = make_centered(out_im, out_height, out_height)
    filename_tissue = "tissue_{}.png".format(suffixe)
    Image.fromarray(out_im, 'RGB').save(filename_tissue)

    return filename_seg, filename_ice, filename_tissue, filename_flair

