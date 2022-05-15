from report_utils import *

#B:TODO: input_root_dir="", global_csv=False, global_csv_filename=""
def report(input_t1_filename, input_flair_filename, MASK_filename, structures_filename, transform_filename,
           crisp_filename, lesion_types_filename, bounds_df, age='Unknown', sex='Unknown', no_pdf_report=False):
    FLAIR_img = nii.load(input_flair_filename)
    MASK_img = nii.load(MASK_filename)
    # LAB_img = nii.load(LAB_filename)
    # LAB = LAB_img.get_fdata()
    # # LAB_img = MASK_img

    sex = sex.lower()

    if (age != "Unknown"):
        try:
            age = float(age)
        except ValueError:
            print("WARNING: invalid value specified for age: {}".format(age))
            age = "Unknown"


    info_filename = os.path.join(os.path.dirname(input_t1_filename), replace_extension(os.path.basename(input_t1_filename).replace("mni_t1_", "img_info_"), ".nii.gz", ".txt"))
    snr, scale, orientation_report = read_info_file(info_filename)

    MASK = MASK_img.get_data()
    MASK = MASK.astype('int')

    FLAIR = FLAIR_img.get_data()
    # LAB = LAB_img.get_data()
    # LAB = LAB.astype('int')
    vol_ice = (compute_volumes(MASK, [[1]], scale))[0]
    CRISP = nii.load(crisp_filename).get_data()
    vols_tissue = (compute_volumes(CRISP, [[1], [2], [3]], scale))
    Structures = nii.load(structures_filename).get_data()
    vols_structures= (compute_volumes(Structures, [[i] for i in range(1,17)], scale))

    colors_lesions = np.array([[0, 0, 0],
                               [255, 0, 0], #Periventricular
                               [0, 255, 0], #Deepwhite
                               [0, 0, 255], #Juxtacortical
                               [255, 255, 0],   #Cerebellar
                               [0, 255, 255]]) #Medular

    colors_tissue = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]])
    colors_ice = np.array([[0, 0, 0], [255, 0, 0]])
    colors_structures= np.array([[0, 0, 0],
                                 [255, 0, 0], [255, 0, 0], #Lateral ventricles
                                 [0, 0, 255], [0, 0, 255], #Caudate
                                 [255, 255, 0], [255, 255, 0], #Putamen
                                 [0, 255, 255], [0, 255, 255], #Thalamus
                                 [255, 0, 255], [255, 0, 255], #Globus pallidus
                                 [255, 128, 0], [255, 128, 0], #Hippocampus
                                 [0, 128, 255], [0, 128, 255], #Amygdala
                                 [255, 0, 128], [255, 0, 128] ]) #Accumbens

    if not no_pdf_report:

        T1 = nii.load(input_t1_filename).get_fdata()
        T1 /= 300
        T1 = np.clip(T1, 0, 1)

        FLAIR /= 300
        FLAIR = np.clip(FLAIR, 0, 1)
        OUT_HEIGHT = 217
        DEFAULT_ALPHA = 0.5

        lesion_types = nii.load(lesion_types_filename).get_data()

        #I changed colors_ice with colors_structures and MASK with Structures
        # Axial
        slice_index = 80
        filename_seg_0, filename_ice_0, filename_tissue_0, filename_flair_0 = save_images("0",
                                                                                          T1[:, :, slice_index],
                                                                                          FLAIR[:, :, slice_index],
                                                                                          CRISP[:, :, slice_index],
                                                                                          lesion_types[:, :, slice_index],
                                                                                          Structures[:, :, slice_index],
                                                                                          colors_structures, colors_lesions, colors_tissue)
        # Coronal
        slice_index = 120
        filename_seg_1, filename_ice_1, filename_tissue_1, filename_flair_1 = save_images("1",
                                                                                          T1[:, slice_index, :],
                                                                                          FLAIR[:, slice_index, :],
                                                                                          CRISP[:, slice_index, :],
                                                                                          lesion_types[:, slice_index, :],
                                                                                          Structures[:, slice_index, :],
                                                                                          colors_structures, colors_lesions, colors_tissue)
        # Sagittal
        slice_index = 70
        filename_seg_2, filename_ice_2, filename_tissue_2, filename_flair_2 = save_images("2",
                                                                                          T1[slice_index, :, :],
                                                                                          FLAIR[slice_index, :, :],
                                                                                          CRISP[slice_index, :, :],
                                                                                          lesion_types[slice_index, :, :],
                                                                                          Structures[slice_index, :, :],
                                                                                          colors_structures, colors_lesions, colors_tissue)

        plot_images_filenames = np.array([[filename_flair_0, filename_ice_0, filename_tissue_0, filename_seg_0],
                                          [filename_flair_1, filename_ice_1, filename_tissue_1, filename_seg_1],
                                          [filename_flair_2, filename_ice_2, filename_tissue_2, filename_seg_2]])

    else:
        plot_images_filenames = np.array([["", "", "", ""],
                                          ["", "", "", ""],
                                          ["", "", "", ""]])


    all_lesions = save_pdf(input_t1_filename, age, sex, snr, orientation_report, scale,
                           bounds_df, 
                           vols_tissue, vol_ice, vols_structures,
                           colors_ice, colors_lesions, colors_tissue, colors_structures,
                           lesion_types_filename, plot_images_filenames,
                           no_pdf_report)

    save_csv(input_t1_filename, age, sex, all_lesions, vol_ice, snr, scale)

    os.remove(info_filename)

