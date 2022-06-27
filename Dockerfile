FROM nvcr.io/nvidia/tensorflow:21.08-tf1-py3
#FROM tensorflow/tensorflow:1.15.4-gpu-py3

LABEL name=new_les \
      version=1.0 \
      maintainer=reda-abdellah.kamraoui@labri.fr \
      net.volbrain.pipeline.mode=gpu_only \
      net.volbrain.pipeline.name=assemblyNet

RUN curl -LO https://ssd.mathworks.com/supportfiles/downloads/R2017b/deployment_files/R2017b/installers/glnxa64/MCR_R2017b_glnxa64_installer.zip && \
    mkdir MCR && \
    cp MCR_R2017b_glnxa64_installer.zip MCR && \
    cd MCR && \
    unzip MCR_R2017b_glnxa64_installer.zip && \
    ./install -mode silent -agreeToLicense yes && \
    cd .. && \
    rm -rf MCR

ARG DEBIAN_FRONTEND=noninteractive
ARG DEBCONF_NONINTERACTIVE_SEEN=true
RUN mkdir -p /opt/new_les
WORKDIR /opt/new_les

RUN apt-get update
# RUN apt -qqy install libx11-dev xserver-xorg libfontconfig1 libxt6 libxcomposite1 libasound2 libxext6 texlive-xetex
RUN apt -qqy install libfontconfig1 libxt6 libxext6 texlive-xetex

COPY Compilation_deepnewlesion_v10_fullpreprocessing/ /opt/new_les/

RUN pip3 install statsmodels  keras==2.2.4 pillow nibabel==2.5.2 scikit-image==0.17.2 pandas

RUN mkdir /Weights
COPY CHALLENGE_WEIGHTS_inpaint_detection_decoder_FMs_da_v3_dataloader_by_lesion3_by_tile5_negative_samples1_challenge/ /Weights/

RUN apt -qqy install libatk1.0-0
RUN  apt-get install sudo -y

#RUN chmod 777 /opt/new_les/DeepNewLesion_v10_fullpreprocessing_exe
#RUN chmod 777 /opt/new_les/run_DeepNewLesion_v10_fullpreprocessing_exe.sh
#RUN chmod 777 /opt/new_les/IHCorrection/CompilationSPM8/spm8
#RUN chmod 777 -R /opt/new_les/*
#ENV LD_PRELOAD=/lib/x86_64-linux-gnu/libfreetype.so.6
#/usr/lib/x86_64-linux-gnu/libfreetype.so.6

RUN mv /usr/local/MATLAB/MATLAB_Runtime/v93/bin/glnxa64/libfreetype.so.6 /usr/local/MATLAB/MATLAB_Runtime/v93/bin/glnxa64/libfreetype.so.6.bak

RUN pip3 install torch==1.10.0
COPY voxel_only_da_v3_k0_nf_24.pt /Weights/voxel_only_da_v3_k0_nf_24._pt
RUN mkdir /Weights_DLB
COPY DLB_Weights/ /Weights_DLB/
COPY networks_lifespan360x2_FT_trimmed /opt/new_les/networks_lifespan360x2_FT

COPY PIPELINE /opt/new_les/

COPY *.py header.png female_vb_bounds.pkl male_vb_bounds.pkl average_vb_bounds.pkl README.pdf /opt/new_les/



ENTRYPOINT [ "python3", "/opt/new_les/end_to_end_pipeline_file.py" ]

#sudo docker run -it --rm --gpus device=0 -v /media/rkamraoui/4TB/preprocessing/test:/data new_les /data/90_99172_T1.nii.gz /data/90_99172_FLAIR.nii.gz /data/90_122039_T1.nii.gz /data/90_122039_FLAIR.nii.gz