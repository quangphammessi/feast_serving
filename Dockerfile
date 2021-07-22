FROM gcr.io/kf-feast/feast-core:develop
RUN mkdir /app

# install conda
WORKDIR /app
SHELL ["/bin/bash", "-c"]
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.7.12-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

# Install dependency
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
COPY environment.yml /app/conda/
RUN /opt/conda/bin/conda update -n base -c defaults conda
RUN /opt/conda/bin/conda env create -f /app/conda/environment.yml
RUN /opt/conda/bin/conda clean -afy


# switch to the conda environment
RUN echo "conda activate kubeflow-lpr" >> ~/.bashrc
ENV PATH /opt/conda/envs/kubeflow-lpr/bin:$PATH
RUN /opt/conda/bin/activate kubeflow-lpr

CMD [ "/bin/bash" ]