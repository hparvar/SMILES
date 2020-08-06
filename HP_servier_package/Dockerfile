FROM python:3.6

RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda
RUN export PATH=~/miniconda/bin:$PATH
RUN #!/bin/sh conda update -n base conda
RUN #!/bin/sh conda create -y --name servier python=3.6
RUN #!/bin/sh conda activate servier


RUN #!/bin/sh conda update -n base conda
RUN #!/bin/sh conda create -y --name servier python=3.6


RUN conda init bash
RUN #!/bin/sh echo "conda activate servier" > ~/.bashrc

RUN #!/bin/sh conda install -y tensorflow=2.1.0
RUN #!/bin/sh conda install -y scikit-learn
RUN #!/bin/sh conda install -y -c conda-forge rdkit
RUN #!/bin/sh conda install -y -c conda-forge matplotlib
RUN #!/bin/sh conda install -y seaborn
RUN #!/bin/sh conda install -y -c conda-forge flask-restful
ADD setup.py /
CMD [ "python", "setup.py" ]