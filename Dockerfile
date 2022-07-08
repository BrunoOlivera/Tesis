FROM python:3.9-bullseye

ENV MPI_DIR=/opt/ompi
ENV PATH="$MPI_DIR/bin:$HOME/.local/bin:$PATH"
ENV LD_LIBRARY_PATH="$MPI_DIR/lib:$LD_LIBRARY_PATH"

ENV http_proxy=http://D255728:Koala2153Qute1@proxysis:8080
ENV https_proxy=https://D255728:Koala2153Qute1@proxysis:8080


ADD https://download.open-mpi.org/release/open-mpi/v3.1/openmpi-3.1.4.tar.bz2 .
RUN tar xf openmpi-3.1.4.tar.bz2 \
    && cd openmpi-3.1.4 \
    && ./configure --prefix=$MPI_DIR \
    && make -j4 all \
    && make install \
    && cd .. && rm -rf \
    openmpi-3.1.4 openmpi-3.1.4.tar.bz2 /tmp/*

# RUN pip install --upgrade pip
RUN pip install --proxy=$https_proxy pandas
# RUN pip install pandas
RUN pip install --upgrade pip setuptools wheel
RUN pip install --proxy=$https_proxy mpi4py

# COPY docker_test.py ./docker_test.py
# RUN mkdir Prototipo
# COPY ../Prototipo/ Prototipo/
COPY * Prototipo/
RUN cd Prototipo

# CMD python docker_test.py
# CMD /bin/bash

ENTRYPOINT ["tail"]
CMD ["-f","/dev/null"]