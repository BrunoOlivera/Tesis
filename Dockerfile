FROM python:3.9-bullseye

# CMD export http_proxy=http://D255728:Koala2153Qute1@proxysis:8080
# CMD export https_proxy=http://D255728:Koala2153Qute1@proxysis:8080

ENV http_proxy=http://D255728:Koala2153Qute1@proxysis:8080
ENV https_proxy=http://D255728:Koala2153Qute1@proxysis:8080

RUN pip install --proxy=$https_proxy pandas
RUN pip install --upgrade pip setuptools wheel
RUN pip install --proxy=$https_proxy mpi4py
# RUN pip install pandas

COPY docker_test.py ./docker_test.py

# CMD python docker_test.py
# CMD /bin/bash

ENTRYPOINT ["tail"]
CMD ["-f","/dev/null"]