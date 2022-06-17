FROM python:3.9-bullseye

#CMD export http_proxy=http://D255728:Koala2153Qute1@proxysis:8080
#CMD export https_proxy=https://D255728:Koala2153Qute1@proxysis:8080

# RUN pip install --proxy=$https_proxy pandas
RUN pip install pandas

COPY docker_test.py ./docker_test.py

CMD python docker_test.py