FROM opencv4.4_cp38:base
COPY opencv_base /opt/df_face_arm

ENV LANG C.UTF-8
RUN mkdir ~/.pip
RUN echo '[global]   \n\
index-url=http://pypi.douban.com/simple \n\
[install] \n\
trusted-host=pypi.douban.com '\
>> ~/.pip/pip.conf

RUN pip3 install -r /opt/df_face_arm/requirements.txt

WORKDIR /opt/df_face_arm

CMD python3 run.py

