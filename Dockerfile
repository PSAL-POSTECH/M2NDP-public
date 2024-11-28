FROM centos:8
ENV CC=/usr/bin/gcc
ENV CXX=/usr/bin/g++

RUN sed -i 's/mirrorlist/#mirrorlist/g' /etc/yum.repos.d/CentOS-Linux-*
RUN sed -i 's|#baseurl=http://mirror.centos.org|baseurl=http://vault.centos.org|g' /etc/yum.repos.d/CentOS-Linux-*

RUN yum update -y && yum install -y gcc gcc-c++ make python38 openssl-devel wget cmake bison flex
RUN pip3 install conan==1.56.0 numpy==1.24.4 torch==2.2.0 pandas==1.2 seaborn==0.13.2

COPY ./ m2ndp
RUN cd m2ndp && ./scripts/build_functional.sh
RUN rm -rf m2ndp/build
RUN cd m2ndp && ./scripts/build_timing.sh
RUN rm -rf m2ndp/build


RUN echo "Welcome to M2NDP simulator!"