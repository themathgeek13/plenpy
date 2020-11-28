ARG version
FROM python:$version-stretch

RUN groupadd -g 999 user && \
    useradd -r -u 999 -g user user && \
    mkdir /home/user

WORKDIR /home/user/

COPY . /home/user/plenpy/

RUN cd plenpy; pip --no-cache-dir install -r requirements.txt .
RUN chown -R user /home/user

USER user