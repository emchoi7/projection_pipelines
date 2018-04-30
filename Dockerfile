FROM python

WORKDIR /code
ADD . /code
RUN pip install sklearn
RUN pip install pandas
RUN pip install numpy
RUN pip install scipy
CMD python test_combination.py