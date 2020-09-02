FROM python:3.8
ADD . /code
WORKDIR /code
RUN pip install -r requirements.txt
RUN python -m spacy download en_core_web_sm
CMD python ./app/controller.py

