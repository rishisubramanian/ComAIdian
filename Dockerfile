# To build:
# docker build -t comaidian .
# To run:
# docker run -it -v $PWD:/tmp/src -w /tmp/src -p 3001:3000 invoice_report

FROM python:3.6

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache -r requirements.txt && pip freeze > requirements.txt

COPY . .

CMD [ "jupyter", "notebook" ]