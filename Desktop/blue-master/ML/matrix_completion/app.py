from flask import Flask, request
from flask_restful import Resource, Api
from json import dumps
from run_predictor import main

app = Flask(__name__)
api = Api(app)

class Predictor(Resource):
    def get(self):
        num = request.args.get('num')
        rater_id = request.args.get('rater_id')
        db_name = request.args.get('db_name')
        db_host = request.args.get('db_host')
        db_username = request.args.get('db_username')
        db_password = request.args.get('db_password')
        local = request.args.get('local')
        # print(num)
        # print(rater_id)
        # print(db_name)
        # print(db_host)
        # print(db_username)
        # print(db_password)
        # print(local)
        args = ['dummy', num, rater_id, db_name, db_host, db_username, db_password, local]
        jids = main(args)
        return jids


    

api.add_resource(Predictor, '/predict')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)