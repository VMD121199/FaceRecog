from src import attendence as attend
import json
from flask import Flask, request
app = Flask(__name__)

# @app.route('/result', methods=['POST'])
# def result():
#     if request.method == 'POST':
#         result = attend.AttendenceCheck("None")
#         name = result[0]
#         percents = result[1]
#         return render_template("result.html", name=name, percents=percents)


@app.route('/api/checked', methods = ['POST', 'GET'])
def test():
    if request.method == 'POST':
        defination_id = request.json.get('defination_id')
        imgUrl = request.json.get('imgUrl')
        result = attend.AttendenceCheck(imgUrl)
        predict_name = result[0]
        probability = result[1]
        if predict_name != defination_id:
            result_respone = { "defination_id": defination_id, 
                                "match_percent": 0}
            json_result = json.dumps(result_respone)
            print(json_result)
        else:
            result_respone = { "defination_id": defination_id, 
                                "match_percent": probability}
            json_result = json.dumps(result_respone)
            print(json_result)
    return "DONE"

if __name__ == "__main__":
    app.run()