from distutils.log import debug
from fileinput import filename
from flask import *
import time
import os.path

app = Flask(__name__)


@app.route('/')
def main():
    return render_template('index.html')


@app.route('/modelUploaded', methods=['POST'])
def modelUploaded():
    if request.method == 'POST':
        f = request.files['file']
        if(f != null):
            extension = os.path.splitext(f.filename)[1]
            timeEq = str(time.time())
            f.filename = './model/model' + timeEq.split(".")[0] + extension
            print(f.filename)
            f.save(f.filename)
            return render_template('modelUploaded.html', name=f.filename)
        else:
            return render_template('index.html', name=f.filename)
        

@app.route('/prepareModel')
def prepareModel():
    return render_template('prepareModel.html')


def configUploaded():
    if request.method == 'POST':
        f = request.files['file']
        extension = os.path.splitext(f.filename)[1]
        timeEq = str(time.time())
        f.filename = './model/model' + timeEq.split(".")[0] + extension
        print(f.filename)
        f.save(f.filename)
    return render_template('modelUploaded.html', name=f.filename)


if __name__ == '__main__':
    app.run(debug=True)

