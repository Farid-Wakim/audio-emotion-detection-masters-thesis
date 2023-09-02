from distutils.log import debug
from fileinput import filename
from flask import *
import time
import os.path
import os, glob

app = Flask(__name__)


@app.route('/')
def main():
    for filename in glob.glob("files/model*"):
        os.remove(filename) 
    for filename in glob.glob("files/data*"):
        os.remove(filename) 
    return render_template('index.html')


@app.route('/modelUploaded', methods=['POST'])
def modelUploaded():
    if request.method == 'POST':
        f = request.files['file']
        if(f.filename.__len__() > 0): 
            extension = os.path.splitext(f.filename)[1]
            # timeEq = str(time.time())
            f.filename = 'files/data'+ extension            
            f.save(f.filename)
            return render_template('dataUploaded.html')
        else:
            return render_template('index.html')


@app.route('/dataUploaded', methods=['POST'])
def dataUploaded():
    if request.method == 'POST':
        f = request.files['file']
        if(f.filename.__len__() > 0):
            extension = os.path.splitext(f.filename)[1]
            # timeEq = str(time.time())
            f.filename = 'files/data'+ extension            
            f.save(f.filename)
            return render_template('testResults.html')
        else:
            return render_template('index.html')




@app.route('/prepareModel')
def prepareModel():
    return render_template('prepareModel.html')

@app.route('/prepareConfig' , methods=['POST'])
def configUploaded():
    if request.method == 'POST':
        f = request.form.get('config')
        print(f)
        
    return render_template('configUpdated.html')


if __name__ == '__main__':
    app.run(debug=True)

