# importing Flask and other modules
from flask import Flask, render_template, request, url_for, flash, redirect
from flask_wtf import FlaskForm
from werkzeug.utils import secure_filename
from flask_wtf.file import FileField, FileRequired
import  os

# Flask constructor
app = Flask(__name__)  
import os
SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY

class UploadForm(FlaskForm):
    file = FileField()

@app.route('/', methods =["GET", "POST"])
def render():
    form = UploadForm()
    
    if form.validate_on_submit():
        filename = secure_filename(form.file.data.filename)
        form.file.data.save('uploads/' + filename)
        return redirect(url_for('upload'))
    return render_template("index.html" , form=form)
 
if __name__=='__main__':
   app.run()
