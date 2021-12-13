import os
import io

import numpy as np
import pandas as pd
from ensembles import RandomForestMSE, GradientBoostingMSE

import plotly
import plotly.graph_objects as go

from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from flask import Flask, request, url_for
from flask import render_template, redirect

from flask_wtf.file import FileAllowed
from wtforms.validators import DataRequired
from wtforms import StringField, SubmitField, FileField


app = Flask(__name__, template_folder='html')
app.config['BOOTSTRAP_SERVE_LOCAL'] = True
app.config['SECRET_KEY'] = 'hello'
data_path = './../data'
Bootstrap(app)
messages = []
data_input = False
model = RandomForestMSE(n_estimators=1000)
    
class Params:
    n_estimators = 1000
    max_depth = 5
    feature_size = 6
    learning_rate = 0.1
    forest = False
    shape = None
    columns = []
    history = {}

class Data:
    data = {'train': None, 'val': None, 'pred': None}
    files = {'train': False, 'val': False}
    target = 'target'
    del_cols = ''

params = Params()
data_files = Data()

class Check_ParamsGB(FlaskForm):
    n_estimators = StringField('n_estimators', validators=[DataRequired()])
    max_depth = StringField('max_depth', validators=[DataRequired()])
    feature_size = StringField('feature_size', validators=[DataRequired()])
    learning_rate = StringField('learning_rate', validators=[DataRequired()])
    submit = SubmitField('Вернуться назад')
    
class Check_ParamsRF(FlaskForm):
    n_estimators = StringField('n_estimators', validators=[DataRequired()])
    max_depth = StringField('max_depth', validators=[DataRequired()])
    feature_size = StringField('feature_size', validators=[DataRequired()])
    submit = SubmitField('Вернуться назад')

class Data_Info(FlaskForm):
    shape = StringField('Shape', validators=[DataRequired()])
    columns = StringField('Columns', validators=[DataRequired()])
    submit = SubmitField('Вернуться назад')

class FileForm(FlaskForm):
    file_path = FileField('Path', validators=[
        DataRequired('Specify file'),
        FileAllowed(['csv'], 'CSV only!')
    ])
    submit = SubmitField('Open File')


@app.route('/file_train', methods=['GET', 'POST'])
def file_train():
    file_form = FileForm()
    global data_files

    if request.method == 'POST' and file_form.validate_on_submit():
        stream = io.StringIO(file_form.file_path.data.stream.read().decode("UTF8"), newline=None)
        data_files.data['train'] = pd.read_csv(stream)
        data_files.files['train'] = True
        return redirect(url_for('upload_data'))

    return render_template('from_form.html', form=file_form)

@app.route('/file_val', methods=['GET', 'POST'])
def file_val():
    file_form = FileForm()
    global data_files

    if request.method == 'POST' and file_form.validate_on_submit():
        stream = io.StringIO(file_form.file_path.data.stream.read().decode("UTF8"), newline=None)
        data_files.data['val'] = pd.read_csv(stream)
        data_files.files['val'] = True
        return redirect(url_for('upload_data'))

    return render_template('from_form.html', form=file_form)

@app.route('/file_predict', methods=['GET', 'POST'])
def file_predict():
    file_form = FileForm()
    global data_files

    if request.method == 'POST' and file_form.validate_on_submit():
        stream = io.StringIO(file_form.file_path.data.stream.read().decode("UTF8"), newline=None)
        data_files.data['pred'] = pd.read_csv(stream)
        data_files.files['pred'] = True
        return redirect(url_for('predict_test'))

    return render_template('from_form.html', form=file_form)

@app.route('/')
@app.route('/index')
def index():
    global data_input
    global data_files
    global params
    params = Params()

    data_files.data = {'train': None, 'val': None, 'pred': None}
    data_files.files = {'train': False, 'val': False}
    data_files.target = 'target'
    data_files.del_cols = ''

    data_input = False
    return render_template('index.html')


@app.route('/index_js')
def get_index():
    return '<html><center><script>document.write("Hello, i`am Flask Server!")</script></center></html>'

@app.route('/data', methods=['GET', 'POST'])
def upload_data():
    global data_input
    global params
    data_input = True
    return render_template('file_input.html', train=data_files.files['train'])

@app.route('/processing', methods=['GET', 'POST'])
def processing():
    global data_files

    if request.method == 'POST':
        data_files.target, data_files.del_cols =\
            request.form['target'], request.form['del_cols']
        
        if data_files.target == '':
            data_files.target = 'target'
        
        del_cols = data_files.del_cols
        data_files.del_cols = del_cols.split() + [data_files.target]
        
        for clm in data_files.del_cols:
            if clm not in data_files.data['train'].columns.values:
                return render_template('exeption.html', func='processing')
        
        return redirect(url_for('choise_model'))
    
    return render_template('processing.html')

@app.route('/choise', methods=['GET', 'POST'])
def choise_model():
    global data_input
    return render_template('choise_model.html', data_input=data_input)

@app.route('/training', methods=['GET', 'POST'])
def train_model():
    global params
    global model
    global data_files
    
    try:
        if data_files.files['train'] == True:
            data = data_files.data['train']
            target = data[data_files.target].values
            data_files.del_cols += [data_files.target]
            data = data.drop(data_files.del_cols, axis=1)
        else:
            data = pd.read_csv('kc_house_data.csv')
            target = data.price.values
            data = data.drop(['price', 'id', 'date'], axis=1)
            data_files.del_cols = ['price', 'id', 'date']
            data_files.target = 'price'
        
        if data_files.files['val'] == True:
            data_val = data_files.data['val']
            y_val = data_val[data_files.target].values
            data_val = data_val.drop(data_files.del_cols, axis=1)
        
        X = data.values
        if data_files.files['val']:
            X_val = data_val.values
        else:
            X_val, y_val = None, None
        
        params.shape = X.shape
        params.columns = data.columns.values
        
        if params.n_estimators is None or params.n_estimators == '' or params.n_estimators == 'None':
            params.n_estimators = 1000
        else:
            params.n_estimators = int(params.n_estimators)
            if params.n_estimators <= 0:
                raise TypeError("Nor right arguments")
        
        if params.max_depth is None or params.max_depth == '' or params.max_depth == 'None':
            if params.forest:
                params.max_depth = None
            else:
                params.max_depth = 5
        else:
            params.max_depth = int(params.max_depth)
            if params.max_depth <= 0:
                raise TypeError("Nor right arguments")
        
        if params.feature_size is None or params.feature_size == '' or params.feature_size == 'None':
            params.feature_size = X.shape[1] // 3
        else:
            params.feature_size = int(params.feature_size)
            if params.feature_size <= 0:
                raise TypeError("Nor right arguments")
        
        if params.learning_rate is None or params.learning_rate == '':
            params.learning_rate = 0.1
        else:
            params.learning_rate = float(params.learning_rate)
            if params.learning_rate < 0:
                raise TypeError("Nor right arguments")
            
        if params.forest:
            model = RandomForestMSE(n_estimators = params.n_estimators, max_depth=params.max_depth,
                                  feature_subsample_size =params.feature_size)
        else:
            model = GradientBoostingMSE(n_estimators = params.n_estimators, max_depth=params.max_depth,
                                  feature_subsample_size =params.feature_size, learning_rate=params.learning_rate)
        params.history = model.fit(X, target, X_val=X_val, y_val=y_val)
        
        return render_template('train_model.html', forest=params.forest)
    except Exception as exc:
        if params.forest:
            return render_template('exeption.html', func='prepare_forest')
        else:
            return render_template('exeption.html', func='prepare_boosting')

@app.route('/check_result', methods=['GET', 'POST'])
def check_result():
    global params
    return render_template('check_result.html', forest=params.forest)

@app.route('/params', methods=['GET', 'POST'])
def check_params():
    try:
        global params
        if params.forest is True:    
            check = Check_ParamsRF()
        else:
            check = Check_ParamsGB()
            check.learning_rate.data = params.learning_rate

        if check.validate_on_submit():
            return redirect(url_for('check_result'))
        
        check.n_estimators.data = params.n_estimators
        if params.max_depth is None:
            check.max_depth.data = 'None'
        else:
            check.max_depth.data = params.max_depth
        check.feature_size.data = params.feature_size
        
        return render_template('from_form.html', form=check)
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))

@app.route('/info_data', methods=['GET', 'POST'])
def check_info_data():
    try:
        global params
        info = Data_Info()

        if info.validate_on_submit():
            return redirect(url_for('check_result'))
        
        info.shape.data = params.shape
        info.columns.data = params.columns
        
        return render_template('from_form.html', form=info)
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))

@app.route('/predict', methods=['GET', 'POST'])
def predict_test():
    global model
    global data_files
    
    data = data_files.data['pred']
    
    del_cols = data_files.del_cols
    while data_files.target in del_cols:
        del_cols.remove(data_files.target)
    
    data = data.drop(del_cols, axis=1)
    X = data.values
    res = model.predict(X)

    return render_template('predictions.html', res=res)


@app.route('/forest', methods=['GET', 'POST'])
def prepare_forest():
    global params
    params = Params()
    params.forest = True

    if request.method == 'POST':
        params.n_estimators, params.max_depth, params.feature_size =\
            request.form['n_estimators'], request.form['max_depth'], request.form['feature_size']
        return redirect(url_for('train_model'))

    return render_template('messages.html')

@app.route('/boosting', methods=['GET', 'POST'])
def prepare_boosting():
    global params
    params = Params()
    params.forest = False

    if request.method == 'POST':
        params.n_estimators, params.max_depth, params.feature_size, params.learning_rate =\
            request.form['n_estimators'], request.form['max_depth'], request.form['feature_size'], request.form['learning_rate']
        return redirect(url_for('train_model'))

    return render_template('boosting.html')


@app.route('/dashboard', methods=['GET', 'POST'])
def get_dashboard():
    global params
    hist = params.history
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(len(hist['loss_train'])), y=hist['loss_train'], name='train'))
    if len(hist['loss_val']) > 0:
        fig.add_trace(go.Scatter(x=np.arange(len(hist['loss_val'])), y=hist['loss_val'], name='val'))
    fig.update_layout(title="Значения функции ошибки от итерации",
                      xaxis_title="Итерация",
                      yaxis_title="Loss",
                      height=700,
                      width=1500)

    return render_template(
        'dashboard.html',
        dashboard_div=fig.to_html(full_html=False)
    )
