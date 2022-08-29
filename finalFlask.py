from multiprocessing import connection
import dill as pickle
from flask import Flask, render_template
from flask import Response
import io
from io import BytesIO
import pandas as pd
from pandas import DatetimeIndex as dt
from flask_restful import Api, Resource, reqparse
import json
import matplotlib.pyplot as plt
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
from pandas import DataFrame
import warnings
import base64
import matplotlib
from matplotlib.dates import DateFormatter
matplotlib.use('Agg')


app = Flask(__name__)
api = Api(app)
if __name__ == "__main__":
    app.run(debug = True)

import pandas as pd
from pandas import *
import matplotlib.pyplot as plt
import csv

def benerinDataset():

    import psycopg2
    import pandas as pd
    connection = psycopg2.connect("host='{}' port={} dbname='{}' user={} password={}".format("ec2-23-23-151-191.compute-1.amazonaws.com",
                                                                    5432,
                                                                    "d2rrek30kgf61d",
                                                                    "ydwguthxphaceh",
                                                                    "95455357251bd91746ff8d6d415d97c1ddc8103c4386ec31db3bfa9dbb9206d3"))
    sql_query = "SELECT waktu FROM booking;"
    data = pd.read_sql_query(sql_query,con=connection)

    # a = data['dates'][0].date()
    # dateString = a.strftime("%d/%m/%y")

    hours = []

    for i in range(len(data)):
        h = data['waktu'][i].hour
        hours.append(h)
        
    def get_unique_numbers(numbers):

        list_of_unique_numbers = []

        unique_numbers = set(numbers)

        for number in unique_numbers:
            list_of_unique_numbers.append(number)

        return list_of_unique_numbers


    newHours = get_unique_numbers(hours)

    newHours2 = []

    for i in range(len(newHours)):
        f = str(newHours[i])+':00:00'
        newHours2.append(f)

    from datetime import datetime

    amount = []
    stringHours = []

    for i in range(len(newHours)):
        s = newHours[i]
        number = hours.count(s)
        amount.append(float(number))
        h = str(s)
        string = h+':00:00'
        stringHours.append(datetime.strptime(string, '%H:%M:%S'))

    newData = {'date':stringHours, 'amount':amount}
    df = pd.DataFrame(newData)

    import pandas as pd

    from io import StringIO

    csv_string = df.to_csv(index=False)
    dataset = pd.read_csv(StringIO(csv_string), index_col=0, parse_dates=['date'])

    return (dataset, newHours2)



parser = reqparse.RequestParser()
parser.add_argument('data')

@app.route('/')
def plot_png():

    (dataset, newHours) = benerinDataset()

    from pmdarima.arima import ADFTest

    adf_test = ADFTest(alpha = 0.05)
    value = adf_test.should_diff(dataset)
    type(value)
    value[0]

    train_ = dataset[:85]
    test_ = dataset[-20:]

    from pmdarima.arima import auto_arima

    if value[0] < 0.05:
        diff = 1
    else:
        diff = 0
    
    model = auto_arima(train_, start_p=0,d=diff, start_q=0,max_p=8,
                        max_d=8,max_q=8,start_P=0,D=diff,start_Q=0,
                    max_P=8,max_D=8,max_Q=8,m=1,seasonal=False,
                    error_action='warn',trace=True,
                    supress_warnings=True,stepwis=True,random_state=20,
                    n_fits=50)

    pdq = model.order

    from statsmodels.tsa.arima.model import ARIMA

    newModel = ARIMA(dataset.values,order=(pdq[0],pdq[1],pdq[2]))
    trainned_model= newModel.fit()

    date_time = pd.to_datetime(newHours)
    data_forecast = trainned_model.predict()

    fig, ax = plt.subplots(figsize = (11,5))
    fig, ax = plt.subplots()
    # Use automatic FuncFormatter creation
    hh_mm = DateFormatter('%H:%M')
    ax.xaxis.set_major_formatter(hh_mm)
    ax.bar(date_time, data_forecast, width=0.035, edgecolor="white", linewidth=0.7)
    plt.xticks(date_time)
    plt.show()

    img = BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    # return render_template('show_arima.blade.php', plot_url=plot_url)
    pp = {'pictureData': plot_url}
    return pp

    # plt.ylabel("Amount", size = 5)

    # output = io.BytesIO()
    # FigureCanvas(fig).print_png(output)
    # return Response(output.getvalue(), mimetype='image/png')
    # return render_template('test.blade.php')