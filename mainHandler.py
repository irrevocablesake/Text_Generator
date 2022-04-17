from main import *
from flask import *

app = Flask(__name__)   

trained = 0
default = "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt"
textGeneration = ""

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/generate', methods=['GET'])
def generate():
    if request.method == 'GET':
        data_url = request.args.get('data_url')
        prompt = request.args.get('prompt')
        epoch = int(request.args.get('epoch'))
        result_length_data = int(request.args.get('length'))
        retrain = request.args.get('retrain')

        global trained
        global textGeneration
        global default

        if data_url == "":
            data_url = default

        if(retrain == 'yes'):
            textGeneration = TextGeneration(epoch, data_url)
            textGeneration.trainModel()
            text = textGeneration.generateText(prompt, result_length_data)
        elif(retrain == 'no'):
            text = textGeneration.generateText(prompt, result_length_data)

        return render_template('generate.html', value = text)

if __name__=="__main__":
    app.run()
    