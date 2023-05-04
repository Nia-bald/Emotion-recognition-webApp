from flask import Flask, render_template, request
import pandas as pd
import json
import plotly
import plotly.graph_objs as go
from plotly.subplots import make_subplots
# importing libraries
import numpy as np
import librosa
import librosa.display
import IPython.display as ipd
from itertools import cycle, islice
import torchaudio
from transformers import pipeline
import torch
from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor
import os

model = HubertForSequenceClassification.from_pretrained("superb/hubert-large-superb-er")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/hubert-large-superb-er")

app = Flask(__name__)
app.config["UPLOAD_DIR"] = "uploaded"
@app.route('/', methods=['GET', 'POST'])
def upload():
   if request.method == 'POST':
      # try:

      audio_file = request.files['audio']
      audio_file.save(os.path.join(app.config['UPLOAD_DIR'], audio_file.filename))
      file_path = "uploaded/"+audio_file.filename
      # using librosa to load the data with sample_rate 16000 as it is recommended sample rate for the model we will be using
      waveform, sampleRate = librosa.load(file_path, sr=16000, mono=True)
      os.remove(file_path)
      interval_size = 16000 * 3
      if interval_size < len(waveform):
         noOfIntervals = int(len(waveform) / interval_size)
      else:
         noOfIntervals = 1

      emotions = {"angry": [], "happy": [], "neutral": [], "sad": []}
      for subwaveform in np.array_split(waveform, noOfIntervals):
         nwaveform = [float(number) for number in list(subwaveform)]
         # compute attention masks and normalize the waveform if needed
         inputs = feature_extractor(list(nwaveform), sampling_rate=16000, padding=True, return_tensors="pt")
         logits = model(**inputs).logits
         emotions["angry"].append(float(logits[0][2]))
         emotions["happy"].append(float(logits[0][1]))
         emotions["neutral"].append(float(logits[0][0]))
         emotions["sad"].append(float(logits[0][3]))
      x = ["happy", "angry", "sad", "neutral"]
      # since time = sampleNo/SampleRate
      times = [sampleNo / 16000 for sampleNo in range(len(waveform))]
      # splits the times list into noOfintervals number of intervals
      intervalTime = np.array_split(times, noOfIntervals)
      # select the middle point of each subarray as that represents the average time for the subarray
      middle_intervalTime = [array[len(array) // 2] for array in intervalTime]

      if len(emotions["angry"]) > 1:
         # Create the subplot for anger
         trace1 = go.Scatter(x=middle_intervalTime, y=emotions['angry'])
         subplot1 = make_subplots(rows=2, cols=2,
                                  subplot_titles=['Anger vs time', 'Happiness vs time', 'Neutral vs time',
                                                  'Sad vs time'])
         subplot1.add_trace(trace1, row=1, col=1)
         subplot1.update_xaxes(title_text='Time(s)', row=1, col=1)
         subplot1.update_yaxes(title_text='Anger', row=1, col=1)

         # Create the subplot for happiness
         trace2 = go.Scatter(x=middle_intervalTime, y=emotions['happy'])
         subplot1.add_trace(trace2, row=1, col=2)
         subplot1.update_xaxes(title_text='Time(s)', row=1, col=2)
         subplot1.update_yaxes(title_text='Happiness', row=1, col=2)

         # Create the subplot for neutral
         trace3 = go.Scatter(x=middle_intervalTime, y=emotions['neutral'])
         subplot1.add_trace(trace3, row=2, col=1)
         subplot1.update_xaxes(title_text='Time(s)', row=2, col=1)
         subplot1.update_yaxes(title_text='Neutral', row=2, col=1)

         # Create the subplot for sad
         trace4 = go.Scatter(x=middle_intervalTime, y=emotions['sad'])
         subplot1.add_trace(trace4, row=2, col=2)
         subplot1.update_xaxes(title_text='Time(s)', row=2, col=2)
         subplot1.update_yaxes(title_text='Sad', row=2, col=2)

         fig = go.Figure(data=subplot1, layout=go.Layout(title='Emotion Analysis'))

         # Display the figure
      else:
         trace = go.Bar(x=x,
                        y=[emotions["happy"][0], emotions["angry"][0], emotions["sad"][0], emotions["neutral"][0]])

         # Create layout
         layout = go.Layout(title='Fruit sales', xaxis_title='Fruit', yaxis_title='Sales')

         # Create figure
         fig = go.Figure(data=[trace], layout=layout)

      graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
      return render_template('index.html', graphJSON=graphJSON)
      # except:
      #    return "Wrong file type uploaded"
   else:
      return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)