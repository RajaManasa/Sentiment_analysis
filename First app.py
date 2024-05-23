from flask import Flask, render_template, request
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


app = Flask(__name__)

@app.route('/', methods = ['GET','POST'])
def SpamAnalysis():
    if request.method == 'POST':
        sentiment_text = request.form.get("sentiment_text")
        sid = SentimentIntensityAnalyzer()
        score = sid.polarity_scores(sentiment_text)
        if score['neg'] != 0:
            return render_template('index.html', message="Negative")
        else:
            return render_template('index.html', message="Positive")

    return render_template('index.html')


if __name__ == '__main__':
    app.run(port=3000, debug='True')