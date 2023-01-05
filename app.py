from youtube_transcript_api import YouTubeTranscriptApi, VideoUnavailable, TooManyRequests
from youtube_transcript_api.formatters import TextFormatter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from flask import Flask,jsonify, request
from pytube import YouTube, extract
import os
import sys
import nltk
import speech_recognition as sr
r = sr.Recognizer()
from moviepy.editor import *

from summarize import sumy_lsa_summarize, sumy_text_rank_summarize, transformers_summarize

# Creating Flask Object and returning it.
app = Flask(__name__)

# "Punkt" download before nltk tokenization
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print('Downloading punkt')
    nltk.download('punkt', quiet=True)

# "Wordnet" download before nltk tokenization
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    print('Downloading wordnet')
    nltk.download('wordnet')

# "Stopwords" download before nltk tokenization
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print('Downloading Stopwords')
    nltk.download("stopwords", quiet=True)

#Summarization for Videos without Subtitles
@app.route('/transcription/<path:link>',methods=['GET', 'POST'])
def transcription(link):
    link=link.replace('/watch?v=','/')
    video = YouTube(link)
    yt = video.streams.get_lowest_resolution()
    yt.download(filename = 'audio.mp4')
    
    video_file=VideoFileClip(r"audio.mp4")
    video_file.audio.write_audiofile(r"audio.wav")
    
    audio=sr.AudioFile('audio.wav')
    with audio as source:
        audio_text = r.record(source)
    
    # recoginize_() method will throw a request error if the API is unreachable, hence using exception handling
    try:
        # using google speech recognition
        formatted_text = r.recognize_google(audio_text)
        os.remove('audio.mp4')
        os.remove('audio.wav')
        summary = sumy_lsa_summarize(formatted_text)  # Sumy for extractive summary using LSA.
        #summary = sumy_text_rank_summarize(formatted_text)  # Sumy for Text Rank Based Summary.
        #summary=transformers_summarize(formatted_text) #Hugging Face summarization using Transformers.

        # Returning Result
        response_list = {'processed_summary': summary,}
        
        X_list = word_tokenize(formatted_text) 
        Y_list = word_tokenize(summary)
  
        # sw contains the list of stopwords
        sw = stopwords.words('english') 
        l1 =[];l2 =[]
  
        # remove stop words from the string
        X_set = {w for w in X_list if not w in sw} 
        Y_set = {w for w in Y_list if not w in sw}
  
        # form a set containing keywords of both strings 
        rvector = X_set.union(Y_set) 
        for w in rvector:
            if w in X_set: l1.append(1) # create a vector
            else: l1.append(0)
            if w in Y_set: l2.append(1)
            else: l2.append(0)
        c = 0
  
        # cosine formula 
        for i in range(len(rvector)):
            c+= l1[i]*l2[i]
            cosine = c / float((sum(l1)*sum(l2))**0.5)
                
        return jsonify(success=True,
                        message="Subtitles for this video was fetched and summarized successfully.",
                        response=response_list,similarity=cosine), 200
        
    except Exception as e :
        print(e)
        return 'Sorry Run again'
    
    
#Summarization for Videos with Subtitles
@app.route('/summarize/<path:url>', methods=['GET'])
def transcript_fetched_query(url):
    #url=url.replace('/watch?v=','/')
    id=extract.video_id(url)
        
    if id:
        try:
            # Using Formatter to store and format received subtitles properly.
            formatter = TextFormatter()
            transcript = YouTubeTranscriptApi.get_transcript(id)
            formatted_text = formatter.format_transcript(transcript).replace("\n", " ")

            summary = sumy_lsa_summarize(formatted_text)  # Sumy for extractive summary using LSA.
            #summary = sumy_text_rank_summarize(formatted_text)  # Sumy for Text Rank Based Summary.
            #summary=transformers_summarize(str(transcript)) #Hugging Face summarization using Transformers.

                    # Returning Result
            response_list = {'processed_summary': summary}

            return jsonify(success=True,
                            message="Subtitles for this video was fetched and summarized successfully.",
                            response=response_list), 200

        # Catching Exceptions
        except VideoUnavailable:
            return jsonify(success=False, message="VideoUnavailable: The video is no longer available.",
                                response=None), 400
        except TooManyRequests:
            return jsonify(success=False,
                                message="TooManyRequests: YouTube is receiving too many requests from this IP."
                                        " Wait until the ban on server has been lifted.",
                                response=None), 500

            
    elif id is None or len(id) <= 0:
        # video_id parameter doesn't exist in the request.
        return jsonify(success=False,
                        message="Video ID is not present in the request. "
                                "Please check that you have added id in your request correctly.",
                        response=None), 400
        
    else:
        # Some another edge case happened. Return this message for preventing exception throw.
        return jsonify(success=False,
                        message="Please request the server with your arguments correctly.",
                        response=None), 400


if __name__=="__main__":
    app.run(debug=True)
