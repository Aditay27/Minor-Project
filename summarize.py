# NLTK Imports
from nltk.tokenize import sent_tokenize

# Sumy Imports
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

from transformers import pipeline


def sumy_lsa_summarize(text_content):
    # Latent Semantic Analysis is a unsupervised learning algorithm that can be used for extractive text summarization.
    # Initializing the parser
    parser = PlaintextParser.from_string(text_content, Tokenizer("english"))
    # Initialize the stemmer
    stemmer = Stemmer('english')
    # Initializing the summarizer
    summarizer = LsaSummarizer(stemmer)
    summarizer.stop_words = get_stop_words('english')

    # Finding number of sentences and applying percentage on it: since sumy requires number of lines
    #sentence_token = sent_tokenize(text_content)
    #select_length = int(len(sentence_token) * (int(percent) / 100))

    # Evaluating and saving the Summary
    summary = ""
    for sentence in summarizer(parser.document, 2):
        summary += str(sentence)
    # Returning NLTK Summarization Output
    return summary

def sumy_text_rank_summarize(text_content):
    # TextRank is an unsupervised text summarization technique that uses the intuition behind the PageRank algorithm.
    # Initializing the parser
    parser = PlaintextParser.from_string(text_content, Tokenizer("english"))
    # Initialize the stemmer
    stemmer = Stemmer('english')
    # Initializing the summarizer
    summarizer = TextRankSummarizer(stemmer)
    summarizer.stop_words = get_stop_words('english')

    #sentence_token = sent_tokenize(text_content)
    #select_length = int(len(sentence_token)* (int(percent) / 100))

    # Evaluating and saving the Summary
    summary = ""
    for sentence in summarizer(parser.document, 1):
        summary += str(sentence)
    # Returning NLTK Summarization Output
    return summary

def transformers_summarize(text_content):
    #parser = PlaintextParser.from_string(text_content, Tokenizer("english"))
    
    summarization = pipeline("summarization")
    #summarization.stop_words = get_stop_words('english')
    
    #sentence_token = sent_tokenize(text_content)
    #select_length = int(len(sentence_token))
    
    summary = summarization(text_content)[0]['summary_text']
    #for sentence in summarization(parser.document, sentences_count=select_length):
        #summary += str(sentence)
    return summary



