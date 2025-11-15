import pandas as pd
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg



import os


if os.path.exists("requirements.txt"):
    os.system("pip install -r requirements.txt")


else:
    print("requirements.txt not found.")

# Download NLTK datasets (RUN THIS ONCE AND COMMENT OUT AFTER)
nltk.download(['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4', 'vader_lexicon'])

# Initialize Stopwords
STOPWORDS = set(stopwords.words("english"))  
CUSTOM_STOPWORDS = {"i", "me", "my", "mine", "you", "your", "yours", "he", "she", "it", "we", "they",
    "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would", "can", "could",
    "should", "a", "an", "the", "and", "or", "but", "if", "in", "on", "for", "with",
    "as", "at", "by", "from", "about", "into", "of", "to", "up", "down", "out", "over",
    "under", "again", "further", "then", "once", "also", "like","blocker","really","able", "bit","well", "made", "led",
    "especially", "schedule", "seeing","week", "got","day","consider","nothing","whole"}  

ALL_STOPWORDS = STOPWORDS.union(CUSTOM_STOPWORDS)

#=============================================
# Loading and Cleaning the Dataset
#=============================================
dataframe_raw = pd.read_csv("reference/test_data.csv")
dataframe_raw.columns = dataframe_raw.columns.str.lower().str.strip()

# Find columns based on keywords
win_keywords = ["win", "success", "achievement", "positive"]
loss_keywords = ["loss", "fail", "mistake", "negative"]
blocker_keywords = ["blocker", "issue", "problem", "challenge", "obstacle"]

def find_columns(df, keywords):
    return [col for col in df.columns if any(k in col for k in keywords)]

win_col= find_columns(dataframe_raw, win_keywords)
loss_col = find_columns(dataframe_raw, loss_keywords)   
blocker_col = find_columns(dataframe_raw, blocker_keywords)
selected_columns = win_col + loss_col + blocker_col

dataframe = dataframe_raw[selected_columns].copy()

# Renamed Columns
dataframe.columns = ["win", "loss", "blocker"]

# Cleaning Function
def clean_text(text):
    if pd.isna(text):
        return ""
    # Regex
    text = re.sub(r"(?<![a-zA-Z0-9])'(?![a-zA-Z0-9])", " ", text)  # remove lonely quotes
    text = re.sub(r"[^\w\s']", " ", text)  # remove all other punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text

for col in ["win", "loss", "blocker"]:
    dataframe[col] = dataframe[col].apply(clean_text)

   

#=============================================
# Tokenization & Lemmatization
#=============================================

# Lemmatization
lemmatizer = WordNetLemmatizer()

# Preprocessing Function
def preprocess(text, stopwords=ALL_STOPWORDS):
    text = clean_text(text)
    tokens = text.split()

    # Apply Stopwords
    tokens = [t for t in tokens if t.lower() not in stopwords]
    lemmatized = [lemmatizer.lemmatize(w) for w in tokens]

    return lemmatized

for col in ['win', 'loss', 'blocker']:
    dataframe[col] = dataframe[col].apply(lambda x: preprocess(x, stopwords=ALL_STOPWORDS))

    print(dataframe.head())

#=============================================
# Getting Top Words and Pairs
#=============================================
def get_top_words_and_pairs(series, n=10, stopwords=ALL_STOPWORDS):
    if isinstance(series, pd.Series):
        series = series.dropna().astype(str).tolist()

    cleaned = [clean_text(t) for t in series]

   
    # TOKENIZE and FILTER
    all_tokens = []
    cleaned_tokens_per_row = []

    for text in cleaned:
        tokens = [t for t in text.split() if t not in stopwords]
        cleaned_tokens_per_row.append(tokens)
        all_tokens.extend(tokens)

    # Most Frequent single words
    freq = Counter(all_tokens)
    top_words = dict(freq.most_common(n))

    #       BIGRAMS
    bigrams = []
    for tokens in cleaned_tokens_per_row:
        pairs = [
            (tokens[i], tokens[i+1])
            for i in range(len(tokens)-1)
            if tokens[i] not in stopwords and tokens[i+1] not in stopwords
        ]
        bigrams.extend(pairs)

    bigram_counts = Counter(bigrams)

    #   TOP FOLLOWERS
    top_pairs = {}
    for word in top_words:
        candidates = {
            pair: count
            for pair, count in bigram_counts.items()
            if pair[0] == word and pair[1] not in stopwords
        }

        if candidates:
            best_pair = max(candidates.items(), key=lambda x: x[1])[0]
            top_pairs[word] = best_pair[1]
        else:
            top_pairs[word] = None

    return top_words, top_pairs

get_top_words_and_pairs(dataframe['win'], stopwords=ALL_STOPWORDS)
get_top_words_and_pairs(dataframe['loss'], stopwords=ALL_STOPWORDS)
get_top_words_and_pairs(dataframe['blocker'], stopwords=ALL_STOPWORDS)

#=============================================
# Sentiment Analysis
#=============================================

Sent_Analysis = SentimentIntensityAnalyzer()

def analyze_sentiment(tokens):
    if not tokens:
        return None
    text = " ".join(tokens)
    scores = Sent_Analysis.polarity_scores(text)['compound']

    if scores >= 0.05:
        return "Positive"
    elif scores <= -0.05:
        return "Negative"
    else:  
        return "Neutral"
    
# Apply Sentiment Analysis to each column
for col in ['win', 'loss', 'blocker']:
    dataframe[f"{col}_sentiment"] = dataframe[col].apply(analyze_sentiment)
print(dataframe.head())

#=============================================
# Word Cloud Visualization
#=============================================
win_text = " ".join([" ".join(x) for x in dataframe["win"]])
loss_text = " ".join([" ".join(x) for x in dataframe["loss"]])
blocker_text = " ".join([" ".join(x) for x in dataframe["blocker"]])

def generate_wordcloud():
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, text, title in zip(axes,
                               [win_text, loss_text, blocker_text],
                               ["Wins Word Cloud", "Losses Word Cloud", "Blockers Word Cloud"]):
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        ax.set_title(title, fontsize=18)
    plt.tight_layout()
    plt.show()


#=============================================
# Tkinter Dashboard Setup
#=============================================


def wordpair_frame(root, top_words, pairs, title):
    frame = ttk.Frame(root)
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = [f"{w} → {pairs[w]}" if pairs[w] else w for w in top_words.keys()]
    sns.barplot(x=list(top_words.values()), y=labels, ax=ax, palette="mako")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Words → Commonly Paired Word")

    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

    return frame

def plot_wordclouds():
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, text, title in zip(
        axes,
        [win_text, loss_text, blocker_text],
        ["Wins", "Losses", "Blockers"]
    ):
        wc = WordCloud(width=800, height=400, background_color='black', colormap='viridis').generate(text)
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        ax.set_title(title, fontsize=18, color='gold')
    plt.tight_layout()
    plt.show()


def show_word_pairs():
    main_frame.pack_forget()
    word_frame = ttk.Frame(root)
    word_frame.pack(fill="both", expand=True )

    nb = ttk.Notebook(word_frame)
    nb.pack(fill="both", expand=True)
    
    top_wins, win_pairs = get_top_words_and_pairs(dataframe['win'],stopwords=ALL_STOPWORDS)
    top_losses, loss_pairs = get_top_words_and_pairs(dataframe['loss'],stopwords=ALL_STOPWORDS)
    top_blockers, blocker_pairs = get_top_words_and_pairs(dataframe['blocker'])


    frames = {
        "Wins": (top_wins, win_pairs),
        "Losses": (top_losses, loss_pairs),
        "Blockers": (top_blockers, blocker_pairs)
    }

    for title, (top_words, pairs) in frames.items():
        tab = wordpair_frame(nb, top_words, pairs, title)
        nb.add(tab, text=title  )

    ttk.Button(word_frame, text="⬅ Back to Dashboard",
               command=lambda: [word_frame.pack_forget(), main_frame.pack(fill='both', expand=True)]
               ).pack(pady=10)



# initialize Tkinter
root = tk.Tk()
root.title("Data Insights Dashboard")
root.geometry("1200x700")

main_frame = ttk.Frame(root)
main_frame.pack(fill="both", expand=True)

ttk.Label(main_frame, text="📊 Data Visualization Dashboard",
        font=("Segoe UI", 18, "bold")).pack(pady=20)
ttk.Button(main_frame, text="Show Word Clouds", command=plot_wordclouds).pack(pady=10)
ttk.Button(main_frame, text="View Word Partnerships", command=show_word_pairs).pack(pady=20)


root.mainloop()