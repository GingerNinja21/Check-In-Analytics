import pandas as pd
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
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


STOPWORDS = set(stopwords.words("english"))  # standard English stopwords
CUSTOM_STOPWORDS = {"i", "me", "my", "mine", "you", "your", "yours", "he", "she", "it", "we", "they",
    "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would", "can", "could",
    "should", "a", "an", "the", "and", "or", "but", "if", "in", "on", "for", "with",
    "as", "at", "by", "from", "about", "into", "of", "to", "up", "down", "out", "over",
    "under", "again", "further", "then", "once", "also", "like","blocker","really","able", "bit","well", "made", "led",
    "especially", "schedule", "seeing","week", "got","day","consider","nothing","whole"}  # whatever else annoys you

ALL_STOPWORDS = STOPWORDS.union(CUSTOM_STOPWORDS)

# ======== Download NLTK Data (Run Once) ========
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('vader_lexicon')

# ==============================================
# 1️⃣ Load and Clean the Dataset
# ==============================================
df_raw = pd.read_csv("reference/test_data.csv")
df_raw.columns = df_raw.columns.str.lower().str.strip()

# Keyword groups
win_keywords = ["win", "success", "achievement", "positive"]
loss_keywords = ["loss", "fail", "mistake", "negative"]
blocker_keywords = ["blocker", "issue", "problem", "challenge", "obstacle"]

# Match columns dynamically
def find_columns(df, keywords):
    return [col for col in df.columns if any(k in col for k in keywords)]

win_cols = find_columns(df_raw, win_keywords)
loss_cols = find_columns(df_raw, loss_keywords)
blocker_cols = find_columns(df_raw, blocker_keywords)

selected_cols = win_cols + loss_cols + blocker_cols
df = df_raw[selected_cols].copy()
df.columns = ["win", "loss", "blocker"]



# Cleaning
def clean_text(text):
    if pd.isna(text):
        return ""
     # keep letters, numbers, spaces, and apostrophes BETWEEN letters
    text = re.sub(r"(?<![a-zA-Z0-9])'(?![a-zA-Z0-9])", " ", text)  # remove lonely quotes
    text = re.sub(r"[^\w\s']", " ", text)  # remove all other punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text

for col in ["win", "loss", "blocker"]:
    df[col] = df[col].apply(clean_text)

# ==============================================
# 2️⃣ Tokenization, Stopwords & Lemmatization
# ==============================================
def get_top_words_and_pairs(series, n=10, stopwords=None):
    if stopwords is None:
        stopwords = set()

    if isinstance(series, pd.Series):
        series = series.dropna().astype(str).tolist()

    # CLEAN the text
    cleaned = [clean_text(t) for t in series]

    # --------------------
    #   TOKENIZE + FILTER
    # --------------------
    all_tokens = []
    cleaned_tokens_per_row = []

    for text in cleaned:
        tokens = [t for t in text.split() if t not in stopwords]
        cleaned_tokens_per_row.append(tokens)
        all_tokens.extend(tokens)

    # Top single words
    freq = Counter(all_tokens)
    top_words = dict(freq.most_common(n))

    # --------------------
    #       BIGRAMS
    # --------------------
    bigrams = []
    for tokens in cleaned_tokens_per_row:
        # Only create bigrams from non-stopword tokens
        pairs = [
            (tokens[i], tokens[i+1])
            for i in range(len(tokens)-1)
            if tokens[i] not in stopwords and tokens[i+1] not in stopwords
        ]
        bigrams.extend(pairs)
        print("Bigrams from row:", pairs)

    bigram_counts = Counter(bigrams)

    # --------------------
    #   TOP FOLLOWERS
    # --------------------
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

lemmatizer = WordNetLemmatizer()

def preprocess(text, stopwords=None):
    if stopwords is None:
        stopwords = set()

    # Clean text first
    text = clean_text(text)

    # Tokenize while keeping contractions intact
    tokens = text.split()  # split on spaces, no word_tokenize to avoid "n't" splits

    # Filter stopwords
    tokens = [t for t in tokens if t.lower() not in stopwords]

    # Lemmatize
    lemmas = [lemmatizer.lemmatize(w) for w in tokens]

    return lemmas


for col in ['win', 'loss', 'blocker']:
    df[col] = df[col].apply(lambda x: preprocess(x, stopwords=ALL_STOPWORDS))

print(df.head())

# ==============================================
# 3️⃣ Sentiment Analysis
# ==============================================
sia = SentimentIntensityAnalyzer()

def get_sentiment(tokens):
    if not tokens:
        return None
    text = " ".join(tokens)
    score = sia.polarity_scores(text)['compound']
    if score > 0.05:
        return "Positive"
    elif score < -0.05:
        return "Negative"
    return "Neutral"

df['win_sentiment'] = df['win'].apply(get_sentiment)
df['loss_sentiment'] = df['loss'].apply(get_sentiment)

# ==============================================
# 4️⃣ Word Cloud Visualization (Seaborn Style)
# ==============================================
win_text = " ".join([" ".join(x) for x in df["win"]])
loss_text = " ".join([" ".join(x) for x in df["loss"]])
blocker_text = " ".join([" ".join(x) for x in df["blocker"]])

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




# ==============================================
# 6️⃣ Tkinter Dashboard Setup
# ==============================================
def create_wordpair_frame(root, top_words, pairs, title):
    frame = ttk.Frame(root)
    fig, ax = plt.subplots(figsize=(7, 5))
    pair_labels = [f"{w} → {pairs[w]}" if pairs[w] else w for w in top_words.keys()]
    sns.barplot(x=list(top_words.values()), y=pair_labels, ax=ax, palette="mako")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Word → Partner")
    plt.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)
    return frame

def show_word_pairs():
    main_frame.pack_forget()
    word_frame = ttk.Frame(root)
    word_frame.pack(fill="both", expand=True)

    nb = ttk.Notebook(word_frame)
    nb.pack(fill="both", expand=True)
    
    top_wins, win_pairs = get_top_words_and_pairs(df['win'],stopwords=ALL_STOPWORDS)
    top_losses, loss_pairs = get_top_words_and_pairs(df['loss'],stopwords=ALL_STOPWORDS)
    top_blockers, blocker_pairs = get_top_words_and_pairs(df['blocker'])


    frames = {
        "Wins": (top_wins, win_pairs),
        "Losses": (top_losses, loss_pairs),
        "Blockers": (top_blockers, blocker_pairs)
    }

    for title, (top_words, pairs) in frames.items():
        tab = create_wordpair_frame(nb, top_words, pairs, title)
        nb.add(tab, text=title)

    ttk.Button(word_frame, text="⬅ Back to Dashboard",
               command=lambda: [word_frame.pack_forget(), main_frame.pack(fill='both', expand=True)]
               ).pack(pady=10)

# ==============================================
# 7️⃣ Launch Tkinter App
# ==============================================
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
