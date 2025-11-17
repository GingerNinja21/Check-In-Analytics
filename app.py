import os
import re
import sys
from collections import Counter

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from nltk.corpus import stopwords, wordnet
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import tkinter as tk
from tkinter import ttk


THEMES = {
    # ============================================================
    # PERFORMANCE & PRODUCTIVITY
    # ============================================================
    "productivity": [
        "productive", "efficient", "efficiency", "focus", "focused",
        "time", "timely", "deadline", "deadlines", "progress",
        "workflow", "workload", "task", "tasks", "todo", "complete",
        "completion", "organized", "structure", "planning",
        "planner", "executing", "deliver", "results", "output",
    ],
    "performance issues": [
        "slow", "slowness", "lag", "laggy", "lagging", "delay",
        "delayed", "delays", "unresponsive", "crash", "crashed",
        "crashes", "freeze", "freezing", "timeout", "bottleneck",
        "downtime", "fault", "glitch", "bug", "bugs", "broken",
        "unstable", "issues", "problem", "problems", "failure",
    ],
    "motivation & energy": [
        "motivated", "motivation", "inspired", "enthusiastic",
        "energized", "energy", "driven", "pumped", "committed",
        "commitment", "hyped", "positive", "determined",
        "disciplined",
    ],

    # ============================================================
    # LEARNING, TRAINING, GROWTH
    # ============================================================
    "learning & development": [
        "learn", "learning", "studying", "study", "course",
        "courses", "module", "modules", "lesson", "lessons",
        "tutorial", "tutorials", "training", "improve",
        "improvement", "skill", "skills", "grow", "growth",
        "datacamp", "education", "knowledge", "develop",
        "development", "practice", "revision", "research",
    ],
    "confidence & mastery": [
        "confident", "confidence", "comfortable", "mastered",
        "mastery", "competent", "capable", "clarity", "clear",
        "understanding", "know", "familiar", "skilled",
    ],
    "overwhelmed": [
        "overwhelmed", "confusing", "confused", "lost", "struggling",
        "struggle", "difficult", "difficulty", "stress", "pressure",
        "too much", "stuck", "hard", "frustrated", "frustration",
        "panic", "chaos", "overload",
    ],

    # ============================================================
    # COMMUNICATION & TEAMWORK
    # ============================================================
    "team communication": [
        "communicate", "communication", "communicating", "talk",
        "talking", "speak", "speaking", "reply", "replies",
        "respond", "response", "responses", "chat", "discussion",
        "interact", "interaction", "email", "emails", "message",
        "messages", "call", "calls", "meeting", "meetings",
    ],
    "collaboration": [
        "team", "help", "support", "assist", "together",
        "cooperate", "cooperation", "guidance", "partner",
        "partnership", "collab", "colleague", "colleagues",
        "group", "teamwork", "buddy", "pair",
    ],
    "management & expectations": [
        "manager", "lead", "leader", "leadership", "feedback",
        "expectations", "clarity", "instruction", "instructions",
        "direction", "directionless", "supervisor", "oversight",
        "organize", "assigned", "assignment", "expecting",
    ],

    # ============================================================
    # TECHNICAL / ENVIRONMENT ISSUES
    # ============================================================
    "network issues": [
        "network", "wifi", "wi-fi", "connection", "connectivity",
        "disconnect", "disconnected", "offline", "internet",
        "router", "bandwidth", "signal", "drops", "unstable",
        "latency", "ping", "packet", "timeout", "weak signal",
    ],
    "software problems": [
        "software", "error", "errors", "bug", "bugs", "install",
        "installation", "update", "updates", "crash", "crashes",
        "patch", "framework", "version", "code", "coding", "script",
        "failure", "broken", "unexpected", "malfunction",
    ],
    "hardware problems": [
        "hardware", "pc", "laptop", "computer", "keyboard", "mouse",
        "screen", "monitor", "speaker", "device", "broken",
        "damaged", "faulty", "plug", "cable", "usb", "headphones",
        "battery", "charger", "power", "overheating",
    ],
    "system access issues": [
        "login", "log in", "sign in", "access", "blocked",
        "permission", "permissions", "password", "credentials",
        "authorized", "authorization", "unauthorized", "locked",
        "lockout", "verification",
    ],

    # ============================================================
    # EMOTIONAL & WELLBEING
    # ============================================================
    "stress & anxiety": [
        "stress", "stressed", "anxious", "anxiety", "pressure",
        "burnout", "tired", "exhausted", "drained", "mental",
        "worried", "tense", "nervous", "panicked", "overthinking",
        "depressed",
    ],
    "positive mood": [
        "excited", "happy", "joy", "good", "great", "amazing",
        "satisfied", "glad", "grateful", "optimistic", "hopeful",
        "relieved", "positive", "uplifted", "better", "pleasant",
    ],
    "lack of motivation": [
        "demotivated", "tired", "drained", "bored", "uninspired",
        "unmotivated", "lazy", "sluggish", "low energy",
        "can't focus", "don't feel like", "exhaustion",
    ],

    # ============================================================
    # PROCESS & WORKFLOW PROBLEMS
    # ============================================================
    "planning & time management": [
        "plan", "planning", "time", "schedule", "scheduling",
        "organize", "organization", "prioritize", "delay",
        "delayed", "late", "lateness", "reschedule", "timing",
        "deadline", "deadlines",
    ],
    "requirements unclear": [
        "unclear", "unsure", "vague", "confusing", "ambiguity",
        "missing", "unknown", "not sure", "no idea", "need clarity",
        "lack of information",
    ],
    "workload issues": [
        "too much", "overload", "overloaded", "busy", "hectic",
        "swamped", "pressure", "pileup", "many tasks", "high load",
        "struggling to keep up",
    ],

    # ============================================================
    # PERSONAL DEVELOPMENT
    # ============================================================
    "self improvement": [
        "improve", "improvement", "growth", "develop", "development",
        "progress", "better", "enhance", "enhancing", "practice",
        "reflect", "self-aware", "upgrade", "level up",
    ],
    "discipline & habits": [
        "habit", "habits", "routine", "discipline", "consistent",
        "consistency", "pattern", "practice", "daily", "ritual",
        "habit building", "commitment",
    ],
}


if os.path.exists("requirements.txt"):
    os.system("pip install -r requirements.txt")


else:
    print("requirements.txt not found.")

# Download NLTK datasets (RUN THIS ONCE AND COMMENT OUT AFTER)
nltk.download(['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4', 'vader_lexicon'])

# Initialize Stopwords - These will Be use to filter our data in order to only analyse the words that matter
STOPWORDS = set(stopwords.words("english"))  
CUSTOM_STOPWORDS = {"i", "me", "my", "mine", "you", "your", "yours", "he", "she", "it", "we", "they",
    "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would", "can", "could",
    "should", "a", "an", "the", "and", "or", "but", "if", "in", "on", "for", "with",
    "as", "at", "by", "from", "about", "into", "of", "to", "up", "down", "out", "over",
    "under", "again", "further", "then", "once", "also", "like","blocker","really","able", "bit","well", "made", "led",
    "especially", "schedule", "seeing","week", "got","day","consider","nothing","whole","blocker","honour", "moment", "right" , "result"}  

ALL_STOPWORDS = STOPWORDS.union(CUSTOM_STOPWORDS)

#===============================================================================================================#
# Loading and Cleaning the Dataset                                                                              #
#===============================================================================================================#
# - This section loads the raw dataset                                                                          #
# - Uses keyword matching in order to identify win/loss/blocker columns.                                        #
# - Extracts these columns into a clean dataframe.                                                              #
# - Removes null values.                                                                                        #
# - Applies regex in order to remove unwanted symbols, punctuation and unnecessary spacing from the dataset.    #
#===============================================================================================================#

dataframe_raw = pd.read_csv("Copy of Umuzi XB1 Check in (Responses) - Form Responses 1.csv")
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
    text = re.sub(r"(?<![a-zA-Z0-9])'(?![a-zA-Z0-9])", " ", text) 
    text = re.sub(r"[^\w\s']", " ", text)  
    text = re.sub(r"\s+", " ", text).strip()
    return text

for col in ["win", "loss", "blocker"]:
    dataframe[col] = dataframe[col].apply(clean_text)

   

#===================================================================================#
# Tokenization & Lemmatization                                                      #
#===================================================================================#
# - This section prepocesses text by tokenizing and lemmatizing the cleaned text    #
# - Converts entries into a list of words.                                          #
# - Removes stopwords and reduces each token to it's base form via lemmatization.   #
# - Stores the tokens back into the win/loss/blocker columns of the dataframe       #
#===================================================================================#

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

#========================================================================================================#
# Getting Top Words and Pairs                                                                            #
#========================================================================================================#
# - This section identifies the most frequent words along with it's bigrams (word-pairs)                 #
# - Cleans and tokenizes the text and filters out stopwords again.                                       #
# - Counts how often each word appears and determines which word most frequently follows each top word.  #
#========================================================================================================#
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

#================================================================================================#
# Sentiment Analysis                                                                             #
#================================================================================================#
# - This section focuses on running sentiment analysis on the tokenized text.                    #
# - Uses VADER fir sentiment analysis                                                            #
# - Converts tokenized list back into a full sentence in order to get accurate sentiment score.  #
# - Classifies each entry as Positive, Neutral or Negative.                                      #
#================================================================================================#

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

#===============================================================================#
# Word Cloud Cluster                                                            #
#===============================================================================#
# - This section switches from the dashboard window to the word-cloud window.   #
# - Generates 3 WordCloud visuals for each column (win/loss/blocker).           #
# - Embed the figure inside a Tkinter frame.                                    #
# - The figure is then rendered inside the app.                                 #
#===============================================================================#

win_text = " ".join([" ".join(x) for x in dataframe["win"]])
loss_text = " ".join([" ".join(x) for x in dataframe["loss"]])
blocker_text = " ".join([" ".join(x) for x in dataframe["blocker"]])

def generate_wordcloud():
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, text, title in zip(axes,
                               [win_text, loss_text, blocker_text],
                               ["Wins Word Cloud", "Losses Word Cloud", "Blockers Word Cloud"]):
        wordcloud = WordCloud(width=800, height=400, background_color='#59a5d8').generate(text)
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        ax.set_title(title, fontsize=18)
    plt.tight_layout() 
    


def plot_wordclouds():
    main_frame.pack_forget()
    bg_colors = ["forestgreen", "white", "midnightblue"]  
    word_cloud_frame = tk.Frame(root, bg="#000000")
    word_cloud_frame.pack(fill='both', expand=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 10))

    for ax, text, title, bg,  in zip(
        axes,
        [win_text, loss_text, blocker_text],
        ["Wins", "Losses", "Blockers"],
        bg_colors,
    ):
        wc = WordCloud(
            width=800,
            height=800,
            background_color=bg,
        ).generate(text)

        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        
        ax.text(
            0.5, 0.5,
            title,
            fontsize=22,
            fontweight="bold",
            color="white" ,
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="#FE7F2D",
                edgecolor="#ffffff",
                linewidth=2,
                alpha=0.7
            )
        )
        
    btn_frame = tk.Frame(word_cloud_frame, bg="#000000")
    btn_frame.pack(side="bottom", pady=10)
    tk.Button(btn_frame, text="⬅ Back to Dashboard",font=("Lucida",15,), fg="#2D728F",background="#ffffff", command=lambda: [word_cloud_frame.pack_forget(), main_frame.pack(fill='both', expand=True)]).pack()

    fig.set_facecolor(color="#000000")
    plt.tight_layout()
    canvas = FigureCanvasTkAgg(fig, master=word_cloud_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)
    
#=====================================================================================#
# Word Frequency Analysis                                                             #
#=====================================================================================#
# - This section performs topic modeling using LDA                                    #
# - Performs word-pair frequency analysis                                             #
# - Builds bar charts of the most common words along with thier strongest word-pair.  #
# - Runs LDA on raw text in order to extract common themes.                           #
# - Displays results in a Tkinter notebook.                                           #
#=====================================================================================#
# def run_lda_on_series(series, num_topics=20, num_words=20,tab="wins"):
#     """
#     Run LDA on tokenized pandas Series and return the single dominant theme.
#     Wins: only positive themes
#     Loss/Blocker: only negative/neutral themes
#     """
#     docs = [" ".join(tokens) for tokens in series if isinstance(tokens, list)]
#     if len(docs) < 2:
#         return "Not enough data for LDA"
    
#     stopwords = list(ALL_STOPWORDS) + [
#         "challenging", "even", "one", "enjoyed", "managed", "lot",
#         "went", "everything", "last", "felt", "issue", "none", "loss",
#         "still", "make"
#     ]

#     vectorizer = CountVectorizer(stop_words=stopwords)
#     X = vectorizer.fit_transform(docs)

#     lda = LatentDirichletAllocation(
#         n_components=num_topics,
#         max_iter=20,
#         learning_method='online',
#         random_state=42
#     )
#     lda.fit(X)

#     terms = vectorizer.get_feature_names_out()
#     top_words = []
#     for topic in lda.components_:
#         words = [terms[i] for i in topic.argsort()[-num_words:]]
#         top_words.extend(words)

#     # -------------------------
#     # Theme Mapping
#     # -------------------------
#     positive_themes = {
#         "Learning & Development": [
#             "learn", "learning", "course", "datacamp", "plan", "study", 
#             "practice", "improve", "skill", "knowledge", "training", 
#             "develop", "achieve", "master", "understand", "enjoyed", 
#             "completed", "success", "growth"
#         ],
#         "Collaboration & Teamwork": [
#             "team", "collaboration", "partner", "together", "helped",
#             "supported", "mentored", "guided"
#         ],
#         "Personal Achievement": [
#             "win", "success", "goal", "milestone", "completed", "finished",
#             "achievement", "challenge", "progress"
#         ],
#         "Positive Feedback & Recognition": [
#             "appreciated", "recognized", "acknowledged", "praise",
#             "rewarded", "complimented", "encouraged"
#         ]
#     }

#     negative_themes = {
#         "Network & Tech Issues": ["network", "issue", "problem", "error", "bug", "connectivity"],
#         "Workload & Time Pressure": ["stress", "deadline", "pressure", "overwhelmed", "task", "busy"],
#         "Learning Obstacles": ["confused", "stuck", "challenging", "difficult", "hard", "lost"],
#         "Team & Communication Problems": ["miscommunication", "delay", "conflict", "misunderstood", "ignored"]
#     }

#     themes_dict = positive_themes if tab == "wins" else negative_themes

#     # Count theme hits
#     theme_counts = {theme:0 for theme in themes_dict}
#     for word in top_words:
#         for theme, words in themes_dict.items():
#             if word in words:
#                 theme_counts[theme] += 1

#     # Return dominant theme
#     dominant_theme = max(theme_counts, key=theme_counts.get)
#     return dominant_theme if theme_counts[dominant_theme] > 0 else "General"


def wordpair_frame(root, top_words, pairs, title, raw_series):
    main_frame.pack_forget()
    wordpair_frame = tk.Frame(root, bg="black")
    wordpair_frame.columnconfigure(0, weight=1)
    wordpair_frame.columnconfigure(1, weight=1)
    wordpair_frame.grid_rowconfigure(0, weight=1)
    wordpair_frame.grid_rowconfigure(1, weight=5)

    # ----------------------
    # LEFT SIDE — BAR GRAPH
    # ----------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [f"{w} → {pairs[w]}" if pairs[w] else w for w in top_words.keys()]
    sns.barplot(x=list(top_words.values()), y=labels, ax=ax, palette="mako")
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Frequency", fontsize=16, color="#FE7F2D")
    ax.set_ylabel("Top Word → Pair", fontsize=16, color="#FE7F2D")

    fig.tight_layout()
    canvas = FigureCanvasTkAgg(fig, master=wordpair_frame)
    tk.Label(wordpair_frame, anchor="center", font=("Lucida", 20, "bold"),
             fg="#FE7F2D", bg="#000000", text="Word Frequency Analysis").grid(row=0, column=0)
    canvas.draw()
    canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

    # ----------------------
    # RIGHT SIDE — DOMINANT THEMES
    # ----------------------
    topic_box = tk.Text(wordpair_frame, width=40, height=20, wrap="word",
                        font=("Lucida", 15), bg="#FE7F2D", fg="#000000")
    topic_box.grid(row=1, column=1, sticky="nsew", padx=10, pady=10)

   

    topic_box.tag_configure("big_bold", font=("Lucida", 18, "bold"), foreground="#000000")
    topic_box.insert("end", f"Dominant Theme\n", "big_bold")
    tokens = [word for row in raw_series if isinstance(row, list) for word in row]
    
    themes = LDA(tokens, THEMES, top_n=5)

    top_5 = themes[:5]  
    for name, score in top_5:
        topic_box.insert("end", f"{name.upper()}\n ({score} mentions)\n\n")

    topic_box.configure(state="disabled")

    return wordpair_frame

def LDA(tokens, themes_dict, top_n=5):
    # This function takes a list of tokens and filters through  # 
    # a dictionary of themes in order to return the Top 5 Themes#
    # NOTE: This is based of word occurence rather ran LDA      #
  
    token_counts = Counter(tokens)
    theme_scores = {}

    for theme, keywords in themes_dict.items():
        score = sum(token_counts.get(word, 0) for word in keywords)
        theme_scores[theme] = score

    # Sort themes from highest score to lowest
    sorted_themes = sorted(theme_scores.items(), key=lambda x: x[1], reverse=True)

    # Filter out zero-score themes
    sorted_themes = [t for t in sorted_themes if t[1] > 0]

    # Return only the top N
    return sorted_themes[:top_n]

def show_word_pairs():
    main_frame.pack_forget()
    #Styling
    style = ttk.Style()
    style.configure("TNotebook", background="black")
    style.configure("TNotebook.Tab",
                    background="#000000",
                    foreground="#FE7F2D",
                    font=("Lucida", 12, "bold"),
                    padding=[12, 6])
    style.map("TNotebook.Tab",
            background=[("selected", "#000000")],
            foreground=[("selected", "#FE7F2D")])

    
    word_frame = tk.Frame(root, bg="#000000")
    word_frame.pack(fill="x" ,expand=True )
    
    wordpair_graph = ttk.Notebook(word_frame,style="TNotebook")
    wordpair_graph.pack(fill="x" , expand=True)
    
    top_wins, win_pairs = get_top_words_and_pairs(dataframe['win'],stopwords=ALL_STOPWORDS)
    top_losses, loss_pairs = get_top_words_and_pairs(dataframe['loss'],stopwords=ALL_STOPWORDS)
    top_blockers, blocker_pairs = get_top_words_and_pairs(dataframe['blocker'], stopwords=ALL_STOPWORDS)


    frames = {
        "Wins": (top_wins, win_pairs),
        "Losses": (top_losses, loss_pairs),
        "Blockers": (top_blockers, blocker_pairs)
    }

    column_map = {"Wins": "win", "Losses": "loss", "Blockers": "blocker"}

    for title, (top_words, pairs) in frames.items():
        tab = wordpair_frame(wordpair_graph, top_words, pairs, title, dataframe[column_map[title]])
        wordpair_graph.add(tab, text=title ,padding=10 )


    ttk.Button(word_frame, text="⬅ Back to Dashboard",
               command=lambda: [word_frame.pack_forget(), main_frame.pack(fill='both', expand=True)]
               ).pack(pady=10)

    update_counters()





#==========================================================================#
# Dashboard widgets                                                        #
#==========================================================================#
#  This section updates the dashboard’s summary counters by computing:     #
# (1) how many top words were found across Wins/Losses/Blockers,           #
# (2) how many LDA-derived trends were discovered, and                     #
# (3) the overall sentiment by comparing total positive vs negative        #
# entries. The values are then pushed directly into the dashboard          #
# labels to keep the UI in sync with the latest analysis.                  #
#==========================================================================#

def update_counters():
    # Top words discovered across all themes
    top_wins, _ = get_top_words_and_pairs(dataframe['win'])
    top_losses, _ = get_top_words_and_pairs(dataframe['loss'])
    top_blockers, _ = get_top_words_and_pairs(dataframe['blocker'])
    total_top_words = len(top_wins) + len(top_losses) + len(top_blockers)

    # Entries Received
    total_entries = len(dataframe)

    # Overall sentiment: just count positives vs negatives
    pos_count = sum(dataframe[col+"_sentiment"].value_counts().get("Positive", 0)
                    for col in ['win','loss','blocker'])
    neg_count = sum(dataframe[col+"_sentiment"].value_counts().get("Negative", 0)
                    for col in ['win','loss','blocker'])
    overall_sentiment = "Positive" if pos_count >= neg_count else "Negative"

    # Update labels
    lbl_top_words.config(text=str(total_top_words))
    lbl_entries.config(text=str(total_entries))
    lbl_sentiment.config(text=overall_sentiment)


def on_close():
    for widget in root.winfo_children():
        widget.destroy()
    root.quit()
    sys.exit()

# Initialize Tkinter
root = tk.Tk()
root.title("Umuzi Check-In Data Analysis")
root.geometry("1200x700")
root.configure(bg="#000000")

main_frame = tk.Frame(root,bg="#000000")
main_frame.pack(fill="both", expand=True)

root.protocol("WM_DELETE_WINDOW", on_close)






#================================================================================================#
# Main window layout                                                                             #
#================================================================================================#
# - This section focuses on the styling and Tkinter implementation.                              #
# - Sets up the main dashboard layout using Tkinter’s grid system.                   #
#       display_container holds the page title, main buttons, and counter section.               #
#       counter_container` shows summary metrics: top words, LDA trends, and overall sentiment.  #
#       buttons_container` hosts navigation buttons for Word Cloud and Frequency Analysis views. #
#================================================================================================#


main_frame.grid_columnconfigure(0, weight=1) 
main_frame.grid_columnconfigure(1, weight=0) 
main_frame.grid_rowconfigure(0, weight=1)
main_frame.grid_rowconfigure(1,weight=1)




display_container = tk.Frame(main_frame, bg="#000000")
display_container.grid(row=1, column=0, sticky="new", padx=10, pady=60)

counter_container = tk.Frame(display_container, bg="black", padx=10)
counter_container.grid(row=1, column=1, sticky="new")

display_container.grid_columnconfigure(0, weight=1) 
display_container.grid_columnconfigure(1, weight=2) 
display_container.grid_columnconfigure(2, weight=1)
display_container.grid_rowconfigure(0, weight=1)
display_container.grid_rowconfigure(1,weight=1)
display_container.grid_rowconfigure(2,weight=1)
display_container.grid_rowconfigure(3,weight=1)

tk.Label(display_container, text="Umuzi Check-In Data Analysis",
        font=("Lucida", 40, "bold"),bg="black",fg="#FE7F2D").grid(row=0, column=1, sticky="n",pady=20)


buttons_container= tk.Frame(display_container, bg="#000000")
buttons_container.grid(row=2, column=1, sticky="new",pady=30)
buttons_container.grid_columnconfigure((0,1),weight=1)


tk.Button(buttons_container, text="Word Cloud Clustering",font=("Lucida", 20,"bold"),bg="#ffffff",fg="#2D728F", command=plot_wordclouds).grid(row=0, column=0, sticky="nsew",padx=5)
tk.Button(buttons_container, text="Frequency Analysis",font=("Lucida", 20,"bold"),bg="#ffffff", fg="#2D728F",command=show_word_pairs).grid(row=0, column=1, sticky="nsew",padx=5 )



counter_container.grid_columnconfigure((0,1,2), weight=1)
counter_container.grid_rowconfigure(0, weight=1)
counter_container.grid_rowconfigure(1,weight=2)

# FREQUENT TERMS
tk.Label(counter_container, text="FREQUENT TERMS", font=("Lucida", 16, "bold"), bg="black", fg="#FE7F2D").grid(row=0, column=0)
lbl_top_words = tk.Label(counter_container, text="0", font=("Lucida", 50 ), bg="black", fg="#FFFFFF")
lbl_top_words.grid(row=1, column=0,)

# ENTRIES ANALYSED
tk.Label(counter_container, text="ENTRIES ANALYSED", font=("Lucida", 16, "bold"), bg="black", fg="#FE7F2D").grid(row=0, column=1)
lbl_entries = tk.Label(counter_container, text="0", font=("Lucida", 50), bg="black", fg="#ffffff")
lbl_entries.grid(row=1, column=1,)

# OVERALL SENTIMENT
tk.Label(counter_container, text="OVERALL SENTIMENT", font=("Lucida", 16, "bold"), bg="black", fg="#FE7F2D").grid(row=0, column=2)
lbl_sentiment = tk.Label(counter_container, text="Neutral", font=("Lucida", 50, ), bg="black", fg="#89F7A1")
lbl_sentiment.grid(row=1, column=2)




update_counters()
root.mainloop()