# BGSE Text Mining Assignments

Jupyter notebooks from the Text Mining course at the Barcelona School of Economics (BGSE): three assignments, from building TF-IDF by hand to word embeddings, and a final project on detecting troll tweets about Catalan independence.

| Notebook | Topic | Main techniques | Headline result |
|---|---|---|---|
| [`Working with Text Assignment 1.ipynb`](Working%20with%20Text%20Assignment%201.ipynb) | Document vectorisation from scratch | Tokenisation, stemming, stop words, TF / TF-IDF in pure NumPy, Euclidean & cosine distance | Retrieval score 0.92 → **1.22** with preprocessing + smoothed TF-IDF |
| [`Classifying Text Assignment 2.ipynb`](Classifying%20Text%20Assignment%202.ipynb) | Cross-domain sentiment classification (Yelp → IMDB) | Lemmatisation, `TfidfVectorizer`, logistic regression, `Pipeline` + `GridSearchCV` | ~0.97 in-domain vs **0.80** on the unseen domain |
| [`Word2Vec Assignment 3.ipynb`](Word2Vec%20Assignment%203.ipynb) | Small labelled set + large unlabelled set | TF-IDF vs averaged Word2Vec (gensim) features, logistic regression, KNN imputation | TF-IDF: 0.87 → 0.90 as labelled data grows from 1% to 4% |
| [`Text_Mining_Project.ipynb`](Text_Mining_Project.ipynb) | **Final project:** troll detection on Twitter | Multilingual tweet cleaning, feature engineering, EDA, language detection, TF-IDF, SMOTE, target encoding, logistic regression | Account + tweet-statistics model: **99.3% recall, 74% precision** on trolls |

---

## 1. Working with Text: TF-IDF from scratch

**Task:** implement something like scikit-learn's `TfidfVectorizer` using only Python and NumPy, then check whether a short query ("People who see ghosts") is closest to the right target document among several distractors.

**What was done**
- Text preprocessing with `re`: punctuation removal, whitespace normalisation, case folding, Snowball stemming (NLTK) and stop-word removal.
- Hand-written vocabulary, term-frequency, document-frequency and IDF functions (standard, smoothed and probabilistic IDF), with sublinear TF (`log(1 + tf)`) and L2 row normalisation.
- Euclidean and cosine distance matrices, shown as seaborn heatmaps.
- Comparison of IDF variants, and of substring-based vs whole-token document frequency.

**Result:** raw term frequency gives a score of 0.92 (a distractor is closer than the target). With preprocessing and smoothed TF-IDF the score is 1.22, so the target ranks first.

**Libraries:** NumPy, NLTK (`SnowballStemmer`), seaborn.

## 2. Classifying Text: cross-domain sentiment

**Task:** train a sentiment classifier on labelled Yelp restaurant reviews and test it on IMDB movie reviews, where no labels are available. Then try to improve the transfer using the *unlabelled* IMDB text.

**What was done**
- Label cleaning and class-balance checks (pandas, seaborn).
- Regex cleaning and WordNet lemmatisation (NLTK).
- TF-IDF + logistic regression trained on Yelp and evaluated on IMDB.
- Two ways to use the unlabelled target text: re-estimating IDF on IMDB, and fitting the vocabulary on both corpora.
- Hyperparameter search over vectoriser and classifier settings with `Pipeline` and `GridSearchCV`.

**Result:** 0.97 accuracy on Yelp (in-sample) vs 0.80 on IMDB. Sharing the vocabulary did not close the gap. The notebook ends with a short discussion of why sentiment vocabulary is partly domain-specific.

**Libraries:** pandas, NLTK, scikit-learn, seaborn, matplotlib.

## 3. Word2Vec: using unlabelled data

**Task:** simulate a small labelled set and a large unlabelled set (Yelp + IMDB, ~150k reviews), and compare two document representations as the labelled share grows from 1% to 2% to 4%.

**What was done**
- TF-IDF + logistic regression, tuned by grid search on each labelled subset.
- A gensim `Word2Vec` model, with each document represented as the average of its word vectors, a `KNNImputer` for documents with no known words, and a logistic regression classifier.
- An **extension** (added later, not run) that trains Word2Vec on *all* reviews and compares CBOW vs skip-gram, window sizes and negative sampling.

**Result:** TF-IDF goes from 0.87 to 0.89 to 0.90 test accuracy. Word2Vec trained only on the 1% labelled subset reaches 0.67, which shows why the embeddings need the unlabelled data.

**Libraries:** pandas, NumPy, NLTK, scikit-learn, gensim.

## 4. Final project: detecting troll tweets about Catalan independence

**Question:** can tweets from accounts that Twitter suspended for platform manipulation (its Information Operations archive, `catalonia_201906_1`) be told apart from ordinary Spanish-language tweets on the same political topics?

**What was done**
- **Cleaning** of Spanish, Catalan and English tweets: mentions, links, hashtags, emojis, Twitter abbreviations, symbols and garbage strings removed; stop words from `stopwordsiso` plus custom lists.
- **Feature engineering:**
  - *tweet level:* length, counts of mentions / hashtags / abbreviations / stop words, ratio of capital letters, retweet flag;
  - *account level:* digits in the display name, follower/following ratio, account age, client app, posting time.
- **Exploratory analysis:** token frequency plots and K-means + t-SNE clusters (yellowbrick), time between tweets, and troll vs control distributions of every feature.
- **Control group:** the random sample filtered by political keywords taken from the troll vocabulary.
- **Models:**
  - *text only:* `langdetect`, Spanish stemming, TF-IDF, SMOTE, logistic regression;
  - *metadata:* target-encoded categoricals, standard scaling, SMOTE, logistic regression.

**Findings:** trolls tweet during office hours (09:00–15:00), mostly through TweetDeck and the web client, from accounts created in batches whose display names usually contain digits. Their activity peaked around the first anniversary of the 1 October referendum. The gaps between their tweets vary widely, so they do not behave like simple scheduled bots.

| Model | Accuracy | Majority baseline | Troll precision | Troll recall |
|---|---|---|---|---|
| Text only (TF-IDF + SMOTE + LR) | 0.975 | 0.997 | 0.07 | 0.59 |
| Account + tweet statistics (SMOTE + LR) | **0.991** | 0.976 | **0.74** | **0.99** |

Behavioural and account features identify trolls far better than the wording of their tweets. The notebook also lists known issues in the saved results and next steps.

**Libraries:** pandas, NumPy, NLTK, stopwordsiso, langdetect, scikit-learn, imbalanced-learn, category_encoders, yellowbrick, matplotlib, seaborn.

---

## Tools and libraries

| Library | Used for | Notebooks |
|---|---|---|
| **NumPy** | Vector maths, hand-written TF-IDF, distances | 1, 3, 4 |
| **pandas** | Loading, cleaning and aggregating data | 2, 3, 4 |
| **re** (standard library) | Text cleaning: punctuation, links, mentions, hashtags, emojis | all |
| **NLTK** | Snowball stemmer (EN/ES), WordNet lemmatiser, `word_tokenize` | all |
| **scikit-learn** | `CountVectorizer`, `TfidfVectorizer`, `LogisticRegression`, `Pipeline`, `GridSearchCV`, `KMeans`, `KNNImputer`, `StandardScaler`, metrics | 2, 3, 4 |
| **gensim** | `Word2Vec` word embeddings | 3 |
| **imbalanced-learn** | SMOTE oversampling for the imbalanced troll data | 4 |
| **category_encoders** | `TargetEncoder` for categorical features | 4 |
| **stopwordsiso** | Spanish, Catalan and English stop-word lists | 4 |
| **langdetect** | Tweet language detection | 4 |
| **yellowbrick** | Token frequency plots, t-SNE cluster plots | 4 |
| **matplotlib / seaborn** | Heatmaps, distributions, confusion matrices | all |
| **Google Colab** | Original execution environment (data stored on Google Drive) | 2, 3, 4 |

## Data

The datasets are **not included** in this repository. Assignment 1 is fully self-contained: its documents are defined in the notebook. The others expect:

| Notebook | Files |
|---|---|
| 2, 3 | `yelps.csv`, `movies.csv` (columns `text`, `positive`) next to the notebook |
| 4 | `Text Mining Project/DATA/catalonia_201906_1_tweets_csv_hashed.csv` (Twitter Information Operations archive) and `Text Mining Project/DATA/es_tweets_2017_2019_16k_v2.csv` (random Spanish tweets) |

## Running the notebooks

```bash
pip install -r requirements.txt
jupyter notebook
```

On Google Colab, the notebooks mount Google Drive and `chdir` into `My Drive/Colab Notebooks`; change that path to wherever you keep the data. Locally, the Colab step is skipped automatically. NLTK resources (stop words, punkt, WordNet) are downloaded by the notebooks themselves.

## About the saved outputs

All outputs were produced in the original Google Colab run (2021) and are kept as-is. Since the data is not available, the code was later tidied only in ways that keep those outputs valid:

- imports deduplicated, dead code and unused helpers removed, comments and explanations added;
- APIs updated to current library versions (gensim 4, scikit-learn `get_feature_names_out`, seaborn `histplot`);
- Colab-specific setup made optional.

New code that has **not** been run is clearly marked: the Word2Vec extension in Assignment 3 and the cross-validated text pipeline in the final project. Assignment 1's data is inline, so its code was re-run and checked: it reproduces the saved scores exactly. Known issues in the project's saved results, such as an abbreviation filter that also cut letters out of the middle of words, are documented at the top of that notebook instead of being silently changed.
