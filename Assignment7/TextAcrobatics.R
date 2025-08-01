# TextAcrobatics.R

# Install and load required packages
if (!require("pacman")) install.packages("pacman")
pacman::p_load(pacman, tm, SnowballC, dplyr, magrittr, ggplot2, wordcloud, tidytext, syuzhet)

# Read text files
schatzinsellines <- readLines('pg49424.txt')
zarathustralines <- readLines('thus-spoke-zarathustra-data.txt')


# Define old and custom stopwords and goosefeet removal
german_stop <- stopwords("german")
old_stopwords <- c("dass", "diess", "’s", "wer", "—", "the", "konnt", "ganz", "mehr", "sagt","gutenberg™","project")
combined_stop <- unique(c(german_stop, old_stopwords))
# Precompute stems of combined stopwords to remove after stemming
stemmed_stop <- wordStem(combined_stop, language = "german")

remove_goosefeet <- content_transformer(function(x) {
  gsub("[\u201E\u201A]", "", x)
})

# Custom cleaning function
clean_corpus <- function(corpus) {
  corpus <- tm_map(corpus, remove_goosefeet)
  corpus <- tm_map(corpus, content_transformer(tolower))
  corpus <- tm_map(corpus, removePunctuation)
  corpus <- tm_map(corpus, removeNumbers)
  # remove raw stopwords
  corpus <- tm_map(corpus, removeWords, combined_stop)
  corpus <- tm_map(corpus, stripWhitespace)
  # stem
  corpus <- tm_map(corpus, stemDocument)
  # remove stemmed stopwords
  corpus <- tm_map(corpus, removeWords, stemmed_stop)
  corpus <- tm_map(corpus, stripWhitespace)
  return(corpus)
}

# Create and clean corpora
corpus1 <- Corpus(VectorSource(schatzinsellines)) %>% clean_corpus()
corpus2 <- Corpus(VectorSource(zarathustralines)) %>% clean_corpus()

# Build document-term matrices and remove sparse terms
tdm1 <- DocumentTermMatrix(corpus1)
tdm2 <- DocumentTermMatrix(corpus2)

# Compute word frequencies
word.freqschatz <- sort(colSums(as.matrix(tdm1)), decreasing = TRUE)
word.freqzara   <- sort(colSums(as.matrix(tdm2)), decreasing = TRUE)

# Create frequency tables
tableschatzinsel <- data.frame(
  word = names(word.freqschatz),
  absolute.frequency = word.freqschatz,
  relative.frequency = word.freqschatz / sum(word.freqschatz)
)

tablezarathustra <- data.frame(
  word = names(word.freqzara),
  absolute.frequency = word.freqzara,
  relative.frequency = word.freqzara / sum(word.freqzara)
)

# Display top 15 words
print(head(tableschatzinsel, 15))
print(head(tablezarathustra, 15))

# Export top 1000 words to CSV
write.csv(tableschatzinsel[1:1000, ], "schatz_1000.csv", row.names = FALSE)
write.csv(tablezarathustra[1:1000, ], "zarathustra_1000.csv", row.names = FALSE)

# --- Visualizations ---
# Prepare data frames for plotting
df1 <- data.frame(term = names(word.freqschatz), freq = as.integer(word.freqschatz), stringsAsFactors = FALSE)
df2 <- data.frame(term = names(word.freqzara),   freq = as.integer(word.freqzara),   stringsAsFactors = FALSE)

# Histogram of term frequencies (top 10) for Schatzinsel
top_n <- 10
ggplot(df1[1:top_n, ], aes(x = reorder(term, -freq), y = freq)) +
  geom_bar(stat = "identity") +
  labs(
    title = "Häufigkeits-Histogramm der Top-Terme (Schatzinsel)",
    x = "Term", y = "Absolute Häufigkeit"
  ) +
  theme_minimal()

# Word Cloud for Schatzinsel
set.seed(123)
wordcloud(words = df1$term, freq = df1$freq, min.freq = 1, max.words = 100, random.order = FALSE, rot.per = 0.1, scale = c(4, 0.5))

# Histogram of term frequencies (top 10) for Zarathustra
top_n <- 10
ggplot(df2[1:top_n, ], aes(x = reorder(term, -freq), y = freq)) +
  geom_bar(stat = "identity") +
  labs(
    title = "Häufigkeits-Histogramm der Top-Terme (Zarathustra)",
    x = "Term", y = "Absolute Häufigkeit"
  ) +
  theme_minimal()

# Word Cloud for Schatzinsel
set.seed(123)
wordcloud(words = df2$term, freq = df2$freq, min.freq = 1, max.words = 100, random.order = FALSE, rot.per = 0.1, scale = c(4, 0.5))

# --- Sentiment Analysis ---

# 1) Sentiment per line using 'syuzhet' (German)
sent_schatz <- get_sentiment(schatzinsellines, method = "bing", language = "german")
sent_zara   <- get_sentiment(zarathustralines, method = "bing", language = "german")

# Summary statistics
cat("Schatzinsel - Sentiment summary:\n")
print(summary(sent_schatz))
cat("\nZarathustra - Sentiment summary:\n")
print(summary(sent_zara))

# Plot sentiment distribution
plot(sent_schatz, type = "h", main = "Sentiment-Verlauf: Schatzinsel", ylab = "Sentiment Score", xlab = "Zeilennummer")
plot(sent_zara,   type = "h", main = "Sentiment-Verlauf: Zarathustra", ylab = "Sentiment Score", xlab = "Zeilennummer")

# 2) Token-level sentiment using 'tidytext' and Bing lexicon
tidy_tdm1 <- tidy(tdm1)
tidy_tdm2 <- tidy(tdm2)

bing <- get_sentiments("bing")

sent_by_term1 <- tidy_tdm1 %>%
  inner_join(bing, by = c(term = "word")) %>%
  count(sentiment, sort = TRUE)

sent_by_term2 <- tidy_tdm2 %>%
  inner_join(bing, by = c(term = "word")) %>%
  count(sentiment, sort = TRUE)

# Display sentiment counts
echo("Schatzinsel sentiment counts:")
print(sent_by_term1)
echo("Zarathustra sentiment counts:")
print(sent_by_term2)

# Bar chart of positive vs. negative terms for each text
par(mfrow = c(1,2))
barplot(sent_by_term1$n, names.arg = sent_by_term1$sentiment, main = "Schatzinsel: Bing Sentiment", ylab = "Anzahl Terme")
barplot(sent_by_term2$n, names.arg = sent_by_term2$sentiment, main = "Zarathustra: Bing Sentiment", ylab = "Anzahl Terme")

