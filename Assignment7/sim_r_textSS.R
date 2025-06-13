# sim_r_textSS.R

# Install and download packages
pacman::p_load(pacman, tm, SnowballC, dplyr, wordcloud, RColorBrewer)

# Import data
# Don't need to specify file path if files are in the same directory or folder as R script. 
# The text files must be without metadata.
# Set the working directory where you saved your files.
# setwd("D:/ddoc/DLEARN/Simulation-KL/PSs/PS6")

# "Sense and Sensibility" by Jane Austen, published 1811
bookSS <- readLines('SenseSensibility.txt')

###################################################################################
# Corpus for Sense and Sensibility
corpusSS <- Corpus(VectorSource(bookSS)) %>%
  tm_map(removePunctuation) %>%
  tm_map(removeNumbers) %>%
  tm_map(content_transformer(tolower)) %>%
  tm_map(removeWords, stopwords("english")) %>%
  tm_map(stripWhitespace) %>%
  tm_map(stemDocument)

# Create term-document matrices and remove sparse terms
tdmSS <- DocumentTermMatrix(corpusSS) %>%
  removeSparseTerms(1 -(5/length(corpusSS)))

# Calculate and sort by word frequencies
word.freqSS <- sort(colSums(as.matrix(tdmSS)), decreasing = T)


# Create frequency table
tableSS <- data.frame(word = names(word.freqSS),
                      absolute.frequency = word.freqSS,
                      relative.frequency = word.freqSS/length(word.freqSS))

# Remove the words from the row names
row.names(tableSS) <- NULL

#Show the 15 most common words
head(tableSS, 15)

# Export the 1000 most common words in csv file
write.csv(tableSS[1:1000, ], "SS_1000.csv")

################################################################################
set.seed(42)
# Limit words by specifying min frequency
wordcloud(names(word.freqSS),word.freqSS, min.freq=70)

# Add color
wordcloud(names(word.freqSS),word.freqSS,min.freq=100,colors=brewer.pal(8,"Dark2"))

################################################################################

# Clear workspace
rm(list = ls())






