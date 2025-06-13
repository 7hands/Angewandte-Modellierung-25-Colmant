# sim_r_text.R

# Install and download packages

pacman::p_load(pacman, tm, SnowballC, dplyr)

# Import data
# Don't need to specify file path if files are in the same directory or folder as R script. 
# The text files must be without metadata.

# "Jane Eyre" by Charlotte Bronte, published 1847
bookJE <- readLines('JaneEyre.txt')

# "Wuthering Heights" by Emily Bronte, published 1847
bookWH <- readLines('WutheringHeights.txt')

###################################################################################
# Corpus for Jane Eyre
corpusJE <- Corpus(VectorSource(bookJE)) %>%
  tm_map(removePunctuation) %>%
  tm_map(removeNumbers) %>%
  tm_map(content_transformer(tolower)) %>%
  tm_map(removeWords, stopwords("english")) %>%
  tm_map(stripWhitespace) %>%
  tm_map(stemDocument)

# Create term-document matrices and remove sparse terms
tdmJE <- DocumentTermMatrix(corpusJE) %>%
  removeSparseTerms(1 -(5/length(corpusJE)))

# Calculate and sort by word frequencies
word.freqJE <- sort(colSums(as.matrix(tdmJE)), decreasing = T)


# Create frequency table
tableJE <- data.frame(word = names(word.freqJE),
                      absolute.frequency = word.freqJE,
                      relative.frequency = word.freqJE/length(word.freqJE))
  
# Remove the words from the row names
row.names(tableJE) <- NULL

#Show the 15 most common words
head(tableJE, 15)

# Export the 1000 most common words in csv file
write.csv(tableJE[1:1000, ], "JE_1000.csv")

###################################################################################

# Corpus for Wuthering Heights
corpusWH <- Corpus(VectorSource(bookWH)) %>%
  tm_map(removePunctuation) %>%
  tm_map(removeNumbers) %>%
  tm_map(content_transformer(tolower)) %>%
  tm_map(removeWords, stopwords("english")) %>%
  tm_map(stripWhitespace) %>%
  tm_map(stemDocument)

# Create term-document matrices and remove sparse terms
tdmWH <- DocumentTermMatrix(corpusWH) %>%
  removeSparseTerms(1 -(5/length(corpusWH)))

# Calculate and sort by word frequencies
word.freqWH <- sort(colSums(as.matrix(tdmWH)), decreasing = T)


# Create frequency table
tableWH <- data.frame(word = names(word.freqWH),
                      absolute.frequency = word.freqWH,
                      relative.frequency = word.freqWH/length(word.freqWH))

# Remove the words from the row names
row.names(tableWH) <- NULL

#Show the 15 most common words
head(tableWH, 15)

# Export the 1000 most common words in csv file
write.csv(tableWH[1:1000, ], "WH_1000.csv")

###############################################################################

# Most distinctive words

# Set number of digits for output
options(digits = 2)

# Compare relative frequencies (subtractions)
je_wh <- tableJE %>% merge(tableWH, by = "word") %>%
  mutate(dProp = relative.frequency.x - relative.frequency.y, dAbs = abs(dProp)) %>%
  arrange(desc(dAbs)) %>%
  rename(JE.freq = absolute.frequency.x, JE.prop = relative.frequency.x,
         WH.freq = absolute.frequency.y, WH.prop = relative.frequency.y)

# Show the 15 most distinctive terms
head(je_wh, 15)

# Save the full table to csv
write.csv(je_wh, "diff_table.csv")
write.csv2(je_wh, "diff2_table.csv")

###############################################################################

# Clear workspace
rm(list = ls())










