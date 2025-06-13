text = "thus-spoke-zarathustra-data.txt";
textdata = fileread(text);
% In tokenizedDocument umwandeln
documents = tokenizedDocument(textdata);

% Kleinschreibung
documents = lower(documents);

% Interpunktion entfernen
documents = erasePunctuation(documents);

% Stoppwörter entfernen (Standard)
documents = removeStopWords(documents);

% Optional: Eigene Stoppwörter hinzufügen
eigeneStopWoerter = ["TM", "oh", "will", "selber", "diess", "ach","project", "gutenberg", "the", "of", "or", "you", "gieng", "zarathustra", "sprach", "wahrlich", "immer", "schon", "ihnen", "einst", "eurer", "fand", "sei", "kam", "sah", "hier", "dinge", "giebt", "to", "jetzt", "a", "1", "wird" ];
documents = removeWords(documents, eigeneStopWoerter);

wordcloud(documents);