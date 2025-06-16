import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter

# --- Installationen und Downloads ---
# Führen Sie diese Befehle bei Bedarf in Ihrem Terminal aus:
# pip install nltk matplotlib wordcloud spacy textblob-de
# python -m spacy download de_core_news_sm
#
# NLTK-Download für Stopwörter (nur einmalig nötig)
try:
    stopwords.words('german')
except LookupError:
    nltk.download('stopwords')

# Helper, um Dateipfade relativ zum Skript zu finden
# HINWEIS: Da __file__ in manchen Umgebungen nicht existiert, wird ein fester Pfad angenommen.
# Passen Sie BASE_DIR bei Bedarf an Ihr Arbeitsverzeichnis an.
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else '.'

def load_text(filename):
    """Lädt eine Textdatei aus dem Basisverzeichnis."""
    path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Datei nicht gefunden: {path}. Stellen Sie sicher, dass die Textdateien im selben Ordner wie das Skript liegen.")
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

# 1. Texte laden
try:
    t1 = load_text('pg49424.txt')              # "Die Schatzinsel"
    t2 = load_text('thus-spoke-zarathustra-data.txt')  # "Also sprach Zarathustra"
except FileNotFoundError as e:
    print(e)
    exit() # Beendet das Skript, wenn Dateien nicht gefunden werden

# 2. Vorverarbeitung
def preprocess(text):
    """Bereinigt und tokenisiert den Text."""
    text = re.sub(r'<.*?>', ' ', text)       # Entfernt HTML-Tags
    text = re.sub(r'[^\w\s]', ' ', text)    # Entfernt Satzzeichen
    text = re.sub(r'\d+', ' ', text)         # Entfernt Zahlen
    text = text.lower()                      # Konvertiert zu Kleinbuchstaben
    tokens = text.split()
    german_stops = set(stopwords.words('german'))
    return [t for t in tokens if t.isalpha() and t not in german_stops]

print("Verarbeite Texte...")
toks1 = preprocess(t1)
toks2 = preprocess(t2)
print("Texte verarbeitet.")

# 3. Worthäufigkeitsverteilungen
fd1 = FreqDist(toks1)
fd2 = FreqDist(toks2)

# Vergleichende Einblicke
shared = set(fd1) & set(fd2)
unique1 = set(fd1) - set(fd2)
unique2 = set(fd2) - set(fd1)

# -------------------------------------------------------
# Shared Vocabulary und Unique Lexicons ausgeben
# -------------------------------------------------------
print(f"\n--- Shared Vocabulary ({len(shared)} Wörter) ---")
print(', '.join(sorted(shared)[:100]))  # limitiert auf die ersten 100 Wörter

print(f"\n--- Unique to Schatzinsel ({len(unique1)} Wörter) ---")
print(', '.join(sorted(unique1)[:100]))

print(f"\n--- Unique to Zarathustra ({len(unique2)} Wörter) ---")
print(', '.join(sorted(unique2)[:100]))

# Anzeige der Top-20 Begriffe
def show_top(fd, label, N=20):
    print(f"\nTop {N} Wörter in {label}:")
    for w, c in fd.most_common(N):
        print(f"{w}: {c}")
    print()

show_top(fd1, 'Schatzinsel')
show_top(fd2, 'Also sprach Zarathustra')

# Balkendiagramme für Frequenzen
for fd, title in [(fd1, 'Schatzinsel'), (fd2, 'Zarathustra')]:
    plt.figure(figsize=(10, 5))
    fd.plot(20, title=f'Top 20 in {title}')
    plt.show()

# WordClouds für visuellen Kontrast
def wc_plot(fd, title):
    wc = WordCloud(width=800, height=400, background_color='white', collocations=False)
    wc.generate_from_frequencies(fd)
    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()

wc_plot(fd1, 'WordCloud – Schatzinsel')
wc_plot(fd2, 'WordCloud – Zarathustra')

# Häufigkeitsvergleich für ausgewählte Begriffe
def face_off(words):
    print("\nHäufigkeitsvergleich:")
    for w in words:
        print(f"{w}: Schatzinsel={fd1[w]}, Zarathustra={fd2[w]}")
    
    x = range(len(words))
    w1_counts = [fd1[w] for w in words]
    w2_counts = [fd2[w] for w in words]
    
    plt.figure(figsize=(8, 4))
    plt.bar([i - 0.2 for i in x], w1_counts, width=0.4, label='Schatzinsel')
    plt.bar([i + 0.2 for i in x], w2_counts, width=0.4, label='Zarathustra')
    plt.xticks(x, words)
    plt.ylabel('Häufigkeit')
    plt.title('Frequency Face-Off')
    plt.legend()
    plt.show()

selected_words = ['leben', 'tod', 'welt', 'mensch', 'gott', 'see']
face_off(selected_words)


# ===================================================================
# 4. Erweiterte Untersuchung
# ===================================================================

# 4a. Sentiment Analysis mit TextBlobDE (zuverlässige Methode)
print("\n--- Erweiterte Analysen ---")

# 4b. POS Tagging & NER mit spaCy
try:
    import spacy
    try:
        # Lade das deutsche spaCy-Modell
        nlp = spacy.load('de_core_news_sm')
    except OSError:
        print("\nWARNUNG: SpaCy-Modell 'de_core_news_sm' nicht gefunden.")
        print("Bitte führen Sie im Terminal aus: python -m spacy download de_core_news_sm")
        nlp = None

    if nlp:
        print("\nFühre POS-Tagging und NER durch (kann einen Moment dauern)...")
        
        def pos_ner_analysis(text, label):
            # Wir analysieren auch hier nur einen Teil des Textes zur Performance-Steigerung
            doc = nlp(text[:200000]) 
            
            # Zähle die Part-of-Speech Tags
            pos_counts = doc.count_by(spacy.attrs.POS)
            pos_top = sorted([(nlp.vocab[pos].text, count) for pos, count in pos_counts.items()], key=lambda x: -x[1])[:5]
            
            # Zähle die Named Entities
            ents = [ent.label_ for ent in doc.ents]
            ent_counts = Counter(ents).most_common(5)

            print(f"\nAnalyse für: {label}")
            print(f"Top 5 POS-Tags: {pos_top}")
            print(f"Top 5 Entitäten: {ent_counts}")

        pos_ner_analysis(t1, 'Schatzinsel')
        pos_ner_analysis(t2, 'Also sprach Zarathustra')
    else:
        print("\nPOS-Tagging und NER übersprungen, da das spaCy-Modell fehlt.")
        
except ImportError:
    print("\nWARNUNG: spaCy ist nicht installiert. POS-Tagging und NER werden übersprungen.")
    print("Bitte führen Sie aus: pip install spacy")