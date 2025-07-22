"""
analysis.py

Tool zur Analyse eines Earnings-Call-Transkripts:
1. Sentiment-Analyse der CEO-Eröffnung
2. Häufigkeitsanalyse business-relevanter Nomen
3. Extraktion wichtiger Finanzkennzahlen (inkl. absolute Zahlen)
4. Template-basierte Executive Summary (strukturierte NLG)
"""

import re
import spacy
from collections import Counter
from nltk.corpus import stopwords
from nltk import download
from transformers import pipeline

# Modelle & Ressourcen laden
nlp = spacy.load("en_core_web_sm")
sentiment_model = pipeline("sentiment-analysis", device=0)

# NLTK-Ressourcen herunterladen
download('punkt')
download('stopwords')

# Domänenspezifische Stop-Nomen
DOMAIN_STOP = set(['year','model','question','time','percent','quarter','company','thank','call'])


def read_transcript(path):
    with open(path, encoding='utf-8') as f:
        return f.read()


def analyze_sentiment(text):
    opening = "\n".join(text.splitlines()[:5])
    res = sentiment_model(opening)[0]
    sentiment = {'NEGATIVE': 'Negative', 'POSITIVE': 'Positive'}.get(res['label'], 'Neutral')
    return sentiment, res['score']


def top_nouns(text, n=5):
    doc = nlp(text)
    candidates = []
    for tok in doc:
        if tok.pos_ == 'NOUN' and tok.is_alpha:
            lemma = tok.lemma_.lower()
            if lemma not in stopwords.words('english') and lemma not in DOMAIN_STOP:
                candidates.append(lemma)
    most = Counter(candidates).most_common(n)
    return [word for word, _ in most]


def extract_financials(text):
    """Extrahiert Finanzkennzahlen aus dem Transkript."""
    data = {}
    # Revenue absolute, sequential & YoY
    m = re.search(r"Revenue of \$(\d+\.?\d*) billion was up (\d+)% sequentially and up (\d+)% year on year", text, re.IGNORECASE)
    if m:
        data['Current revenue'] = f"${m.group(1)}B"
        data['Sequential growth'] = f"{m.group(2)}%"
        data['YoY growth'] = f"{m.group(3)}%"
    # Full fiscal year revenue
    fy = re.search(r"For fiscal (\d+) revenue was \$(\d+\.?\d*) billion, up (\d+)%", text, re.IGNORECASE)
    if fy:
        data['FY revenue'] = f"${fy.group(2)}B ({fy.group(3)}% YoY)"
    # Operating margin expansion
    opm = re.search(r"operating margin.*?(\d+) basis points", text, re.IGNORECASE)
    if opm:
        data['Operating margin expansion'] = f"{opm.group(1)} bps"
    # Forecast
    fc = re.search(r"expected (?:in|for) (Q\d)[^\.]*", text, re.IGNORECASE)
    if fc:
        pct = re.search(r"([Mm]id[- ]teens)%? .* expected", text)
        if pct:
            data['Forecast'] = f"{pct.group(1).capitalize()} growth in {fc.group(1)}"
    return data


def generate_structured_summary(sentiment, topics, financials):
    """Erstellt eine prägnante Zusammenfassung basierend auf Sentiment, Top-Themen und Finanzdaten."""
    topic_str = ', '.join(topics[:3])
    fin_parts = [f"{k}: {v}" for k, v in financials.items()]
    fin_str = '; '.join(fin_parts) if fin_parts else 'keine Finanzdaten verfügbar'
    return (
        f"Sentiment der CEO-Eröffnung: {sentiment}. "
        f"Hauptthemen: {topic_str}. "
        f"Finanzdaten: {fin_str}. "
        "Insgesamt deutet alles auf starkes Wachstumspotenzial hin."
    )


def main():
    transcript = read_transcript('earnings_call_transcript.txt')

    # 1. Sentiment
    sentiment, conf = analyze_sentiment(transcript)
    print(f"Sentiment Analysis: {sentiment} (Confidence: {conf:.2f})\n")

    # 2. Key Topics
    topics = top_nouns(transcript)
    print("Key Topics:")
    for i, t in enumerate(topics, 1):
        print(f"  {i}. {t}")
    print()

    # 3. Finanzdaten
    financials = extract_financials(transcript)
    print("Key Financial Data Points:")
    if financials:
        for k, v in financials.items():
            print(f"  - {k}: {v}")
    else:
        print("  Keine Finanzdaten gefunden.")
    print()

    # 4. Executive Summary
    summary = generate_structured_summary(sentiment, topics, financials)
    print("Generated Executive Summary:")
    print(summary)


if __name__ == '__main__':
    main()