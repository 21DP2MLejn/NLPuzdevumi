import nltk
from collections import Counter
import string
from langdetect import detect
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from gensim.models import Word2Vec
import spacy
import random
from googletrans import Translator
import numpy as np
import stanza
import langid

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')
nltk.download('stopwords')
stanza.download('lv')
nlp_stanza = stanza.Pipeline('lv')

# 1. uzdevums: Vārdu biežuma analīze
print('1.uzd')
def word_frequency(text):
    words = nltk.word_tokenize(text.lower())
    words = [word.strip(string.punctuation) for word in words if word]
    frequency = Counter(words)
    sorted_frequency = sorted(frequency.items(), key=lambda x: x[1], reverse=True)[:10]
    return dict(sorted_frequency)

text = "Mākoņainā dienā kaķis sēdēja uz palodzes. Kaķis domāja, kāpēc debesis ir pelēkas. Kaķis gribēja redzēt sauli, bet saule slēpās aiz mākoņiem."
result = word_frequency(text)
print("Word Frequency Analysis:")
for word, count in result.items():
    print(f"{word}: {count}")

# 2. uzdevums: Teksta valodas noteikšana
print('2.uzd')
def identify_language(text):
    try:
        language, _ = langid.classify(text)
        return language
    except:
        return "Unable to determine language"

texts = [
    "Šodien ir saulaina diena.",
    "Today is a sunny day.",
    "Сегодня солнечный день."
]
for i, text in enumerate(texts):
    print(f"Text {i+1} language: {identify_language(text)}")


# 3. uzdevums: Vārdu sakritības procenti
print('3.uzd')
def word_coherence(text1, text2):
    words1 = set(nltk.word_tokenize(text1.lower()))
    words2 = set(nltk.word_tokenize(text2.lower()))
    common_words = words1.intersection(words2)
    total_words = len(words1.union(words2))
    coherence_percentage = (len(common_words) / total_words) * 100
    return coherence_percentage

text1 = "Rudens lapas ir dzeltenas un oranžas. Lapas klāj zemi un padara to krāsainu."
text2 = "Krāsainas rudens lapas krīt zemē. Lapas ir oranžas un dzeltenas."
result = word_coherence(text1, text2)
print(f"Vārdu sakritības līmenis: {result:.2f}%")

# 4. uzdevums: Noskaņojuma analīze
print('4.uzd')
def analyze_sentiment(text):
    translator = Translator()
    translated_text = translator.translate(text, src='lv', dest='en').text  # Translate text to English
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(translated_text)

    # Determine sentiment
    if sentiment_scores['compound'] >= 0.05:
        return "Positive"
    elif sentiment_scores['compound'] <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# Sample sentences
sentences = [
    "Šis produkts ir lielisks, esmu ļoti apmierināts!",
    "Esmu vīlies, produkts neatbilst aprakstam.",
    "Neitrāls produkts, nekas īpašs."
]

for sentence in sentences:
    print(f"{sentence}: {analyze_sentiment(sentence)}")

# 5. uzdevums: Teksta tīrīšana un normalizēšana
print('5.uzd')
def clean_and_normalize_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-ZāčēģīķļņšūžĀČĒĢĪĶĻŅŠŪŽ\s]', '', text)
    text = text.lower()
    words = text.split()
    return ' '.join(words)

text = "@John: Šis ir lielisks produkts!!! Vai ne? 👏👏👏 http://example.com"
clean_text = clean_and_normalize_text(text)
print(clean_text)

# 6. uzdevums: Automātiska rezumēšana
print('6.uzd')
def summarize_text(text, num_sentences=2):
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())
    words = [word for word in words if word not in string.punctuation]
    stop_words = set(stopwords.words('english') + ["un", "ir", "kas", "ar", "par", "tā", "ka", "uz", "bet", "tai", "vai", "kā", "kur", "šis", "tās", "to", "no", "tik", "bija", "tādēļ"])
    filtered_words = [word for word in words if word not in stop_words]
    word_frequencies = Counter(filtered_words)
    sentence_scores = {sentence: sum(word_frequencies.get(word, 0) for word in word_tokenize(sentence.lower())) for sentence in sentences}
    summarized_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
    return ' '.join(summarized_sentences)

text = ("Latvija ir valsts Baltijas reģionā. Tās galvaspilsēta ir Rīga, "
        "kas ir slavena ar savu vēsturisko centru un skaistajām ēkām. "
        "Latvija robežojas ar Lietuvu, Igauniju un Krieviju, kā arī tai ir "
        "piekļuve Baltijas jūrai. Tā ir viena no Eiropas Savienības dalībvalstīm.")
summarized_text = summarize_text(text, num_sentences=2)
print("Summarized text:")
print(summarized_text)

# 7. uzdevums: Vārdu iegulšana (word embeddings)
print('7.uzd')
def generate_word_embeddings(words, vector_size=100):
    model = Word2Vec([words], vector_size=vector_size, window=5, min_count=1)
    return model.wv

words = ["māja", "dzīvoklis", "jūra"]
model = generate_word_embeddings(words)
print("Vārdu vektori:")
for word in words:
    print(f"{word}: {model[word]}")

def compare_similarity(model, word1, word2):
    return model.similarity(word1, word2)

print("\nSemantiskā līdzība:")
for i in range(len(words)):
    for j in range(i + 1, len(words)):
        similarity = compare_similarity(model, words[i], words[j])
        print(f"Līdzība starp '{words[i]}' un '{words[j]}': {similarity:.4f}")

# 8. uzdevums: Frāžu atpazīšana (NER)
print('8.uzd')
def perform_ner_with_stanza(text):
    doc = nlp_stanza(text)
    entities = [(ent.text, ent.type) for ent in doc.ents]
    return entities

text = "Valsts prezidents Egils Levits piedalījās pasākumā, ko organizēja Latvijas Universitāte."
result = perform_ner_with_stanza(text)
print("Īpašas vienības tekstā:")
for entity, label in result:
    print(f"{entity}: {label}")

# 9. uzdevums: Teksta ģenerēšana
print('9.uzd')
def generate_text(seed_sentence, num_sentences=3):
    sentences = [
        "Reiz kādā tālā zemē...",
        "Viņš bija vecs vīrs ar gariem matiem...",
        "Šajā zemē valdīja miers un klusums...",
        "Pēkšņi debesīs parādījās spožs gaismas stars...",
        "Tas bija tikai sākums lielam piedzīvojumam..."
    ]
    generated_text = seed_sentence + " "
    for _ in range(num_sentences):
        generated_text += random.choice(sentences) + " "
    return generated_text.strip()

seed_sentence = "Reiz kādā tālā zemē..."
generated_story = generate_text(seed_sentence)
print(generated_story)

# 10. uzdevums: Tulkotāja izveide
print('10.uzd')
translator = Translator()

def translate_text(text, target_language='en'):
    translation = translator.translate(text, src='lv', dest=target_language)
    return translation.text

texts = [
    "Labdien! Kā jums klājas?",
    "Es šodien lasīju interesantu grāmatu."
]

for i, text in enumerate(texts):
    translated_text = translate_text(text)
    print(f"Latviesu valoda: {text}")
    print(f"Anglu valoda: {translated_text}")
    print()
