import re
import json
import nltk
import pickle
from tqdm import tqdm
import multiprocessing
from nltk.corpus import stopwords
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

print('Loading cards...')
with open('data/AtomicCards.json', 'r', encoding='utf-8') as f:
    cards = json.load(f)['data']

try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

reminder_remover = re.compile('\(.*\)')
allowed_characters = set('abcdefghijklmnopqrstuvwxyzáéïíöú0123456789+-{}[]:/@"!?*•|$# ')
stop_words = set(stopwords.words('english'))

def is_valid(text):
    return len(text) > 0 and all(c in allowed_characters for c in text)

def get_text(card):
    text = ''

    if '//' in card['name'] and cards.get(card['name'].split(' // ')[0]):
        return text

    if 'manaCost' in card:
        text += card['manaCost'].replace('}{', '} {').replace('{', '|').replace('}', '|') + ' '
    if 'power' in card:
        text += '$' + card['power'] + '$ #' + card['toughness'] + '# '
    text += ' '.join(['|' + word + '|' for word in card['type'].split(' — ')[0].split()]) + ' '
    if '—' in card['type']:
        text += card['type'].split(' — ')[1] + ' '
    if 'text' in card:
        if 'Basic' in card['type'] and 'Land' in card['type']:
            text += card['text'].replace('(', '').replace(')', '')
        text += card['text'].replace('&', 'and').replace(card['name'], '@').replace(card['name'].split(',')[0], '@').replace('this creature', '@').replace('this enchantment', '@').replace('this artifact', '@').replace('this land', '@').replace('\n', ' ').replace(';', ' ').replace(':', ' :').replace('|', '•')
    
    text = re.sub(reminder_remover, '', text.lower().replace('−', '-').replace('—', '-').replace('-{', ' - {').replace('’', '\'').replace(',', '').replace('.', '').replace('\'', '').replace('}{', '} {').replace('@s', '@').replace('"', ' " ')).strip()

    words = [word for word in text.split(' ') if word != '']
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

class CardCorpus:
    def __iter__(self):
        card_file = ''

        for i, (card_name, info) in enumerate(cards.items()):
            info = info[0]
            if 'vintage' in info.get('legalities', {}) and (info['legalities']['vintage'] == 'Legal' or info['legalities']['vintage'] == 'Restricted'):
                text = get_text(info)
                if not is_valid(text):
                    continue

                card_file += text + '\n'

                yield TaggedDocument(text.split(), [i])
        
        file = open('data/cards.txt', 'w', encoding='utf-8')
        file.write(card_file[:-1])
        file.close()

corpus = CardCorpus()
cores = multiprocessing.cpu_count()

model = Doc2Vec(
    vector_size=128,
    dm=0,
    dbow_words=0,
    min_count=2,
    epochs=200,
    workers=cores,
    seed=42
)

print('Building vocabulary...')
model.build_vocab(corpus)

print('Training model...')
model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)
model.save('models/embedding_model')

print("Processing legal cards...")
legal_cards = {}
for card_name, info in tqdm(cards.items()):
    info = info[0]
    if 'vintage' in info.get('legalities', {}) and (info['legalities']['vintage'] == 'Legal' or info['legalities']['vintage'] == 'Restricted'):
        card_text = get_text(info)
        if len(card_text) > 0:
            legal_cards[card_name] = card_text

print("Creating embeddings map...")
embeddings_map = {}
for card_name, card_text in tqdm(legal_cards.items()):
    tokens = card_text.split(' ')
    
    try:
        vector = model.infer_vector(tokens)
        embeddings_map[card_name] = vector
    except Exception as e:
        print(f"Error embedding card '{card_name}': {e}")

print(f"Saving embeddings for {len(embeddings_map)} cards...")
with open('data/card_embeddings.pkl', 'wb') as f:
    pickle.dump(embeddings_map, f)