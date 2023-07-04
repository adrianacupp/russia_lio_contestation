import numpy
import os
import pickle
import csv
import math
from gensim.models import Word2Vec
from sklearn.utils import resample
import random

random.seed(3919)

df = pd.read_pickle('/Users/adrianacuppuleri/Desktop/GITHUB ADRIANA/Illiberal_discourse/data/corpus_adriana/corpus_president_of_russia/merged_df.pkl')

all_text = " ".join(df['text_clean'])

word_count = len(re.findall(r'\w+', all_text))
print("Total number of words:", word_count)

#adjust compund words that will be used later for the codebook
word_to_find = 'миропорядок'
mask = df['text_clean'].str.contains(word_to_find)

if mask.any():
    print(f"The word '{word_to_find}' is present in at least one row.")
    rows_with_word = df[mask]
    print(f"The word appears in {len(rows_with_word)} rows.")
else:
    print(f"The word '{word_to_find}' is not present in any row.")


df['text_clean'] = df['text_clean'].apply(lambda x: x.replace("мировой порядок", "миропорядок") 
                if any(phrase in x for phrase in ["мировой порядок", "мирового порядка", 
                                                  "мировому порядку", "мировым порядком",
                                                  "мировом порядке"]) else x)

word_to_find = 'миропорядок'
mask = df['text_clean'].str.contains(word_to_find)

if mask.any():
    print(f"The word '{word_to_find}' is present in at least one row.")
    rows_with_word = df[mask]
    print(f"The word appears in {len(rows_with_word)} rows.")
else:
    print(f"The word '{word_to_find}' is not present in any row.")


word_to_find = 'русский мир'
mask = df['text_clean'].str.contains(word_to_find)

if mask.any():
    print(f"The word '{word_to_find}' is present in at least one row.")
    rows_with_word = df[mask]
    print(f"The word appears in {len(rows_with_word)} rows.")
else:
    print(f"The word '{word_to_find}' is not present in any row.")
# transform all declension of русский мир in compound word with _
# nom: русский мир
# gen: русского мира
# dat: русскому миру
# acc: русский мир
# instr: русским миром
# prepos: русском мире

# if русский мир (in all its declension) is present then русский_мир
df['text_clean'] = df['text_clean'].apply(lambda x: x.replace("русский мир", "русский_мир"))
df['text_clean'] = df['text_clean'].apply(lambda x: x.replace("русского мира", "русский_мир"))
df['text_clean'] = df['text_clean'].apply(lambda x: x.replace("русскому миру", "русский_мир"))
df['text_clean'] = df['text_clean'].apply(lambda x: x.replace("русским миром", "русский_мир"))
df['text_clean'] = df['text_clean'].apply(lambda x: x.replace("русском мире", "русский_мир"))

word_to_find = 'русский_мир'
mask = df['text_clean'].str.contains(word_to_find)

if mask.any():
    print(f"The word '{word_to_find}' is present in at least one row.")
    rows_with_word = df[mask]
    print(f"The word appears in {len(rows_with_word)} rows.")
else:
    print(f"The word '{word_to_find}' is not present in any row.")

word_to_find = 'национальная безопасность'
mask = df['text_clean'].str.contains(word_to_find)

if mask.any():
    print(f"The word '{word_to_find}' is present in at least one row.")
    rows_with_word = df[mask]
    print(f"The word appears in {len(rows_with_word)} rows.")
else:
    print(f"The word '{word_to_find}' is not present in any row.")

# the same for национальная_безопасность
df['text_clean'] = df['text_clean'].apply(lambda x: x.replace("национальная безопасность", "национальная_безопасность"))
df['text_clean'] = df['text_clean'].apply(lambda x: x.replace("национальной безопасности", "национальная_безопасность"))
df['text_clean'] = df['text_clean'].apply(lambda x: x.replace("национальную безопасность", "национальная_безопасность"))
df['text_clean'] = df['text_clean'].apply(lambda x: x.replace("национальной безопасностью", "национальная_безопасность"))

word_to_find = 'национальная_безопасность'
mask = df['text_clean'].str.contains(word_to_find)

if mask.any():
    print(f"The word '{word_to_find}' is present in at least one row.")
    rows_with_word = df[mask]
    print(f"The word appears in {len(rows_with_word)} rows.")
else:
    print(f"The word '{word_to_find}' is not present in any row.")

# same for международное право
word_to_find = 'международное право'
mask = df['text_clean'].str.contains(word_to_find)

if mask.any():
    print(f"The word '{word_to_find}' is present in at least one row.")
    rows_with_word = df[mask]
    print(f"The word appears in {len(rows_with_word)} rows.")
else:
    print(f"The word '{word_to_find}' is not present in any row.")



df['text_clean'] = df['text_clean'].apply(lambda x: x.replace("международное право", "международное_право"))
df['text_clean'] = df['text_clean'].apply(lambda x: x.replace("международного права", "международное_право"))
df['text_clean'] = df['text_clean'].apply(lambda x: x.replace("международному праву", "международное_право"))
df['text_clean'] = df['text_clean'].apply(lambda x: x.replace("международным правом", "международное_право"))
df['text_clean'] = df['text_clean'].apply(lambda x: x.replace("международном праве", "международное_право"))


word_to_find = 'международное_право'
mask = df['text_clean'].str.contains(word_to_find)

if mask.any():
    print(f"The word '{word_to_find}' is present in at least one row.")
    rows_with_word = df[mask]
    print(f"The word appears in {len(rows_with_word)} rows.")
else:
    print(f"The word '{word_to_find}' is not present in any row.")


############ create the 4 dictionaries #######
eurasianism = ['еврази','евразийск','русский_мир', 'цивилизаци', 'запад', 'нацисм']

westphalianism = ['великодержавн','статус','суверенн','международное_право']
security_concerns = ['сфер', 'национальная_безопасность','противовес','дестабилиз','нато','угроз']

multipolarism = ['многополярн', 'партнерств', 'сотрудничеств','альтернатив', 'выбор', 'развити']

# change name in era
#transform each timeframe df in a list
era1 = t1['text_clean'].tolist()
era2 = t2['text_clean'].tolist()
era3 = t3['text_clean'].tolist()
era4 = t4['text_clean'].tolist()

list_of_lists = []
list_of_lists.append(era1)
list_of_lists.append(era2)
list_of_lists.append(era3)
list_of_lists.append(era4)

# bootstrapping see rodman
# model: sg
# vector_size: 300
# n_neighbours: 30


n_bootstraps = 50
eurasia_similarity = []
westph_similarity = []
sc_similarity = []
mp_similarity = []

########### this code takes too long
#### Finished with run 1 out of 50 for era 1 in ONE HOUR!

for j in range(0, len(list_of_lists)):
    sim_stats_eurasia = []
    sim_stats_westph = []
    sim_stats_sc = []
    sim_stats_mp = []
    for k in range(n_bootstraps):
        sentence_samples = resample(list_of_lists[j])
        model = Word2Vec(sentence_samples, vector_size=100, min_count=0, epochs= 200, 
                         sg=1, hs=0, negative=5, window=15, workers=4)
        for word, sim_stats in zip(eurasianism, sim_stats_eurasia):
            try:
                sim_stats.append(model.wv.most_similar('миропорядок', word)[0][1])
            except KeyError:
                sim_stats.append('NA')
        for word, sim_stats in zip(westphalianism, sim_stats_westph):
            try:
                sim_stats.append(model.wv.most_similar('миропорядок', word)[0][1])
            except KeyError:
                sim_stats.append('NA')
        for word, sim_stats in zip(security_concerns, sim_stats_sc):
            try:
                sim_stats.append(model.wv.most_similar('миропорядок', word)[0][1])
            except KeyError:
                sim_stats.append('NA')
        for word, sim_stats in zip(multipolarism, sim_stats_mp):
            try:
                sim_stats.append(model.wv.most_similar('миропорядок', word)[0][1])
            except KeyError:
                sim_stats.append('NA')
        run = k+1
        era = j+1
        print("Finished with run %d out of %d for era %d." % (run, n_bootstraps, era))
    print("*******Finished with era %d.*******" % (era))
    eurasia_similarity.append(sim_stats_eurasia)
    westph_similarity.append(sim_stats_westph)
    sc_similarity.append(sim_stats_sc)
    
stat_types = [eurasia_similarity,westph_similarity,
    sc_similarity, mp_similarity]




############### break the code ############

# Define function for bootstrapping
# used to generate new samples for training the Word2Vec model

def bootstrap_eras(list_of_lists, n_bootstraps=50, vector_size=100, min_count=0, epochs=200, sg=1, hs=0, negative=5, window=15, workers=4):
    random.seed(3919)
    bootstrapped_eras = []
    for era_idx, era in enumerate(list_of_lists):
        sim_stats_era = []
        for i in range(n_bootstraps):
            sentence_samples = resample(era)
            model = Word2Vec(sentence_samples, vector_size=vector_size, min_count=min_count, epochs=epochs, sg=sg, hs=hs, negative=negative, window=window, workers=workers)
            sim_stats_era.append(model)
            # Save each bootstrapped model as a .pkl file
            filename = f"models/bootstrap/era{era_idx+1}_bootstrap{i+1}.pkl"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "wb") as f:
                pickle.dump(model, f)
            print(f"Finished with run {i+1} out of {n_bootstraps} for era {era_idx+1}")
        bootstrapped_eras.append(sim_stats_era)
    return bootstrapped_eras

# Define function for similarity computation
# used to compute the similarity between words using a pre-trained Word2Vec model
def compute_similarity(sentence_samples, target_word):
    # Train word2vec model on sentence samples
    model = Word2Vec(sentence_samples, vector_size=100, min_count=0, epochs= 200, sg=1, hs=0, negative=5, window=15, workers=4)
    
    # Bootstrap model n times and compute similarity between target word and related words for each bootstrap iteration
    n_bootstraps = 50
    sim_stats_eurasia = []
    sim_stats_westph = []
    sim_stats_sc = []
    sim_stats_mp = []
    for i in range(n_bootstraps):
        sentence_samples_boot = resample(sentence_samples)
        sim_stats_eurasia.append(bootstrap(model, target_word, eurasianism))
        sim_stats_westph.append(bootstrap(model, target_word, westphalianism))
        sim_stats_sc.append(bootstrap(model, target_word, security_concerns))
        sim_stats_mp.append(bootstrap(model, target_word, multipolarism))
    return sim_stats_eurasia, sim_stats_westph, sim_stats_sc, sim_stats_mp

# Loop over eras and compute similarity for each era
era_similarity = []
eras = [df[df['era'] == i]['text'].tolist() for i in range(1,5)]
for i, era in enumerate(eras):
    sim_stats = compute_similarity(era, target_word)
    era_similarity.append(sim_stats)
    print(f"Finished with era {i+1}.")
