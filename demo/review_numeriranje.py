import pandas as pd

# 1. Učitavamo s 'utf-8' enkodingom (možeš probati i 'cp1250' ako je original iz starog Excela)
# 'keep_default_na=False' sprječava da se prazne tekstualne ćelije pretvore u NaN i pokvare brojeve
df = pd.read_csv('Projekt1.csv', sep=';', encoding='utf-8', keep_default_na=False)

# 2. Prisilno pretvaramo sentence_id u cijeli broj (int) kako ne bi bilo decimala (.0)
df['sentence_id'] = pd.to_numeric(df['sentence_id'], errors='coerce').fillna(0).astype(int)

# 3. Logika za ažuriranje review_id-a
trenutni_review_id = 0
novi_review_ids = []

for sentence_id in df['sentence_id']:
    if sentence_id == 1:
        trenutni_review_id += 1
    novi_review_ids.append(trenutni_review_id)

df['review_id'] = novi_review_ids

# Osiguravamo da je i review_id čisti cijeli broj
df['review_id'] = df['review_id'].astype(int)

# 4. Spremanje s 'utf-8-sig' enkodingom koji garantira da će Excel odmah prepoznati č, ć, ž, đ, š
df.to_csv('Projekt1_azurirano.csv', index=False, sep=';', encoding='utf-8-sig')

print("Uspješno ažurirano bez decimala i s očuvanim kvačicama!")
