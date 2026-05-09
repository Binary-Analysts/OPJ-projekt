import numpy as np
import pandas as pd
import statsmodels.stats.inter_rater as irr

file_path = "sentimenti.xlsx"  
df = pd.read_excel(file_path)


ratings_df = df.iloc[:, 1:6]  # stupci B-F 


label_mapping = {
    'Positive': 0,
    'Neutral': 1,
    'Negative': 2,
    'Sarcasm': 3,
    'Mixed': 4
}
ratings_df = ratings_df.replace(label_mapping)


ratings_raw = ratings_df.to_numpy()


ratings_agg, category_counts = irr.aggregate_raters(ratings_raw)


kappa = irr.fleiss_kappa(ratings_agg, method='fleiss')


category_names = ['Positive', 'Neutral', 'Negative', 'Sarcasm', 'Mixed']
ratings_export = pd.DataFrame(ratings_agg, columns=category_names)

print("Aggregated Ratings Matrix:\n", ratings_export)
print("\nCategory Counts:", category_counts)
print(f"\nFleiss' Kappa: {kappa:.4f}")
