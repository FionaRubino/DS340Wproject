import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv(r'C:\Users\fiona\OneDrive\Desktop\DS340Wproject\heart_disease_uci.csv')

train_val, test = train_test_split(df, test_size=0.2, random_state=42)

train, val = train_test_split(train_val, test_size=0.125, random_state=42)

# Save each set to CSV
train.to_csv('train.csv', index=False)
val.to_csv('validation.csv', index=False)
test.to_csv('test.csv', index=False)

print("Splits completed: train.csv, validation.csv, test.csv")