import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

print('Loading Dataset')
df = pd.read_csv('Skin_NonSkin.txt', header=None, names=['X'])

print('Preprocessing Data')
df['Y'] = df['X'].apply(lambda x: int(x.split('\t')[-1]))
df['R'] = df['X'].apply(lambda x: int(x.split('\t')[0]))
df['G'] = df['X'].apply(lambda x: int(x.split('\t')[1]))
df['B'] = df['X'].apply(lambda x: int(x.split('\t')[2]))
df.drop(columns=['X'],inplace=True)

df['Y'] = df['Y'].apply(lambda x: 0 if x==2 else 1)

model = SVC()

X_train, X_test, y_train, y_test = train_test_split(df[['R','G','B']],df['Y'],shuffle=True)

print('Training Model')
model.fit(X_train, y_train)
print('Finished Training')

acc = model.score(X_test, y_test)
print(f'Model Accuracy = {acc:.3f}%')

from joblib import dump
f_name = 'svc_model.joblib'
dump(model, f_name)
print(f'Mdoel saved as {f_name}')