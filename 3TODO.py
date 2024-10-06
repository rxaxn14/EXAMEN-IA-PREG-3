import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, KBinsDiscretizer, MinMaxScaler

df = pd.read_csv('C:/Users/ROXANA CASTILLO/Desktop/354/gym_members_exercise_tracking.csv')

def convertir_a_arff(df, nombre_archivo_arff):
    with open(nombre_archivo_arff, 'w') as f:
        f.write(f"@RELATION gym_members_exercise_tracking\n\n")
        for column in df.columns:
            if df[column].dtype == 'object':
                unique_values = df[column].unique()
                unique_values_str = ','.join([str(v) for v in unique_values])
                f.write(f"@ATTRIBUTE {column} {{{unique_values_str}}}\n")
            else:
                f.write(f"@ATTRIBUTE {column} NUMERIC\n")
        f.write("\n@DATA\n")
        for index, row in df.iterrows():
            row_str = ','.join([str(val) for val in row])
            f.write(f"{row_str}\n")

convertir_a_arff(df, 'C:/Users/ROXANA CASTILLO/Desktop/354/gym_members_exercise_tracking.arff')

# OneHotEncoder para la columna 'Workout_Type'
onehotencoder = OneHotEncoder()
onehot_encoded = onehotencoder.fit_transform(df[['Workout_Type']]).toarray()
df_onehot = pd.DataFrame(onehot_encoded, columns=onehotencoder.get_feature_names_out(['Workout_Type']))
df = df.join(df_onehot)
df.drop(columns=['Workout_Type'], inplace=True)

# LabelEncoder para la columna 'Experience_Level'
labelencoder = LabelEncoder()
df['Experience_Level'] = labelencoder.fit_transform(df['Experience_Level'])

# Discretización para la columna 'Age' en 3 categorías
discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
df['Age_Discretized'] = discretizer.fit_transform(df[['Age']])

# Normalización para las columnas 'Weight (kg)', 'Height (m)' y 'BMI'
scaler = MinMaxScaler()
df[['Weight (kg)', 'Height (m)', 'BMI']] = scaler.fit_transform(df[['Weight (kg)', 'Height (m)', 'BMI']])

df.head()
