import zipfile
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from config import OUTPUT_FILE

def write_and_print(message, out_handle=None):  
    print(message)
    if out_handle:
        out_handle.write(message + '\n')
def main():
 with open(OUTPUT_FILE, 'w') as out:
    with zipfile.ZipFile('DataSets.zip', 'r') as z:
        with z.open('DataSets/telecomChurn.csv') as f:
            df = pd.read_csv(f)
    write_and_print('=== Aperçu des données ===', out)
    write_and_print(df.head().to_string(), out)

    if 'State' in df.columns:
        df.drop(columns='State', inplace=True)

    write_and_print('=== Valeurs manquantes avant suppression ===', out)
    write_and_print(df.isna().sum().to_string(), out)
    df.dropna(inplace=True)
    write_and_print(f"Forme après dropna : {df.shape}", out)

    write_and_print('=== Répartition de la cible Churn ===', out)
    cnt = df['Churn'].value_counts()
    write_and_print(cnt.to_string(), out)
    plt.figure()
    cnt.plot(kind='bar')
    plt.title("Répartition de Churn")
    plt.xlabel("Churn")
    plt.ylabel("Nombre")
    plt.tight_layout()
    plt.savefig('churn_distribution.png')
    plt.show()

    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop('Churn', errors='ignore')
    for col in num_cols:
        plt.figure()
        df[col].hist()
        plt.title(f"Histogramme de {col}")
        plt.xlabel(col)
        plt.ylabel("Fréquence")
        plt.tight_layout()
        plt.savefig(f"hist_{col}.png")
        plt.show()

    plt.figure(figsize=(10, 8))
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True)
    plt.title("Matrice de corrélation")
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.show()

    df['Churn'] = LabelEncoder().fit_transform(df['Churn'])
    cat_cols = [c for c in df.select_dtypes(include=['object']).columns if c != 'Churn']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop('Churn')
    df[num_cols] = StandardScaler().fit_transform(df[num_cols])

    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        'Régression Logistique': LogisticRegression(max_iter=1000),
        'Arbre de Décision': DecisionTreeClassifier(random_state=42)
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        write_and_print(f"--- {name} ---", out)
        write_and_print('Classification Report:', out)
        write_and_print(classification_report(y_test, preds), out)
        write_and_print('Matrice de confusion:', out)
        write_and_print(str(confusion_matrix(y_test, preds)), out)

 write_and_print(f"Résultats enregistrés dans {OUTPUT_FILE}")

 print("Les graphiques ont aussi été sauvegardés en fichiers PNG.")