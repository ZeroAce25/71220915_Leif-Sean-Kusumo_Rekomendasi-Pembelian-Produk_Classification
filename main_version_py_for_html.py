# Langkah 1: Impor pustaka yang dibutuhkan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import category_encoders as ce
import joblib  # Untuk menyimpan model dan data

# Langkah 2: Membaca file CSV
df = pd.read_csv('data_fix.csv')

# Langkah 3: Konversi kolom TIME_STAMP ke datetime
df['TIME_STAMP'] = pd.to_datetime(df['TIME_STAMP'])

# Langkah 4: Memeriksa data yang null
null_check = df.isnull().sum()
any_null = df.isnull().any().any()

print("Pemeriksaan nilai null di setiap kolom:")
print(null_check)

if any_null:
    print("\nAda nilai null di DataFrame.")
else:
    print("\nTidak ada nilai null di DataFrame.")

# Langkah 5: Normalisasi data (contoh: Min-Max Scaling)
df['RATING_NORMALIZED'] = (df['RATING'] - df['RATING'].min()) / (df['RATING'].max() - df['RATING'].min())

# Langkah 6: Membuat target diskret dari rating
df['RATING_DISCRETE'] = pd.cut(df['RATING_NORMALIZED'], bins=3, labels=[0, 1, 2]).astype(int)

# Langkah 7: Menampilkan Distribusi Data (contoh: Distribusi Normal)
plt.figure(figsize=(8, 6))
sns.histplot(df['RATING_NORMALIZED'], bins=20, kde=True, stat='density')
x = np.linspace(df['RATING_NORMALIZED'].min(), df['RATING_NORMALIZED'].max(), 1000)
mu = df['RATING_NORMALIZED'].mean()
sigma = df['RATING_NORMALIZED'].std()
y = (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
plt.plot(x, y, color='red', linestyle='--', label='Distribusi Normal')
plt.title('Distribusi Nilai Rating (Setelah Normalisasi)')
plt.xlabel('Rating Normalized')
plt.ylabel('Density')
plt.legend()
plt.show()

# Menampilkan DataFrame baru
print("\nDataFrame baru:")
print(df.head())

# Langkah 8: Menentukan fitur dan target
X = df.drop(['RATING', 'TIME_STAMP', 'RATING_NORMALIZED', 'RATING_DISCRETE'], axis=1)  # Fitur
y = df['RATING_DISCRETE']  # Target diskret

# Langkah 9: Encoding Kategori yang Sering untuk 100% data latih
encoder = ce.TargetEncoder(cols=['USER_ID', 'PRODUCT_ID'])
X_encoded = encoder.fit_transform(X, y)

# Simpan encoder yang dilatih ke file .pkl
joblib.dump(encoder, 'encoder.pkl')

# Simpan data yang telah di-encode untuk prediksi
joblib.dump(X_encoded, 'X_encoded.pkl')

# Langkah-langkah lainnya tetap sama...


# Langkah 11: Membuat dan melatih model Decision Trees dengan 100% data latih
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_encoded, y)

# Langkah 12: Menyimpan model yang dilatih dengan 100% data latih ke file .pkl
joblib.dump(dt_classifier, 'dt_classifier.pkl')

# Langkah 13: Membagi data menjadi data latih dan data uji (80% data latih)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Langkah 14: Membuat dan melatih model Decision Trees dengan 80% data latih
dt_classifier_80 = DecisionTreeClassifier(random_state=42)
dt_classifier_80.fit(X_train, y_train)

# Langkah 15: Menyimpan model yang dilatih dengan 80% data latih ke file .pkl
joblib.dump(dt_classifier_80, 'dt_classifier_80.pkl')

# Langkah 16: Memprediksi kelas untuk data uji (80%)
y_pred_80 = dt_classifier_80.predict(X_test)

# Langkah 17: Menghitung Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_80)
print("\nConfusion Matrix pada data uji:")
print(conf_matrix)

# Langkah 18: Menghitung laporan klasifikasi
class_report = classification_report(y_test, y_pred_80)
print("\nClassification Report pada data uji:")
print(class_report)

# Langkah 19: Fungsi Rekomendasi Produk Berdasarkan Prediksi Model
def rekomendasi_produk_berdasarkan_prediksi(X, model, top_n=5):
    # Membuat prediksi untuk setiap produk
    rating_pred = model.predict(X)
    # Membuat DataFrame hasil prediksi
    pred_df = X.copy()
    pred_df['RATING_PREDICTED'] = rating_pred
    # Mengelompokkan data berdasarkan PRODUCT_ID dan menghitung rata-rata rating prediksi
    rekomendasi = pred_df.groupby('PRODUCT_ID')['RATING_PREDICTED'].mean().sort_values(ascending=False).head(top_n)
    return rekomendasi

# Menggunakan model yang dilatih dengan 100% data latih untuk prediksi
print("\nRekomendasi produk berdasarkan prediksi model (dengan 100% data latih):")
print(rekomendasi_produk_berdasarkan_prediksi(X_encoded, dt_classifier))
