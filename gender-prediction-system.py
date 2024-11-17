import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

class FetalGenderPredictor:
    def __init__(self):
        self.model = GaussianNB()
        self.label_encoders = {}
        self.features = [
            'usia_ibu',
            'bentuk_perut',
            'warna_puting',
            'morning_sickness',
            'nafsu_makan',
            'jenis_makanan',
            'pergerakan_janin',
            'detak_jantung_janin',
            'pigmentasi_kulit',
            'posisi_tidur'
        ]
        
    def preprocess_data(self, data):
        processed_data = data.copy()
        
        categorical_features = [
            'bentuk_perut',
            'warna_puting',
            'morning_sickness',
            'nafsu_makan',
            'jenis_makanan',
            'pergerakan_janin',
            'pigmentasi_kulit',
            'posisi_tidur'
        ]
        
        for feature in categorical_features:
            self.label_encoders[feature] = LabelEncoder()
            processed_data[feature] = self.label_encoders[feature].fit_transform(processed_data[feature])
            
        return processed_data
    
    def train(self, X_train, y_train):
        self.model.fit(X_train[self.features], y_train)
        
    def predict(self, X_test):
        predictions = self.model.predict(X_test[self.features])
        probabilities = self.model.predict_proba(X_test[self.features])
        return predictions, probabilities
    
    def evaluate(self, X_test, y_test):
        predictions = self.model.predict(X_test[self.features])
        return {
            'accuracy': accuracy_score(y_test, predictions),
            'report': classification_report(y_test, predictions)
        }
    
    def explain_prediction(self, data, prediction, probabilities):
        explanation = {
            'predicted_gender': 'Laki-laki' if prediction == 0 else 'Perempuan',
            'confidence': f"{max(probabilities) * 100:.2f}%",
            'indicators': []
        }
        
        if data['bentuk_perut'] in ['Runcing', 'Menonjol ke depan']:
            explanation['indicators'].append({
                'feature': 'Bentuk Perut',
                'value': data['bentuk_perut'],
                'indication': 'Laki-laki',
                'confidence': '70%'
            })
        elif data['bentuk_perut'] in ['Bulat', 'Melebar']:
            explanation['indicators'].append({
                'feature': 'Bentuk Perut',
                'value': data['bentuk_perut'],
                'indication': 'Perempuan',
                'confidence': '65%'
            })
            
        if float(data['detak_jantung_janin']) > 140:
            explanation['indicators'].append({
                'feature': 'Detak Jantung',
                'value': data['detak_jantung_janin'],
                'indication': 'Perempuan',
                'confidence': '75%'
            })
        else:
            explanation['indicators'].append({
                'feature': 'Detak Jantung',
                'value': data['detak_jantung_janin'],
                'indication': 'Laki-laki',
                'confidence': '75%'
            })
            
        if data['morning_sickness'] == 'Berat':
            explanation['indicators'].append({
                'feature': 'Morning Sickness',
                'value': data['morning_sickness'],
                'indication': 'Perempuan',
                'confidence': '60%'
            })
            
        return explanation

# Contoh penggunaan dengan beberapa input berbeda
def main():
    # Inisialisasi predictor
    predictor = FetalGenderPredictor()
    
    # Membuat dataset sampel untuk training
    print("Membuat dataset training...")
    dataset = pd.DataFrame({
        'usia_ibu': np.random.randint(20, 40, 100),
        'bentuk_perut': np.random.choice(['Runcing', 'Bulat', 'Melebar', 'Menonjol ke depan'], 100),
        'warna_puting': np.random.choice(['Gelap', 'Sangat Gelap', 'Sedikit Gelap'], 100),
        'morning_sickness': np.random.choice(['Ringan', 'Sedang', 'Berat'], 100),
        'nafsu_makan': np.random.choice(['Meningkat', 'Menurun', 'Normal'], 100),
        'jenis_makanan': np.random.choice(['Manis', 'Asam', 'Asin', 'Pedas'], 100),
        'pergerakan_janin': np.random.choice(['Aktif', 'Sangat Aktif', 'Normal'], 100),
        'detak_jantung_janin': np.random.randint(120, 160, 100),
        'pigmentasi_kulit': np.random.choice(['Meningkat', 'Sedikit', 'Normal'], 100),
        'posisi_tidur': np.random.choice(['Kiri', 'Kanan', 'Terlentang'], 100),
        'jenis_kelamin': np.random.choice([0, 1], 100)  # 0: Laki-laki, 1: Perempuan
    })
    
    # Preprocessing dataset
    processed_dataset = predictor.preprocess_data(dataset)
    
    # Split dan training
    X_train, X_test, y_train, y_test = train_test_split(
        processed_dataset, dataset['jenis_kelamin'], test_size=0.2, random_state=42
    )
    predictor.train(X_train, y_train)
    
    # Evaluasi model
    print("\nEvaluasi Model:")
    evaluation = predictor.evaluate(X_test, y_test)
    print(f"Akurasi: {evaluation['accuracy']*100:.2f}%")
    print("\nLaporan Klasifikasi:")
    print(evaluation['report'])
    
    # Contoh beberapa kasus prediksi
    test_cases = [
        {
            'usia_ibu': 28,
            'bentuk_perut': 'Bulat',
            'warna_puting': 'Gelap',
            'morning_sickness': 'Berat',
            'nafsu_makan': 'Meningkat',
            'jenis_makanan': 'Manis',
            'pergerakan_janin': 'Normal',
            'detak_jantung_janin': '145',
            'pigmentasi_kulit': 'Meningkat',
            'posisi_tidur': 'Kiri'
        },
        {
            'usia_ibu': 32,
            'bentuk_perut': 'Runcing',
            'warna_puting': 'Sedikit Gelap',
            'morning_sickness': 'Ringan',
            'nafsu_makan': 'Normal',
            'jenis_makanan': 'Asin',
            'pergerakan_janin': 'Sangat Aktif',
            'detak_jantung_janin': '130',
            'pigmentasi_kulit': 'Sedikit',
            'posisi_tidur': 'Kanan'
        }
    ]
    
    print("\nContoh Prediksi:")
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nKasus #{i}:")
        print("Input:")
        for key, value in test_case.items():
            print(f"- {key}: {value}")
            
        # Preprocessing data test
        test_df = pd.DataFrame([test_case])
        processed_test = predictor.preprocess_data(test_df)
        
        # Prediksi
        prediction, probabilities = predictor.predict(processed_test)
        explanation = predictor.explain_prediction(test_case, prediction[0], probabilities[0])
        
        print("\nHasil Prediksi:")
        print(f"Prediksi Jenis Kelamin: {explanation['predicted_gender']}")
        print(f"Tingkat Keyakinan: {explanation['confidence']}")
        print("\nIndikator-indikator:")
        for indicator in explanation['indicators']:
            print(f"- {indicator['feature']}: {indicator['value']}")
            print(f"  Mengindikasikan: {indicator['indication']} (Keyakinan: {indicator['confidence']})")

if __name__ == "__main__":
    main()
