\# Mood Tracker AI



AI destekli günlük ruh hâli takip ve öneri sistemi. Kullanıcının günlük metin ve sayısal verilerini analiz ederek ruh hâlini tahmin eder ve kişiselleştirilmiş öneriler sunar.



---



\## Açıklama



Bu proje, kullanıcının girdiği kısa günlük özet metni ve sayısal verilerini (adım sayısı, uyku süresi, sıcaklık, hava durumu) birlikte değerlendirerek ruh hâli tahmini yapar. Model, basit bir Random Forest regresyon ile sayısal ve metinsel veriyi birleştirir. Ayrıca Transformers kütüphanesi ile DistilBERT kullanılarak duygu analizi entegre edilmiştir.  



Düşük ruh hâli veya negatif duygu durumunda kullanıcıya nefes egzersizi, kısa yürüyüş gibi öneriler sunulur. Yüksek ruh hâlinde motivasyonu artıracak tavsiyeler verilir.



---



\## Özellikler



\- Günlük ruh hâli tahmini (1–10)

\- Metin analizi ile pozitif/negatif duygu skorları

\- Kişiselleştirilmiş öneriler (egzersiz, meditasyon, motivasyon tavsiyeleri)

\- Son 7 günün görselleştirilmiş takibi

\- Kullanıcı dostu Streamlit arayüzü

\- Kolayca veri eklenebilir ve loglanabilir



---



\## Kullanım



\### Gereksinimler

```bash

pip install -r requirements.txt



Uygulamayı çalıştırma

streamlit run app.py



Örnek Girdi

Tarih	Günün Özeti	Adım	Uyku (saat)	Sıcaklık	Hava

2025-08-28	Bugün biraz yorgunum	4500	6	24	cloudy

2025-08-29	Enerjik ve motiveyim	9000	8	30	sunny

Öneriler



Düşük ruh hâli veya negatif metin → nefes egzersizi, kısa yürüyüş



Orta ruh hâli → hafif aktivite ve sevdiğin müzik



Yüksek ruh hâli → motivasyonu artıracak hedefler

