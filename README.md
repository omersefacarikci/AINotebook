<p align="center">
  <img src="https://github.com/omersefacarikci/omersefacarikci/blob/main/llmbannerim.gif" alt="Banner" />
</p>

## 📌 İçindekiler
- [Deep Learning](#deep-learning)
- [Neural Network](#neural-network)
- [Markov Decision Process (MDP)](#markov-decision-process)
- [Convolutional Neural Network (CNN)](#convolutional-neural-network)

---

<details>
<summary><h2 id="deep-learning">🔹 Deep Learning</h2></summary>

<img src="https://github.com/omersefacarikci/AINotebook/blob/main/dl.jpg" alt="Deep Learning" />
**Deep Learning (Derin Öğrenme)**, yapay zekânın bir alt alanıdır ve çok katmanlı yapay sinir ağları (deep neural networks) kullanarak büyük miktarda veriden öğrenme gerçekleştirir. Geleneksel makine öğrenmesinden farkı, özellik çıkarımını (feature extraction) otomatik olarak yapabilmesidir. Bu sayede insan müdahalesine daha az ihtiyaç duyar.

📌 Özellikler:
- **Büyük veri** gerektirir: Daha fazla veri, daha iyi performans sağlar.
- **Derin katmanlar** sayesinde karmaşık ilişkileri öğrenebilir.
- **GPU ve TPU** gibi donanımlar sayesinde hızlandırılmış hesaplama yapılır.
- Özellikle **bilgisayarla görme (computer vision)**, **doğal dil işleme (NLP)** ve **konuşma tanıma** alanlarında çok başarılıdır.

📌 Avantajları:
- İnsan müdahalesi olmadan otomatik özellik çıkarımı.
- Karmaşık ve lineer olmayan ilişkileri öğrenebilme.
- Birçok alanda **state-of-the-art** sonuçlar elde etme.

📌 Dezavantajları:
- Büyük veri ve güçlü donanım ihtiyacı.
- Yorumlanabilirlik zorluğu (black box problem).

## 📖 Kaynaklar
- [3Blue1Brown – But what is a Neural Network?](https://www.youtube.com/watch?v=aircAruvnKk) : Intuitive visual explanation of how neural networks work.
- [freeCodeCamp – Deep Learning Crash Course](https://www.youtube.com/watch?v=5tvmMX8r_OM) : Covers the core concepts of deep learning in a beginner-friendly way.
- [Fast.ai – Practical Deep Learning](https://course.fast.ai/) : Hands-on course for developers with coding experience.
- [Patrick Loeber – PyTorch Tutorials](https://www.youtube.com/playlist?list=PL1w8k37X_6L9NWeoXQ0IDi1j8weP4YuqF) : Beginner-friendly PyTorch tutorials.

---


</details>

---

<details>
<summary><h2 id="neural-network">🔹 Neural Network</h2></summary>

**Neural Network (Yapay Sinir Ağı)**, biyolojik sinir hücrelerinden (nöronlardan) esinlenmiş bir matematiksel modeldir. Yapısı katmanlardan oluşur: **Giriş katmanı**, **gizli katman(lar)** ve **çıkış katmanı**.

📌 Temel Çalışma Prensibi:
1. Veriler giriş katmanından nöronlara aktarılır.
2. Her bağlantının bir **ağırlığı (weight)** vardır.
3. Nöronlar, **aktivasyon fonksiyonları** aracılığıyla bilgiyi işler.
4. Çıkış, hata ile karşılaştırılır ve **geri yayılım (backpropagation)** ile ağırlıklar güncellenir.

📌 Avantajları:
- Lineer olmayan ilişkileri modelleyebilir.
- Çok çeşitli veri tiplerini işleyebilir (görüntü, metin, ses).
- Paralel hesaplama ile verimli çalışabilir.

📌 Kullanım Alanları:
- Görüntü sınıflandırma
- Ses tanıma
- Tıbbi teşhis sistemleri
- Finansal tahmin modelleri

📊 **Şema:** [Neural Network Şeması](linkini-buraya-koy)

</details>

---

<details>
<summary><h2 id="markov-decision-process">🔹 Markov Decision Process (MDP)</h2></summary>

**Markov Decision Process (MDP)**, belirsizlik içeren ortamlarda karar verme problemlerini modellemek için kullanılan matematiksel bir çerçevedir. Özellikle **reinforcement learning (pekiştirmeli öğrenme)** algoritmalarının temelini oluşturur.

📌 Temel Bileşenleri:
- **Durumlar (States):** Ortamın mevcut koşulları.
- **Eylemler (Actions):** Ajanın seçebileceği hareketler.
- **Geçiş olasılıkları (Transition Probabilities):** Bir durumdan diğerine geçiş ihtimalleri.
- **Ödüller (Rewards):** Ajanın belirli bir durumda aldığı geri bildirim.

📌 MDP’nin Önemi:
- Ajanın uzun vadede en yüksek toplam ödülü elde edecek stratejiyi (policy) öğrenmesini sağlar.
- Karar verme süreçlerinde belirsizlik ve olasılıkların hesaba katılmasını mümkün kılar.

📌 Kullanım Alanları:
- Robotik kontrol sistemleri
- Oyun yapay zekâsı (ör. satranç, Go)
- Öneri sistemleri
- Otonom araçlar

📊 **Şema:** [MDP Şeması](linkini-buraya-koy)

</details>

---

<details>
<summary><h2 id="convolutional-neural-network">🔹 Convolutional Neural Network (CNN)</h2></summary>

**Convolutional Neural Network (CNN)**, özellikle görüntü işleme alanında yaygın olarak kullanılan bir derin öğrenme mimarisidir. İnsan beynindeki görsel korteksten esinlenilmiştir. CNN, görüntülerdeki uzamsal (spatial) ilişkileri öğrenmede çok etkilidir.

📌 Temel Bileşenleri:
- **Convolution Layer (Konvolüsyon Katmanı):** Filtreler (kernels) ile görüntüden özellik çıkarır.
- **Pooling Layer (Havuzlama Katmanı):** Boyut indirgeme yapar, önemli bilgiyi korur.
- **Fully Connected Layer (Tam Bağlantılı Katman):** Özellikleri sınıflandırma için kullanır.

📌 Avantajları:
- Görüntülerdeki kenar, şekil ve nesne gibi özellikleri otomatik çıkarır.
- Parametre paylaşımı sayesinde daha verimli çalışır.
- Yüksek doğruluk oranları sağlar.

📌 Kullanım Alanları:
- Görüntü sınıflandırma (ör. kedi vs köpek)
- Nesne tespiti (ör. otonom araçlardaki trafik işaretleri)
- Yüz tanıma
- Medikal görüntü analizi

📊 **Şema:** [CNN Şeması](linkini-buraya-koy)

</details>

---

## 📖 Kaynaklar
- Ian Goodfellow, Yoshua Bengio, Aaron Courville — *Deep Learning*
- Sutton & Barto — *Reinforcement Learning: An Introduction*
- Stanford CS231n — *Convolutional Neural Networks for Visual Recognition*
