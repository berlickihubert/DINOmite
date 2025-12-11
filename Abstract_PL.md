## Abstrakt (PL)

Poster: DINOmite – Adversarial Robustness of DINOv3 Vision Transformers

Modele głębokiego uczenia, zwłaszcza vision transformery, coraz częściej trafiają do zastosowań krytycznych, takich jak rozpoznawanie znaków drogowych czy analiza obrazu medycznego. W takich scenariuszach szczególnie groźne okazują się ataki adwersarialne – subtelne, często niewidoczne dla człowieka modyfikacje obrazu, prowadzące do błędnych decyzji systemu. Jednocześnie rodzina modeli DINOv3 stała się jednym z domyślnych wyborów do ekstrakcji cech wizualnych.

W projekcie DINOmite badamy odporność na ataki adwersarialne modelu DINOv3 wykorzystywanego jako zamrożony ekstraktor cech, nad którym trenujemy lekki klasyfikator dopasowany do danego zbioru danych. Analizujemy zachowanie takiej architektury na trzech datasetach o rosnącej złożoności: CIFAR-10, GTSRB oraz Tiny ImageNet, stosując różne klasy ataków adwersarialnych, m.in. FGSM, PGD i Carlini–Wagner. Uzupełniamy tę analizę implementacją trzech powszechnie stosowanych metod obrony – PGD Adversarial Training, TRADES oraz MART – aby ocenić ich efektywność w połączeniu z reprezentacjami DINOv3.

Wstępne wyniki sugerują, że self-supervised reprezentacje DINOv3 nie są z natury odporne na ataki adwersarialne, lecz stanowią solidny punkt wyjścia do projektowania skuteczniejszych metod wzmacniania bezpieczeństwa modeli wizyjnych.
