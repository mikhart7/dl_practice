# CV

Учебные задания по компьютерному зрению — практический курс по DL и курс Yandex School of Data Analysis "Deep Vision and Graphics".

## cell_detection.ipynb — детекция объектов
Датасет: yeast cells in microstructures (TU Darmstadt). Собственная реализация TinyYOLO, YOLOv3-style loss, подбор anchor box через k-means, финальная residual-архитектура (`resTinyYOLO`).
**Результат:** val mAP@[0.5:0.95] ≈ 0.696.

## cell_segmentation.ipynb — семантическая сегментация
Тот же датасет с клетками. Собственная реализация U-Net + облегчённый вариант с dilated-свёртками (<50K параметров). Сравнение лоссов: cross-entropy, weighted BCE, soft Dice. Метрики: Jaccard Index, Dice, Accuracy.

## gan.ipynb — генерация лиц
Датасет: CelebA. Реализованы и сравнены DCGAN, LSGAN и WGAN-GP; отдельно — увеличенная Res-архитектура генератора/дискриминатора (до 26M параметров).

## vit-transformers.ipynb — классификация изображений
Датасет: CIFAR-10. Compact Convolutional Transformer (CCT) реализован с нуля: собственный multi-head self-attention, свёрточный токенизатор, sequence pooling, DropPath (по статье "Escaping the Big Data Paradigm with Compact Transformers", arXiv:2104.05704).
**Результат:** accuracy ≈ 86.3%.
