# sk_video_stimulus

**Потоковое преобразование видео в низкоуровневые характеристики динамики зрительного стимула.**

[English](README_EN.md) · [Алгоритм](docs/algorithm.md) · [Калибровка](docs/calibration.md) · [Признаки](docs/features.md) · [Интеграция](docs/emotional-state-integration.md) · [Примеры](docs/examples.md)

![Из кадров видео во временную структуру зрительных стимулов](assets/images/03-temporal-structure.webp)

`sk_video_stimulus` отвечает на вопрос **«как меняется зрительный поток во времени?»**. Пакет не распознаёт объекты, жесты или эмоции. Он превращает кадры видео либо поток камеры в одномерный сигнал `x_t`, а затем вычисляет четыре характеристики его динамики:

| Признак | Смысл |
|---|---|
| `activity` | подписанная интенсивность и направление изменения |
| `anxiety` | нерегулярность и резкость динамики; это не распознанная эмоция |
| `tempo` | частота значимых максимумов и минимумов, событий/с |
| `deviation` | мощность отклонения сигнала от спокойного режима |

```text
кадры → пространственные μ-признаки → быстрые/медленные изменения
      → глобальный сигнал x_t → activity / anxiety / tempo / deviation
      → калибровка относительно нейтрального видео
```

## Установка

Из GitHub со всеми зависимостями для видео и графиков:

```bash
git clone https://github.com/iproxy3d/sk_video_stimulus.git
cd sk_video_stimulus
python -m pip install -e ".[all]"
```

Для программной обработки уже загруженных кадров достаточно базовой установки:

```bash
python -m pip install -e .
```

## Быстрый запуск

Три демонстрационных видео должны находиться в [`examples/videos`](examples/videos/README.md). После их добавления весь сценарий выполняется четырьмя командами.

### 1. Извлечь исходные признаки

```bash
sk-video-stimulus examples/videos/input_25fps.mp4 \
  --output results/video_features.csv \
  --image-size 256 \
  --region-grid 4
```

### 2. Построить нейтральный профиль

```bash
sk-video-stimulus examples/videos/neutral_25fps.mp4 \
  --calibrate-profile results/video_neutral_profile.json \
  --output results/neutral_features.csv
```

### 3. Нормализовать тестовое видео

```bash
sk-video-stimulus examples/videos/test_25fps.mp4 \
  --profile results/video_neutral_profile.json \
  --output results/test_features.csv
```

### 4. Построить диагностические графики

```bash
sk-video-stimulus-plot \
  results/video_features.csv \
  results/test_features.csv \
  --output-dir results/plots
```

Исходные имена скриптов также сохранены для совместимости:

```bash
python video_feature_extractor.py examples/videos/input_25fps.mp4 --output results/video_features.csv
python plot_video_features.py results/video_features.csv --output-dir results/plots
```

## Программное использование

```python
from sk_video_stimulus import VideoFeatureExtractor

extractor = VideoFeatureExtractor()

for frame in frames:
    sample = extractor.process_frame(frame)
    if sample.ready:
        print(sample.activity, sample.anxiety, sample.tempo, sample.deviation)
```

Состояние потокового анализатора можно сохранить через `state_dict()` и восстановить через `load_state_dict()`.

## Что происходит внутри

![Пространственное кодирование кадра](assets/images/01-spatial-encoding.webp)

Кадр переводится в оттенки серого и делится на области. В каждой области вычисляются отклики 15 знаковых пространственных масок. Между соседними кадрами измеряются быстрые и медленные изменения, после чего глобальный максимум по модулю образует `x_t` с сохранением знака.

![От пространственных признаков к временному сигналу](assets/images/02-temporal-signal.webp)

Подробное математическое описание приведено в [docs/algorithm.md](docs/algorithm.md).

## Нейтральная калибровка

Абсолютное значение признака ещё не показывает значимость стимула: шум камеры, сжатие и микродвижения дают ненулевой фон. Поэтому пакет строит фиксированный профиль по нейтральному видео и измеряет отклонение текущего сигнала от него.

![Калибровка нейтрального видеостимула](assets/images/04-neutral-calibration.webp)

Пример готового профиля находится в [`examples/profiles/video_neutral_profile.json`](examples/profiles/video_neutral_profile.json).

## Результаты и диагностика

Репозиторий содержит реальные CSV и заранее построенные графики:

- [`examples/data`](examples/data) — сырые, нейтральные и нормализованные временные ряды;
- [`assets/plots`](assets/plots) — временные графики, автокорреляция и спектр;
- [docs/examples.md](docs/examples.md) — объяснение результатов и ограничений.

![Сопоставление кадров с временными рядами](assets/images/05-frames-and-features.webp)

## Место в системе эмоционального состояния

`sk_video_stimulus` извлекает сенсорную динамику. Семантическое описание происходящего — кто действует, что делает и с каким объектом — выполняет отдельная модель [Florence-2 SVOA](https://github.com/iproxy3d/florence2-svoa).

Опциональный `VideoProtoEmotionMapper` переводит нормализованные признаки в потенциалы протoэмоций для интеграции с [emotional-state-agents](https://github.com/iproxy3d/emotional-state-agents). Он не является классификатором эмоций и не создаёт состояние с нуля.

## Структура репозитория

```text
src/sk_video_stimulus/
├── extractor.py        # кадры → x_t → четыре признака → калибровка
└── plotting.py         # временные ряды, ACF и частотный спектр
tests/                  # алгоритмические и интеграционные тесты
examples/data/          # готовые CSV
examples/profiles/      # нейтральный профиль
examples/videos/        # место для трёх демонстрационных MP4
assets/images/          # объясняющие инфографики
assets/plots/           # рассчитанные результаты
docs/                   # подробное описание метода
```

## Тесты

```bash
python -m unittest discover -s tests -v
```

## Статус и ограничения

Это исследовательская реализация. Текущие пороги и масштабы проверены на приложенных примерах, но ещё не являются универсально валидированными для разных камер, частот кадров и типов сцен. Сравнивать разные записи следует только при совместимых настройках и нейтральном профиле.

## Лицензия

Код распространяется по лицензии [MIT](LICENSE). Пользователь отвечает за право распространения добавляемых видео.
