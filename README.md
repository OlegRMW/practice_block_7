# Суммаризация диалогов: Baseline, Qwen3 и LoRA

Модульная версия логики `notebooks/main.ipynb`. Исходные CSV и вся папка `notebooks/` используются только для чтения. План и требования находятся в `spec.md`, `plan.md`, ход работ — в `tasks.md`.

## Структура

```text
configs/                   YAML для baseline, qwen (Ollama), lora
scripts/train.py           дообучение Qwen3 с LoRA
scripts/evaluate.py        инференс test + BLEU/ROUGE + необязательный судья, без таблиц
scripts/report.py          сравнение моделей по готовым CSV
src/text_summary_block_7/  конфиги, данные, модели, обучение, метрики, артефакты, отчёты
tests/                     автоматические проверки
results/                   новые CSV и настройки запусков
models/                    новые адаптеры и checkpoints (не включаются в Git)
data/                      исходные train/test/validation (не изменяются)
notebooks/                 исходные исследования, предсказания и адаптер (не изменяются)
```

## Установка

Нужен Python 3.12 или 3.13 и Poetry. LoRA использует NVIDIA GPU, CUDA-сборку PyTorch и Unsloth. В проекте зафиксированы PyTorch 2.14.0+cu130, Transformers 5.17.0, Unsloth/Unsloth Zoo 2026.11.4, TRL 1.15.0 и Datasets 4.8.5. Для Torch/torchvision задан отдельный официальный CUDA-источник Poetry: обычная установка CPU-сборки не подменяет GPU-зависимость.

```powershell
poetry install
poetry run python -m nltk.downloader punkt punkt_tab
poetry run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Для Windows нужен `triton-windows` (в зависимостях проекта); требования к драйверу и инструментам сборки приведены в [инструкции Unsloth](https://unsloth.ai/docs/get-started/install/windows-installation). Источник CUDA wheels — [PyTorch](https://pytorch.org/get-started/locally/). Короткие реальные проверки и ограничения локальной среды записываются в `verification.md`.

Для zero-shot запустите Ollama и загрузите модель:

```powershell
ollama pull qwen3:0.6b
```

Для судьи добавьте `ROUTERAI_API_KEY` в окружение либо локальный `.env` в корне. Ключ не записывается в YAML или результаты. API вызывается **только** с флагом `--llm-as-judge`; без него ключ не нужен. Судья использует RouterAI и `openai/gpt-oss-120b`, prompt и критерии из notebook.

## Конфиги и аргументы командной строки

`baseline.yaml`, `qwen.yaml`, `lora.yaml` — самостоятельные конфиги. Их разделы: `seed`, `data`, `paths`, `model`, `generation`, `metrics`, `judge`, `lora`, `training`. Последние два применяются только к обучению LoRA. Все настройки можно переопределить без редактирования YAML.

Приоритет: YAML → повторяемый `--set` → явно указанный именованный аргумент. Например:

```powershell
poetry run python scripts/train.py --config configs/lora.yaml --learning-rate 0.0002 --set lora.r=32
```

Здесь остальные параметры берутся из YAML, learning rate равен `0.0002`, ранг LoRA — `32`. Значения `--set` разбираются как YAML: числа, `true`/`false`, списки и строки. Неизвестные ключи и неверные типы отвергаются до загрузки модели. Относительные пути в YAML считаются от каталога конфига, пути из CLI — от текущего каталога.

## Дообучение

```powershell
poetry run python scripts/train.py --config configs/lora.yaml
```

По умолчанию используются `Qwen/Qwen3-0.6B`, 3000 train-примеров с seed 42, полный validation, 3 эпохи, LoRA r/alpha 16, learning rate `1e-4`. Обучаются только параметры адаптера, loss — только по completion. Train и validation не пересекаются по ID. Входы: `data/train.csv/train.csv`, `data/validation.csv`.

Адаптер и токенизатор сохраняются в `models/qwen3_summary_lora`, checkpoints — в его подкаталог; `training-run.yaml` содержит настройки и результат обучения. Существующий адаптер в выходном каталоге не перезаписывается: выберите новый `--output-dir`. В новых версиях TRL `warmup_ratio` переводится в эквивалентное число `warmup_steps`.

## Инференс и оценка

```powershell
poetry run python scripts/evaluate.py --config configs/baseline.yaml
poetry run python scripts/evaluate.py --config configs/qwen.yaml
poetry run python scripts/evaluate.py --config configs/lora.yaml
```

Каждый запуск заново генерирует ответы выбранной модели на всех 1500 строках test, пересчитывает индивидуальные и сводные BLEU/ROUGE и обновляет только результаты этой модели. В терминале остаются прогресс и короткое сообщение о сохранении; таблицы выводит только `report.py`. Для другого test можно передать `--test PATH`; обязательные колонки: `id`, `dialogue`, `summary`, ID уникальны и непусты. Пустые диалоги и эталоны в test вызывают ошибку; пустое предсказание оценивается как пустой ответ.

Чтобы использовать существующий адаптер без нового обучения:

```powershell
poetry run python scripts/evaluate.py --config configs/lora.yaml --adapter-path notebooks/models/qwen3_summary_lora
```

Чтобы дополнительно пересчитать LLM-as-judge:

```powershell
poetry run python scripts/evaluate.py --config configs/lora.yaml --adapter-path notebooks/models/qwen3_summary_lora --llm-as-judge
```

Судья оценивает 100 случайных диалогов с seed 42, одинаковых для всех моделей при одинаковом test. BLEU/ROUGE по-прежнему считаются на всём test. Для проверки можно изменить объём: `--judge-sample-size 1`; seed меняется через `--set judge.seed=42`. На test меньше 100 строк используются все доступные строки. С флагом выполняются новые запросы, даже если старые оценки имеются. В YAML включить API-запросы нельзя.

Без флага судьи старые оценки не удаляются. Если модель, диалог, эталон и текст ответа совпадают, оценки показываются как текущие. Если ответ изменился, старые оценки выводятся отдельно как исторические. Ошибки API/JSON сохраняются с пустыми оценками и типом ошибки, не превращаются в нули; при ошибках судьи команда завершается ненулевым кодом, сохраняя успешные результаты.

Baseline сохраняет исходную склейку первого и последнего предложения без дополнительного разделителя. Prompts и лимиты генерации zero-shot (64 токена) и LoRA (128 токенов) сохранены раздельно.

## Сравнительный отчёт

```powershell
poetry run python scripts/report.py
poetry run python scripts/report.py --models baseline qwen_lora
```

Отчёт показывает только краткие сводные таблицы BLEU, ROUGE, судьи и исторических оценок: одна строка на модель и конфигурацию судьи. Построчные оценки ответов остаются в CSV и не печатаются в терминале. GPU, Ollama и API-ключ не нужны. Данные берутся из CSV, согласованность запуска проверяется по `run.yaml`.

В `results/` подготовлены метрики трёх моделей из существующего `notebooks/qwen_lora_test_predictions.csv`; исходный test проверяется по ID, текстам и порядку. Старый `routerai_judge_results.csv` содержит 99/100/100 оценок Baseline/Qwen/LoRA. В нём нет текста оценённого ответа: такие оценки показаны как исторические с непроверенной связью, а не как новые оценки текущих ответов. Функция `bootstrap.py` использовалась однократно для подготовки данных; обычный `report.py` ничего не генерирует и не перезаписывает.

## Форматы и метрики

В `results/<baseline|qwen|qwen_lora>/`:

| Файл | Содержимое |
| --- | --- |
| `predictions.csv` | ID, диалог, эталон, ответ, модель, run ID и хеш ответа |
| `metrics.csv` | BLEU и ROUGE каждой строки с ключами ответа |
| `summary.csv` | Размер выборки, corpus BLEU, средние sentence BLEU и ROUGE |
| `judge.csv` | Четыре критерия, overall, источник, версия судьи, хеш ответа и статус |
| `run.yaml` | Эффективный конфиг, хеш входа, версии метрик и параметры расчёта |

`results/all_models_metrics.csv` объединяет индивидуальные метрики всех моделей и подтверждённые текущие оценки судьи. Исторические оценки остаются в отдельных `judge.csv`.

- **Sentence BLEU:** SacreBLEU, шкала 0–100, `tokenize=13a`, `smooth_method=exp`, `effective_order=True`.
- **Corpus BLEU:** отдельный расчёт по всем ответам с `effective_order=False`; это не среднее sentence BLEU.
- **ROUGE-1/2/L/Lsum:** F1, шкала 0–1, среднее по строкам; `use_stemmer=False`. Для Lsum предложения разделяются через NLTK. Эти настройки явно зафиксированы, поэтому Lsum может отличаться от старого notebook без подготовки границ предложений.
- **Судья:** faithfulness, coverage, relevance, coherence — целые 1–5, `overall` — их среднее. Разные версии настроек судьи агрегируются отдельно. Пропуск в таблице обозначается `—`.

Число оценок судьи показывается отдельно от числа test-ответов. Таблицы по разным test-выборкам нельзя напрямую сравнивать; отчёт отмечает различие ID. Запись в `notebooks/` и исходные датасеты блокируется, включая пути через ссылки. Неполная публикация CSV обнаруживается по идентификаторам запуска.

## Проверки

```powershell
poetry run python -m unittest discover -v
poetry check
```

Тесты используют маленькие временные данные и не обращаются к API. Реальные GPU/API smoke-проверки проводятся отдельно с сохранением результатов вне `notebooks/`. Список исходных защищённых файлов и SHA-256 для локального контроля находится в игнорируемой `.verification/protected.json`.
