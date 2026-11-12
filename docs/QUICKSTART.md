# Запуск и устройство приложения
> Дата создания: 2026-11-09

Я использую одно вычислительное ядро для CLI, локального Web и панели
KStars. Здесь собраны действующие команды 0.10; физическое содержание
сцены описано в [MODELS](../MODELS.md), ограничения — в [LIMITS](../LIMITS.md).

## 1. Подготовка и первый запуск

Требуются Python≥3.12 и uv. Из корня проекта:

```bash
uv sync --locked
uv run lightpollution serve
```

Интерфейс находится по `http://127.0.0.1:8765`. Рабочий каталог по
умолчанию `data/mvp`, исходные HDF5 — `data/blackmarble`; они не входят
в Git. Для другого расположения задаю `--workspace` и `--raw-directory`.
Для разных экземпляров нужны разные workspace и порты.

Учебный пример работает без сети/NASA и помечен как синтетический:

```bash
uv run lightpollution example --spectral --output data/example.json
uv run lightpollution plan --scene data/example.json --output data/example-result
uv run lightpollution datasets
```

В форме выбираю источник/период, точку, UTC или условный режим, атмосферу
и направления. Сохраняю `scene.json`, затем могу открыть его повторно.
Экспорт содержит запрос, разрешённую сцену, результат, CSV и происхождение данных.

## 2. Реальные источники и высоты

Для NASA использую заранее проверенные HDF5 либо явную загрузку:

```bash
uv run lightpollution datasets --download petersburg-2025
```

Доступ Earthdata задаётся локальным `BLACKMARBLE_TOKEN_FILE` или
`BLACKMARBLE_TOKEN`; содержимое секрета не включаю в сцену/документы/Git.
Скрытого скачивания нет: `download_missing=false` по умолчанию.
Нет нужного месяца — нет автоматической подмены годом или соседним периодом.

Для подготовки DSM нужен optional `rasterio`:

```bash
uv sync --locked --extra terrain
uv run lightpollution prepare-terrain \
  --dataset petersburg-copdem90 --directory data/terrain
```

Приготовленный NPZ читается без rasterio. Для нестандартного workspace
подготовленные высоты ищутся в `workspace.parent/terrain`. Входная высота
наблюдателя с DSM — AGL; фактическая MSL сохраняется отдельно.
Данные, манифесты и права — [DATA_SOURCES](DATA_SOURCES.md).

## 3. Глаз и камера

Глаз включаю в dated-сцене полем `"eye": {}`; подробный профиль может
задавать `observer_factor`, `unresolved_background_sp_ratio`, `view`,
`sampling`, `optics`, `display`, `appearance` и `observer_profile`.
Исходное `appearance.mode` — `monochrome`; экспериментальное —
`experimental_stellar_colour`. Экспорт профиля не переносит место/UTC.

Камеру включаю полем `"camera": {}`. Нужны дата и спектральный режим,
который включается валидатором. Пример переопределений внутри сцены:

```json
{
  "camera": {
    "exposure_s": 30,
    "profile": {
      "sensor": {"width_px": 960, "height_px": 640},
      "optics": {"focal_length_mm": 16, "aperture_diameter_mm": 8}
    },
    "view": {"azimuth_deg": 0, "altitude_deg": 85, "rotation_deg": 0}
  }
}
```

Это фрагмент, а не полный вход plan. Физический смысл и остальные поля
профиля находятся в [MODELS](../MODELS.md); готовый отдельный профиль —
[camera-profile.json](../artifacts/camera-model-20261112/city-30s/camera-profile.json).

Камера выдаёт `camera.json`, `camera.fits`, `camera-preview.png`,
`camera-stars.csv`, `camera-profile.json`, `camera-ekos.json`.
Глаз выдаёт JSON/CSV, PNG,16-bit PNG и линейный NPZ; цветовой CSV
появляется при соответствующем режиме. Неизвестная область возвращает
причины/маску, а не правдоподобный выдуманный кадр.

Повтор уже сохранённого сенсорного опыта не требует NASA/DSM/KStars:

```bash
uv run python scripts/run_camera_case.py --render-saved \
  --output scratch/camera-replay --workspace scratch/camera-replay-exports
uv run python scripts/validate_camera.py --output scratch/camera-replay
```

Это повтор сенсора из замороженного света. Новый город/направление
требует нового атмосферного расчёта. В новом замороженном входе сохраняю
также `integration` с полными контурами KStars: один профиль сенсора
не содержит маску местного горизонта. Полная нативная интеграция и сборка —
[KSTARS_INTEGRATION](KSTARS_INTEGRATION.md).

## 4. Сравнение и фотометрическая калибровка

```bash
uv run lightpollution compare \
  --left data/first/result.json --right data/second/result.json
uv run lightpollution fit-photometry \
  --dataset artifacts/photometry-spectra-20261111/ground-dataset.json \
  --output data/shared-calibration.json
```

Fit требует готовых результатов, на которые ссылается набор. Калибровка
проверяет область применимости и не меняет молча полосу/решатель.
Подготовка независимого набора, QA и разделение ночей — [VALIDATION](VALIDATION.md).

## 5. Архитектура и задания

`domain/` содержит SceneSpec/SkyField/StarFluxTable/RunResult и ошибки.
`adapters/` читает файлы/каталоги, `models/` вычисляет по массивам без HTTP.
`application/Planner` подготавливает источники, масштаб и общую сцену.
Существующие публичные старые импорты сохранены для повторения опытов.

Кэш разделён на данные, перенос и сцену. Физические ключи учитывают
источники, место, атмосферу, настройки/код и библиотеку спектров.
Результат дополнительно зависит от времени/каталога/профилей.
Смена AOD пересчитывает лучи, сохраняя одну опору C. Публикация файлов атомарна.

Один сервер использует одного численного работника и допускает 8 активных/
ожидающих заданий. Отмена проверяется между вычислительными шагами;
отдельная NumPy-квадратура не прерывается мгновенно. Новый непокрытый
ракурс может считаться минуты, даже когда повтор из кэша значительно быстрее.

## 6. Локальный API

| Маршрут | Назначение |
|---|---|
| GET `/api/catalogue` | Наборы и периоды |
| POST `/api/jobs` | Создать сцену-задание |
| GET `/api/jobs/{id}` | Состояние/прогресс |
| POST `/api/jobs/{id}/cancel` | Отмена |
| GET `/api/jobs/{id}/result` | Завершённый результат |
| GET `/api/jobs/{id}/export/{name}` | Разрешённый экспорт, включая ZIP |
| POST `/api/compare` | Сравнить результаты |
| GET `/api/spectral-library` | Состав спектральной библиотеки |
| POST `/api/photometry/fit` | Калибровка по завершённым cases |
| POST `/api/eye-profile/export`, `/merge`, `/apply` | Переносимый профиль глаза |
| POST `/api/camera-profile/export`, `/merge`, `/ekos` | Профиль камеры и частичный XML-импорт |
| GET `/api/kstars/capabilities`, POST `/api/kstars/jobs` | Контракт нативного клиента |

В последней группе eye/camera окончания перечислены относительно своих
префиксов. Форматы валидирует [web.py](../src/lightpollution/interfaces/web.py).
JSON профиля камеры имеет `kind=camera_profile`, `schema_version=1`;
глаза — `kind=eye_view_profile`, `schema_version=1`.

## 7. Проверки и экспорт документации

```bash
uv run pytest -q -p no:cacheprovider
uv lock --check
uv run python scripts/check_documentation.py
uv run python scripts/build_theory_bundle.py --output scratch/documentation-export
```

Последняя команда собирает PDF из канонических документов, не создаёт
вторую редактируемую версию теории. Даты в техническом журнале служат
аудиту правок; [HISTORY](../HISTORY.md) организована по этапам.
