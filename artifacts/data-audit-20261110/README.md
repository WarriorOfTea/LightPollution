# Проверка баз данных и первый технический этап
> Дата создания: 2026-11-10

> Я сохраняю этот отчёт как архив конкретного опыта. Его статусы относятся к описанной версии; текущее устройство — в [MODELS](../../MODELS.md), ход развития — в [HISTORY](../../HISTORY.md).

Подробный [обзор пригодности данных](../../docs/DATA_SOURCES.md) и отдельное
[объяснение физики с основ](https://github.com/WarriorOfTea/LightPollution/blob/52a3604b6603be48e07c4bd0d60a9cf32f0b9f38/docs/PHYSICS_AND_MATH_GUIDE.md).

## Файлы

- `sources.json`: первичные URL, авторство, лицензии, размеры и контрольные
  суммы; крупные оригиналы в Git не включены.
- `astmon-audit.json`: 452 321 исходная запись B/V/R, численные проверки и
  неизвестные флаги условий.
- `galicia-audit.json`: 14 площадок, 6 668 098 исходных записей,
  поправки приборов и положительные конечные показания.
- `astmon-original-sample.csv`: 12 исходных записей без исправления
  несогласованных пространственных заголовков; это не отбор ясного неба.
- `nasa-2025-01-catalogue.json`, `nasa-2025-07-catalogue.json`: реальные
  месячные каталоги, 8 файлов перечислено, 0 месячных HDF5 скачано этим этапом.
- `petersburg-summer-context.json`: геометрия Солнца/Луны 21 июня 2025,
  без вычисления лунной радиансности и без изображения неба.
- `pytest.xml`: 64 passed, 0 failed, 0 skipped; полный запуск текущего этапа.
- `physics-and-math-guide.pdf`: подробное объяснение модели для чтения.
- `data-sources-report.pdf`: отчёт о пригодности найденных баз.
- `verification.json`: окончательные проверки файлов и отчётов.

## Воспроизведение

Оригиналы скачиваются по `download_url` в `sources.json`; для двух Zenodo
наборов сверяются опубликованные MD5. Достаточно сохранить их в локальный
исключённый из Git каталог `data/references/`, затем выполнить:

```bash
uv run lightpollution audit-ground --kind astmon \
  --file data/references/AstMon_evol.csv \
  --output artifacts/data-audit-20261110/astmon-audit.json
uv run lightpollution audit-ground --kind galicia \
  --file data/references/NightSkyBrightnessData_2015.zip \
  --output artifacts/data-audit-20261110/galicia-audit.json
uv run lightpollution catalogue --year 2025 --month 1 \
  --output artifacts/data-audit-20261110/nasa-2025-01-catalogue.json
```

## Авторство и интерпретация

AstMon и его исходный отрывок: José Robles, Jaime Zamorano, Sergio Pascual,
Alejandro Sánchez de Miguel, Jesús Gallego, Kevin J. Gaston,
[Zenodo 4633001](https://zenodo.org/records/4633001), CC BY 4.0.
Galicia: Salvador Bará, [Dryad / Zenodo](https://zenodo.org/records/4977265),
CC0 по метаданным. Контрольные суммы и полные библиографические сведения —
в `sources.json`. Приведённые счётчики описывают файлы и числовые отборы,
не физическую точность нашей модели и не валидацию человеческого восприятия.

Работа перенесена в `main` без переписывания прежней истории. Новые
исследовательские изображения неба не создавались; абсолютная калибровка,
лунное рассеяние, сезонный масштаб и модели глаза/сенсора остаются открыты.
