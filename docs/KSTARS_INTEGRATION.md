# Интеграция в KStars
> Дата создания: 2026-11-11

Я встраиваю интерфейс в настоящий KStars как C++/Qt `QDockWidget`.
Физический расчёт работает в отдельном локальном Python-процессе; карта
остаётся отзывчивой. Web — другой клиент того же ядра, не способ
встроить панель. Исходники: [integrations/kstars](../integrations/kstars/).

## 1. Что я переиспользую

| Возможность KStars | Фактическое применение |
|---|---|
| GeoLocation/SimClock/SkyMap | Атомарные место, виртуальный UTC и направление центра |
| SkyPoint | Геометрические az/alt; рефракция повторно не вводится |
| StarComponent/DeepStarComponent | Установленные каталоги, JIT-позиции, V/B−V и HD для камеры |
| ArtificialHorizon | Штатная проверка выбранного направления |
| FITS Viewer | Открытие линейного синтетического FITS, stretch/zoom/гистограмма |
| XML camera-data Ekos | Частичный перенос pitch и RN выбранного режима/gain |
| Критерий Exposure Calculator | Та же допустимая доля шума считывания; физическая b из нашей сцены |

Ekos уже учитывает фон, QE, размер пикселя и RN; INDI CCD Simulator уже
имеет sky-glow, шум и дефекты. Моё добавление — направленная физическая
сцена и электронный фон по полю. Я не выдаю существующие функции за новые.
TerrainComponent с панорамой не заменяет метрический DSM двух путей света.

## 2. Управление панелью

В панели выбираю период, DSM, AGL, AOD, ближние площадки и спектр;
включаю глаз/экспериментальный цвет либо камеру. У камеры доступны
выдержка/фокусное/апертура, JSON-профиль и частичный XML-импорт Ekos.
Остальные настройки сохраняются при открытии полного `scene.json`.

При расчёте камера запрашивает звёзды доV12 в поле с запасом.
Это ограничение запроса, не полнота каталога. В проверенной установке
получено 311 записей доV8;188 центров в поле. Дедупликация по ID/точному
ключу не заменяет межкаталожный cross-ID; три близкие пары перечислены
в [аудите](../artifacts/camera-model-20261112/catalogue-audit.json).

Результат глаза открываю отдельным Qt-окном; FITS камеры — штатным
FITS Viewer с проверяемым сигналом `loaded`. Параметры реального INDI-устройства
и Capture queue от этих действий не меняются.

## 3. Контракт снимка и задания

Snapshot schema1 содержит геометрические значения с явными соглашениями:

```json
{
  "schema_version": 1,
  "kstars_version": "3.6.2",
  "utc": "2026-11-11T21:00:00Z",
  "altitude_convention": "geometric_unrefracted",
  "location": {"latitude_deg": 60.3, "longitude_deg": 30.315, "elevation_m": 20},
  "focus": {"azimuth_deg": 180, "altitude_deg": 45},
  "artificial_horizon_visible": true
}
```

`POST /api/kstars/jobs` принимает `{scene,snapshot}`. Геометрия KStars
подменяет ручные место/дату/центр, сохраняя AGL и настройки модели;
ручная Луна исключается в пользу эфемерид. MSL места сохраняется как
происхождение, а модельная поверхность берётся из DSM, если он включён.
UTC каталога камеры обязан совпадать с UTC снимка.

Я отвергаю неверную схему, неявный пояс времени, NaN и подмену
геометрической высоты рефрагированной. Snapshot и его хеш сохраняются
с заданием. Параметры физического кэша остаются точными.

Автослежение по умолчанию выключено. Смена карты/параметров помечает
результат устаревшим; включённый follow использует стабильность и кванты
0,25°/60 с для запуска, но посылает полную фактическую геометрию.
Одновременно панель ведёт одно задание. Отмена проверяется между лучами.
HTTP-тайм-аут 15 с не ограничивает полную длительность расчёта.

Наш D-Bus `org.lightpollution.KStars`, путь `/LightPollution`, использует
session service `org.kde.kstars`. Методы: `snapshotJSON`, `setSceneJSON`,
`calculate`, `statusJSON`, `showEyeView`, `showCameraFITS`,
`savePanelScreenshot`, `saveWindowScreenshot`, `saveEyeScreenshot`,
`saveCameraScreenshot`. Внутренний `cameraCatalogue()` не является
отдельным публичным D-Bus методом.

## 4. Сборка

```bash
uv sync --locked --extra terrain --extra kstars
uv run python integrations/kstars/apply_native.py --source /path/to/kstars-source
cmake -S /path/to/kstars-source -B /path/to/kstars-build -DBUILD_TESTING=OFF
cmake --build /path/to/kstars-build --target kstars -j2
uv run lightpollution serve --host 127.0.0.1 --port 8765
LIGHTPOLLUTION_URL=http://127.0.0.1:8765 /path/to/kstars-build/bin/kstars
```

Требуются C++/Qt/KDE зависимости выбранной версии; для FITS Viewer — CFITSIO.
`apply_native.py` идемпотентен, проверяет структуру, копирует модуль и
добавляет панель/меню; он не устанавливает системные пакеты.
Исходник модуля имеет GPL-2.0-or-later, [COPYING](../integrations/kstars/COPYING).
Панель обращается только к loopback HTTP через `LIGHTPOLLUTION_URL`.

## 5. Проверенная версия и непроверенная ветвь

Проверена сборка Qt5/KStars3.6.2 с отключённым INDI/Ekos device control.
Исходный архив, зависимости и параметры закреплены в
[build provenance](../artifacts/kstars-terrain-near-20261111/native-build-provenance.json).
Аудит master и реально собранный релиз — разные источники, я их не смешиваю.

В 3.6.2 нет современного Exposure Calculator. При наличии его заголовка
и HAVE_INDI код предусматривает кнопку `ExposureCalculatorDialog` с
V-прокси/f-ratio/ID. Это не SQM и не прямая передача всей электронной карты.
Условная новая ветвь пока не проверена на Qt6; в тестовой сборке кнопка выключена.

Реальные 2/2 задания, FITS loaded и снимок Qt-окна:
[протокол](../artifacts/camera-model-20261112/native/native-verification.json).
Штатный CCD Simulator этой проверкой не запускался. Для повторения
использую `scripts/check_kstars_native.py --camera --job-timeout 900`
с обязательными путями binary/runtime/data-root/output и отдельной session D-Bus.

Готового пакета для обычной установки, upstream-включения и проверенного
управления реальной камерой пока нет. Остальные границы — [LIMITS](../LIMITS.md),
первичные исходники и документация — [LITERATURE](LITERATURE.md).
