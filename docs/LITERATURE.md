# Первоисточники и их роль
> Дата создания: 2026-11-10

Я перечисляю основания, которые использую, и отделяю реализованную часть
от исследованного продолжения. Ссылка на статью не означает, что весь её
метод перенесён в программу. Формальные уравнения собраны в [MODELS](../MODELS.md),
начальные определения — в [THEORY](../THEORY.md).

## 1. Перенос, радиометрия и источники

- **Garstang1986**, [Model for Artificial Night-Sky Illumination](https://doi.org/10.1086/131768).
  Основание постановки городского свечения. Я не утверждаю побитовое
  воспроизведение всей авторской модели.
- **Cinzano & Falchi2012**, [The propagation of light pollution in the atmosphere](https://arxiv.org/abs/1209.2031).
  Геометрия, нормировки фазовых функций, различие первого и многократного
  рассеяния. Использую ограниченный скалярный первый порядок.
- **Bruneton**, [исходные параметры атмосферного примера](https://github.com/ebruneton/precomputed_atmospheric_scattering/blob/master/atmosphere/demo/demo.cc).
  Основа части исходных коэффициентов, не измерение Петербурга;
  AOD0,1 — мой отдельный сценарий.
- **Bará et al.2020**, [Magnitude to luminance conversions and visual brightness of the night sky](https://arxiv.org/abs/2002.01494).
  Спектральная зависимость перевода астрономической шкалы в зрительную яркость.
  Не даёт универсального коэффициента V/SQM/кд·м⁻².
- **NASA**, [Black Marble User Guide Collection2](https://viirsland.gsfc.nasa.gov/PDF/BlackMarbleUserGuide_Collection2.0.pdf).
  Смысл композитов, QA и временная поддержка. DNB не инвертируется
  автоматически в мощность/спектр всех фонарей.
- **NASA**, [редакция руководства October2024](https://landweb.modaps.eosdis.nasa.gov/data/userguide/BlackMarbleUserGuide_Collection2.0_20241203.pdf),
  §§2.2–2.3, и [карточка VNP46A4.002](https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/products/VNP46A4).
  Повторно проверено различие TOA и скорректированных композитов; оно
  не заменяет мою неопределённую DNB→мощность/спектр/углы инверсию.
- **AERONET V3**, [официальные продукты инверсии](https://aeronet.gsfc.nasa.gov/print_web_data_help_v3_inv.html).
  AOD/SIZ/RIN/SSA/ASY/PFN доступны как кандидаты; конкретные ночные профили
  не подключены. **NOAA GML**, [определения аэрозольных оптических параметров](https://gml.noaa.gov/grad/about/aero_term.html):
  различие AOD/SSA/асимметрии и связь с составом/размерами.
- **Bruneton**, [рекурсия повышенных порядков](https://ebruneton.github.io/precomputed_atmospheric_scattering/atmosphere/functions.glsl).
  Иллюстрирует отдельные источники рассеяния и интегралы по порядкам;
  алгоритм для солнечной атмосферы не перенесён автоматически на наши города.
- **Emde et al.2016**, [libRadtran2.0.1](https://gmd.copernicus.org/articles/9/1647/2016/).
  Изучено различие дискретных и Monte Carlo решателей. Полная 3D-версия
  MYSTIC не входит в публичную поставку описанной версии; не объявляю её
  готовым открытым модулем проекта.
- **Copernicus**, [DSM GLO-30/GLO-90 и права](https://registry.opendata.aws/copernicus-dem/).
  Поверхность, геопривязка/датум. Полученные тайлы закреплены манифестами.

## 2. Луна, естественная ночь, сумерки и снег

- **Krisciunas & Schaefer1991**, [A Model of the Brightness of Moonlight](https://articles.adsabs.harvard.edu/pdf/1991PASP..103.1033K),
  [DOI](https://doi.org/10.1086/132921). Использую фазовую/угловую V-модель;
  расстояние корректирую отдельно по обратному квадрату. Не полный спектральный перенос.
- **Noll et al.2012**, [A&A543 A92](https://arxiv.org/pdf/1205.2003), §4.1 eq.21.
  Использую геометрию van Rhijn; полное атмосферное рассеяние airglow
  из статьи не реализую одним прямым пропусканием.
- **Patat 2003**, [A&A400,1183](https://arxiv.org/pdf/astro-ph/0301115), приложениеC.
  Ночной фон/угловые приближения и опора Paranal 21,6V.
- **Patat, Ugolnikov & Postylyakov2006**, [полный текст](https://arxiv.org/html/astro-ph/0604128), Table1.
  Переношу коэффициенты полной околозенитной V-кривой и её диапазон.
  Публикуемый RMS0,18mag не объявляю ошибкой применения в Петербурге.
- **Jechow & Hölker2019**, [Snowglow](https://www.mdpi.com/2313-433X/5/8/69).
  Основание существенности снега и осторожности сравнения состояний.
  Множитель×33 относится к облачным случаям, не калибрует мой ясный сценарий.
- **MODIS/SNICAR**, [MOD10A1.061](https://nsidc.org/data/mod10a1/versions/61),
  [руководство MCD43A3](https://www.umb.edu/spectralmass/modis-user-guide-v006-and-v0061/mcd43a3-albedo-product/),
  [SNICAR-ADv3](https://gmd.copernicus.org/articles/14/7673/2021/).
  Изучены для следующих входов. NDSI и солнечное broadband albedo
  не объявляю готовыми долями освещённого снега/ламповым альбедо.

## 3. Глаз, оптика и отображение

- **Crumey 2014**, [Human Contrast Threshold and Astronomical Visibility](https://arxiv.org/abs/1405.4209).
  Использую точечную скотопическую ветвь на основе Blackwell, спектральное
  приведение и нижнее ограничение фона. Мой Feye исключает атмосферу,
  уже учтённую в потоке. Порог не равен вероятности или закону яркости PNG.
- **Watson 2013**, [DOI10.1167/13.6.18](https://doi.org/10.1167/13.6.18), eq.1–5.
  Средняя MTF при 2…6 мм. Положительная Gaussian-аппроксимация — моя
  численная реализация; это не измеренная личная PSF.
- **Ferwerda et al.1996**, [A Model of Visual Adaptation for Realistic Image Synthesis](https://www.cs.ucf.edu/~sumant/publications/sig96.pdf), eq.3–5.
  Использую TVI-масштаб. Мягкий shoulder и цветовая заслонка в проекте
  собственные; не приписываю их статье как измеренный порог цвета звёзд.
- **Ward Larson et al.1997**, [полный отчёт LBNL](https://radsite.lbl.gov/radiance/papers/lbnl39882/tonemap.pdf), §§4.2–4.5.
  Изучено и реализовано отдельное сравнительное гистограммное ядро,
  но не полный pcond/glare/acuity/colour. Основной вывод остаётся фиксированным.
- **Kirk & O’Brien2011**, [авторская страница](https://jamesobrien.com/papers/Kirk-PBT-2011-08/index.html),
  [DOI](https://doi.org/10.1145/2010324.1964937), eq.10–13.
  Алгебра отдельно проверена в paper-normalized q. Мост от физических
  CIE LMS и весь display/HDR-pipeline пока отсутствуют.
- **CIE**, [XYZ1931](https://doi.org/10.25039/CIE.DS.xvudnb9b),
  [LMS2006](https://doi.org/10.25039/CIE.DS.tijidesg),
  [D65](https://doi.org/10.25039/CIE.DS.hjfjmt59),
  [фотопическая функция](https://www.cie.co.at/datatable/cie-spectral-luminous-efficiency-photopic-vision),
  [скотопическая функция](https://www.cie.co.at/datatable/cie-spectral-luminous-efficiency-scotopic-vision).
  Использую открытые энергетические таблицы с собственной атрибуцией;
  стандартный 2° наблюдатель не является доказанной точечной периферической моделью.
- **W3C**, [CSS Color4, conversion code](https://www.w3.org/TR/css-color-4/#color-conversion-code).
  Матрицы XYZ/sRGB/D65. Моё сохранение Y при gamut-сжатии отдельно от стандарта.
- **Fritsch & Butland1984**, [DOI10.1137/0905021](https://doi.org/10.1137/0905021).
  Основание формы PCHIP; доступны библиография/аннотация, закрытый полный
  текст не объявлен прочитанным. Собственные формулы и контроль заданы в коде.

## 4. Камера, KStars и независимая проверка

- **Hy Murveit, KStars GreatCircle**,
  [закреплённая реализация сферической навигации](https://github.com/KDE/kstars/blob/c46dbb4f0cfe4b8ca190a0c391e975e52a005aa8/kstars/tools/greatcircle.cpp)
  и [правила ArtificialHorizon](https://github.com/KDE/kstars/blob/c46dbb4f0cfe4b8ca190a0c391e975e52a005aa8/kstars/skycomponents/artificialhorizoncomponent.cpp).
  Использую геометрию большого круга, ограничение вершины89,999° и
  ближайшие floor/ceiling; тип сегмента явно входит в обмен.
  Реализация KStars имеет GPL-2.0-or-later. Независимые контрольные числа —
  [его greatCircleTest](https://github.com/KDE/kstars/blob/c46dbb4f0cfe4b8ca190a0c391e975e52a005aa8/Tests/tools/testartificialhorizon.cpp).

- **EMVA1288 Linear4.0**, [официальный стандарт](https://www.emva.org/wp-content/uploads/EMVA1288Linear_4.0Release.pdf), §§2.1–2.4.
  Прочитан полный текст; использую фотонно-электронную линейную модель,
  Poisson и RN. Моя g=e⁻/ADU обратна K=DN/e⁻. Сертификацию EMVA не заявляю.
- **IAU/NASA**, [FITS Standard4.0](https://fits.gsfc.nasa.gov/fits_standard.html).
  Основание формата, времени и WCS; TAN-геометрия не равна plate solve.
- **KStars/Ekos**, [руководство Capture](https://kstars-docs.kde.org/en/user_manual/ekos-capture.html),
  [закреплённый калькулятор](https://github.com/KDE/kstars/blob/7aea013b99165e80d1426d183de4a31ae22b0ca5/kstars/ekos/capture/exposurecalculator/optimalsubexposurecalculator.cpp).
  Повторно использую критерий RN и формат существующих камер.
- **INDI**, [закреплённый sky_renderer](https://github.com/indilib/indi/blob/76b457edca2d3ab7f30527f9b93b53ed58c2cb2c/drivers/ccd/sky_renderer.cpp).
  Исследован существующий sky-glow/шум. Драйвер этой интеграцией не переписан.
- **camera-data**, [закреплённая ревизия](https://github.com/indilib/camera-data/tree/44984a2e9303350ba34521681d2c162a8131dba7).
  Формат partial XML pitch/RN, а не полная измеренная QE/g/dark-модель.
- **Jechow, Kyba & Hölker**, [Mapping the brightness and color of urban to rural skyglow](https://arxiv.org/abs/2202.08258).
  Методика пространственной калиброванной проверки; полного согласованного
  RAW-пакета для нашего случая не получено.
- **NPS**, [Methods](https://www.nps.gov/subjects/nightskies/methods.htm).
  Основание строгой съёмки/калибровки all-sky вместо сравнения случайных JPEG.

## 5. Аудит фактического доступа

Хеши скачанных файлов, ревизии, HTTP-отказы и прочитанные области сохранены
в исследованиях: [первый обзор](../artifacts/research-20261110/source-audit.json),
[фон/снег](../artifacts/natural-twilight-snow-20261111/literature-audit.json),
[визуализация](../artifacts/eye-visualization-research-20261111/source-audit.json),
[цвет](../artifacts/eye-colour-20261111/source-audit.json),
[камера](../artifacts/camera-model-20261112/source-audit.json).
Это журнал прежнего исследования; редакция документации не означает
повторную сетевую проверку всех ссылок или новые измерения.

## 6. Каталоги и нативная поставка

- **Høg et al., Tycho-2 Guide**, [руководство ESO](https://archive.eso.org/ASTROM/TYC-2/docs/guide.pdf):
  собственные движения, эпохи, различие Bₜ/Vₜ и Johnson; приближения5/6
  относятся к незапылённой главной последовательности.
- **CDS I/259**, [каталог](https://cdsarc.cds.unistra.fr/viz-bin/cat/I/259):
  точечная проверка исходных BTmag/VTmag/PM. Сохранённые ответы и отказ БД —
  [source-audit](../artifacts/kstars-catalogue-integration-20261112/source-audit.json).
- **KStars**, [закреплённый repair-инструмент](https://github.com/KDE/kstars/blob/c46dbb4f0cfe4b8ca190a0c391e975e52a005aa8/kstars/data/tools/fix-star-catalogs.py):
  критерии duplicate/ghost, разделение эпох и сохранение кратных компонент.
- **CMake3.30**, [предкомпилированные заголовки](https://cmake.org/cmake/help/v3.30/command/target_precompile_headers.html):
  приватное ускорение сборки, условие CXX и экранирование ANGLE-R.
