# Спектральная библиотека
> Дата создания: 2026-11-11

`library.npz` содержит обработанные численные таблицы, `bsc-colours.csv` —
дополнительные B−V и спектральные классы BSC5. Источники, байтовые хеши,
преобразования и атрибуция — `provenance.json`. Сырые скачивания находятся
в игнорируемом `scratch/`; воспроизведение: `scripts/prepare_spectral_library.py`.

Массивы, полученные из таблиц CIE, распространяются на условиях
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
Атрибуция: CIE, 2018/2019, DOI в provenance; изменения — линейная интерполяция
на сетку 360–830 нм с шагом 5 нм, ноль вне табличного диапазона.
Лицензия этих данных не является объявлением лицензии собственного кода.

Профили Johnson B/V: SVO Filter Profile Service «Carlos Rodrigo»,
Rodrigo et al. 2024, A&A 689 A93; указанный сервисом тип — **energy counter**.
Его отклик интегрируется без дополнительного множителя λ.
Благодарность: This research has made use of the SVO Filter Profile Service
"Carlos Rodrigo", funded by MCIN/AEI/10.13039/501100011033/ through grant
PID2023-146210NB-I00.

Спектры звёзд: STScI CALSPEC (Bohlin et al. 2022); ESA Gaia DR3,
обработка DPAC (Gaia Collaboration et al. 2023; Montegriffo et al. 2023).
Цветовые показатели: NASA HEASARC BSC5, Hoffleit & Warren 1991.
Это публичные численные научные данные, не заимствованный программный код.

Шаблоны ламп не являются измеренным составом фонарей города. Полный отклик
VIIRS DNB в ближнем ИК не восстановлен. По B−V строится приближённый
чернотельный континуум, а не измеренная температура или спектральные линии.
