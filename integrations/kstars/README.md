# LightPollution native dock
> Дата создания: 2026-11-11

0.8: экспериментальный цвет звёзд, открытие/сохранение переносимого
eye_view_profile через общее ядро. Новые проверки:
`scripts/check_kstars_native.py --eye --colour`. Нужна пересборка native dock;
контракт существующего snapshot schema 1 не менялся. Проверка на Qt5/KStars
3.6.2 не распространяется автоматически на Qt6, HiDPI или точность восприятия.

Версия 0.6 добавляет переключатель eye-порога и просмотр монохромного
результата внутри Qt. Проверка: `scripts/check_kstars_native.py --eye`;
все остальные обязательные аргументы запуска сохраняются. Это локальная
модификация исходников, не опубликованный upstream-плагин KDE.

Панель в главном окне KStars, не в браузере. Читает атомарный снимок места,
виртуального UTC, направления и штатного Artificial Horizon. Физическое ядро
работает отдельным локальным процессом; KStars остаётся отзывчивым.

```bash
uv run python integrations/kstars/apply_native.py --source /path/to/kstars-source
cmake -S /path/to/kstars-source -B /path/to/kstars-build -DBUILD_TESTING=OFF
cmake --build /path/to/kstars-build --target kstars -j2
uv run lightpollution serve --host 127.0.0.1 --port 8765
LIGHTPOLLUTION_URL=http://127.0.0.1:8765 /path/to/kstars-build/bin/kstars
```

`apply_native.py` идемпотентен, не скачивает код и не устанавливает пакеты.
Укажите отдельную исходную копию KStars; потребуются её Qt/KDE зависимости.
В штатный KDE-пакет модуль автоматически не устанавливается.

Полный контракт, используемые алгоритмы, ограничения и статус проверки:
[docs/KSTARS_INTEGRATION.md](../../docs/KSTARS_INTEGRATION.md).
Проверка живого приложения: `scripts/check_kstars_native.py` под
`dbus-run-session`, с отдельными XDG-каталогами; пользовательские настройки
не меняет. Отчёт и снимки настоящей панели:
[artifacts/kstars-terrain-near-20261111](../../artifacts/kstars-terrain-near-20261111/README.md).

Исходники C++: GPL-2.0-or-later, текст лицензии — `COPYING`.
