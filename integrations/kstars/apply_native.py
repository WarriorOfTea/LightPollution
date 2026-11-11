"""Apply the native dock to an explicitly selected KStars source checkout.

No network, binary installation or user settings changes. Idempotent.
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    root = source/'kstars'
    here = Path(__file__).resolve().parent
    cmake_path, init_path = root/'CMakeLists.txt', root/'kstarsinit.cpp'
    cmake, init = cmake_path.read_text(), init_path.read_text()
    marker = 'tools/lightpollutiondock.cpp'
    if marker not in cmake:
        anchor = '    tools/altvstime.cpp'
        if cmake.count(anchor) != 1:
            raise ValueError('Unsupported KStars CMake layout; no changes applied.')
        cmake = cmake.replace(anchor, anchor+'\n    '+marker)
    if '#include "tools/lightpollutiondock.h"' not in init:
        anchor = '#include "kstars.h"'
        if init.count(anchor) != 1 or 'setupGUI(StandardWindowOptions(Default), m_KStarsUIResource);' not in init:
            raise ValueError('Unsupported KStars UI layout; no changes applied.')
        init = init.replace(anchor, anchor+'\n#include "tools/lightpollutiondock.h"\n#include <QMenuBar>\n#include <QMenu>')
        anchor = '    setupGUI(StandardWindowOptions(Default), m_KStarsUIResource);'
        native = '''
    // LightPollution uses KStars' live geometry and its native local horizon.
    auto *lightPollutionDock = new LightPollutionDock(this);
    addDockWidget(Qt::RightDockWidgetArea, lightPollutionDock);
    lightPollutionDock->hide();
    auto *lightPollutionAction = lightPollutionDock->toggleViewAction();
    actionCollection()->addAction(QStringLiteral("lightpollution"), lightPollutionAction);
    auto *toolsMenu = findChild<QMenu *>(QStringLiteral("tools"));
    if (toolsMenu) toolsMenu->addAction(lightPollutionAction);
    else menuBar()->addMenu(QStringLiteral("LightPollution"))->addAction(lightPollutionAction);
    if (!qEnvironmentVariableIsEmpty("LIGHTPOLLUTION_URL")) lightPollutionDock->show();
'''
        init = init.replace(anchor, anchor+'\n'+native)
    if cmake_path.read_text() != cmake:
        cmake_path.write_text(cmake)
    if init_path.read_text() != init:
        init_path.write_text(init)
    for name in ['lightpollutiondock.cpp', 'lightpollutiondock.h']:
        target = root/'tools'/name
        content = (here/name).read_bytes()
        if not target.exists() or target.read_bytes() != content:
            target.write_bytes(content)
    print('Native LightPollution dock applied to', source)


if __name__ == '__main__':
    main()
