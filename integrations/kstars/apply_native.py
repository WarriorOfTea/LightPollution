"""Apply the native dock to an explicitly selected KStars source checkout.

No network, binary installation or user settings changes. Idempotent.
"""

import argparse
from pathlib import Path


def prepare(source):
    """Validate every anchor before writing any file in the selected source."""
    source=Path(source).resolve()
    updates={}
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
    if '# LightPollution: optional common Qt precompiled headers.' not in cmake:
        cmake+='''
# LightPollution: optional common Qt precompiled headers.
option(LIGHTPOLLUTION_PRECOMPILE_QT "Reuse common Qt headers in the private native build" ON)
if(TARGET KStarsLib AND LIGHTPOLLUTION_PRECOMPILE_QT)
    target_precompile_headers(KStarsLib PRIVATE
        "$<$<COMPILE_LANGUAGE:CXX>:<QtCore$<ANGLE-R>>"
        "$<$<COMPILE_LANGUAGE:CXX>:<QtGui$<ANGLE-R>>"
        "$<$<COMPILE_LANGUAGE:CXX>:<QtWidgets$<ANGLE-R>>")
endif()
'''
    for header in ['QtCore','QtGui','QtWidgets']:
        cmake=cmake.replace(':<'+header+'>>',':<'+header+'$<ANGLE-R>>')
    spherical=False
    horizon_path=root/'skycomponents/artificialhorizoncomponent.cpp'
    if horizon_path.is_file():
        horizon=horizon_path.read_text()
        start=horizon.find('double ArtificialHorizonEntity::altitudeConstraint(')
        end=horizon.find('ArtificialHorizonComponent::ArtificialHorizonComponent',start)
        if start<0 or end<0:raise ValueError('Unsupported artificial horizon API; no changes applied.')
        spherical='GreatCircle gc(' in horizon[start:end]
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
    updates[cmake_path]=cmake
    updates[init_path]=init
    for name in ['lightpollutiondock.cpp', 'lightpollutiondock.h']:
        updates[root/'tools'/name]=(here/name).read_text()
    if spherical:
        path=root/'tools/lightpollutiondock.cpp'
        updates[path]='#define LIGHTPOLLUTION_HORIZON_GREAT_CIRCLE 1\n'+updates[path]
    star_path=root/'skyobjects/starobject.cpp'
    star=star_path.read_text()
    marker='// LightPollution: retain the catalogue colour; encoded zero is ambiguous.'
    if marker not in star:
        anchor='    B = V = 99.9;'
        if star.count(anchor)!=1:
            raise ValueError('Unsupported StarData colour layout; no changes applied.')
        star=star.replace(anchor,anchor+"""
    // LightPollution: retain the catalogue colour; encoded zero is ambiguous.
    // StarData bv_index scale=100 (mysql2bin.c), NOT the deep B/V scale=1000.
    if (stardata->bv_index != 0 && stardata->bv_index >= -100 && stardata->bv_index <= 400)
    {
        V = mag();
        B = V + stardata->bv_index / 100.0;
    }
""")
    updates[star_path]=star
    deep_path=root/'skycomponents/deepstarcomponent.cpp'
    if deep_path.exists():
        deep=deep_path.read_text()
        marker='    // LightPollution: pin existing dynamic blocks before aperture loading.'
        if marker not in deep:
            function='bool DeepStarComponent::starsInAperture('
            if deep.count(function)!=1:raise ValueError('Unsupported deep catalogue aperture API; no changes applied.')
            start=deep.index(function)
            anchor='    while (region.hasNext())'
            position=deep.find(anchor,start)
            if position<0:raise ValueError('Unsupported deep catalogue iterator; no changes applied.')
            protection='''    // LightPollution: pin existing dynamic blocks before aperture loading.
    // Otherwise fillToMag can recycle a previously appended StarObject pointer.
    if (!staticStars)
    {
        auto *factory = StarBlockFactory::Instance();
        if (factory->drawID == 0) factory->drawID = 1;
        MeshIterator protecting(m_skyMesh, OBJ_NEAREST_BUF);
        while (protecting.hasNext())
        {
            const auto blocks = m_starBlockList[protecting.next()];
            if (!blocks) continue;
            for (int i = 0; i < blocks->getBlockCount(); ++i)
            {
                auto block = blocks->block(i);
                // Preserve the bright-to-faint chain: only its faint end may
                // be recycled later by StarBlockList::releaseBlock().
                if (i == 0) factory->markFirst(block);
                else
                {
                    auto previous = blocks->block(i - 1);
                    factory->markNext(previous, block);
                }
            }
        }
    }

'''
            deep=deep[:position]+protection+deep[position:]
        # Upgrade an earlier private patch which pinned every block at the head.
        # That reversed each trixel and broke the factory's release invariant.
        old_pin='''                auto block = blocks->block(i);
                factory->markFirst(block);'''
        if old_pin in deep:
            deep=deep.replace(old_pin,'''                auto block = blocks->block(i);
                // Preserve the bright-to-faint chain: only its faint end may
                // be recycled later by StarBlockList::releaseBlock().
                if (i == 0) factory->markFirst(block);
                else
                {
                    auto previous = blocks->block(i - 1);
                    factory->markNext(previous, block);
                }''',1)
        updates[deep_path]=deep
    ekos=root/'ekos/capture/exposurecalculator'
    header=ekos/'optimalsubexposurecalculator.h'
    if header.is_file():
        h=header.read_text();cpath=ekos/'optimalsubexposurecalculator.cpp';c=cpath.read_text()
        if 'setExternalSkyElectronRate' not in h:
            anchor='        CameraExposureEnvelope calculateCameraExposureEnvelope();'
            if h.count(anchor)!=1:raise ValueError('Unsupported Ekos declaration layout; no changes applied.')
            h=h.replace(anchor,anchor+"""
        // LightPollution: detector/optics already included in this pixel rate.
        bool setExternalSkyElectronRate(double rate)
        {
            if (!(rate > 0) || rate > 1e20) return false;
            m_ExternalSkyElectronRate = rate;
            return true;
        }
""")
            anchor='        double calculateCFactor(double noiseTolerance);'
            if h.count(anchor)!=1:raise ValueError('Unsupported Ekos private layout; no changes applied.')
            h=h.replace(anchor,'        double m_ExternalSkyElectronRate { -1.0 };\n'+anchor)
        marker='    // LightPollution: never reapply focal ratio, filter width or QE to electron rates.'
        if marker not in c:
            anchor='\ndouble OptimalSubExposureCalculator::calculateLightPolutionForOpticFocalRatio('
            if c.count(anchor)!=1:raise ValueError('Unsupported Ekos electron rate layout; no changes applied.')
            start=c.index('{',c.index(anchor))+1
            c=c[:start]+'\n'+marker+'\n    if (m_ExternalSkyElectronRate > 0) return m_ExternalSkyElectronRate;\n'+c[start:]
        updates[header]=h;updates[cpath]=c
    return updates


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--check',action='store_true',help='Validate and list changes without writing.')
    args=parser.parse_args()
    updates=prepare(args.source)
    changes={p:s for p,s in updates.items() if not p.exists() or p.read_text()!=s}
    if not args.check:
        for path,content in changes.items():
            temporary=path.with_suffix(path.suffix+'.lightpollution-tmp')
            temporary.write_text(content);temporary.replace(path)
    print(('Validated' if args.check else 'Applied'),len(changes),'native changes in',args.source.resolve())


if __name__ == '__main__':
    main()
