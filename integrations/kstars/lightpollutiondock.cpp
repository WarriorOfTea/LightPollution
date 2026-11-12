// SPDX-License-Identifier: GPL-2.0-or-later
#include "lightpollutiondock.h"
#include "kstars.h"
#include "kstarsdata.h"
#include "skymap.h"
#include "geolocation.h"
#include "skyobjects/skyobject.h"
#include "skyobjects/starobject.h"
#include "skycomponents/skymapcomposite.h"
#include "skycomponents/artificialhorizoncomponent.h"
#include "skycomponents/linelist.h"
#include "auxiliary/kspaths.h"
#include "skycomponents/starcomponent.h"
#include "version.h"
#ifdef HAVE_CFITSIO
#include "fitsviewer/fitsviewer.h"
#endif
#if __has_include("ekos/capture/exposurecalculator/exposurecalculatordialog.h") && defined(HAVE_INDI) && HAVE_INDI
#include "ekos/capture/exposurecalculator/exposurecalculatordialog.h"
#define LP_HAVE_EKOS_EXPOSURE 1
#endif

#include <QCheckBox>
#include <QComboBox>
#include <QCryptographicHash>
#include <QDBusConnection>
#include <QDoubleSpinBox>
#include <QDialog>
#include <QScrollArea>
#include <QPixmap>
#include <QFile>
#include <QFileDialog>
#include <QFormLayout>
#include <QGroupBox>
#include <QInputDialog>
#include <QHeaderView>
#include <QJsonArray>
#include <QJsonDocument>
#include <QLabel>
#include <QNetworkReply>
#include <QPushButton>
#include <QTableWidget>
#include <QUrl>
#include <QSet>
#include <QStandardPaths>
#include <QVBoxLayout>
#include <cmath>
#include <algorithm>

#ifdef HAVE_CFITSIO
namespace {
[[maybe_unused]] FITSViewer *viewerPointer(FITSViewer *viewer) { return viewer; }
[[maybe_unused]] FITSViewer *viewerPointer(const QSharedPointer<FITSViewer> &viewer) { return viewer.data(); }
}
#endif

LightPollutionDock::LightPollutionDock(KStars *parent) : QDockWidget("LightPollution", parent), m_KStars(parent)
{
    setObjectName("LightPollutionDock");
    setMinimumWidth(360);
    auto *body=new QWidget(this);
    auto *layout=new QVBoxLayout(body);
    m_Position=new QLabel(body); m_Position->setWordWrap(true); layout->addWidget(m_Position);
    auto *form=new QFormLayout;
    m_Region=new QComboBox(body); m_Period=new QComboBox(body); m_Terrain=new QComboBox(body);
    m_Terrain->addItem(tr("Без высот"), QString());
    m_Terrain->addItem(tr("Петербург · Copernicus DSM"), QStringLiteral("petersburg-copdem90"));
    m_Terrain->addItem(tr("Мадрид · Copernicus DSM"), QStringLiteral("madrid-copdem90"));
    m_Terrain->addItem(tr("Тест: плоская поверхность"), QStringLiteral("demo-flat"));
    m_Height=new QDoubleSpinBox(body);m_Height->setRange(.5,100);m_Height->setDecimals(1);m_Height->setValue(2);
    m_AOD=new QDoubleSpinBox(body);m_AOD->setRange(0,2);m_AOD->setSingleStep(.05);m_AOD->setValue(.1);
    m_Near=new QCheckBox(tr("Конечные ближние ячейки"),body);m_Near->setChecked(true);
    m_Spectral=new QCheckBox(tr("Спектральный сценарий"),body);
    m_Eye=new QCheckBox(tr("Порог невооружённого глаза"),body);
    m_Colour=new QCheckBox(tr("Экспериментальный цвет звёзд"),body);
    m_OpenEye=new QPushButton(tr("Открыть модель неба"),body);m_OpenEye->setEnabled(false);
    m_Follow=new QCheckBox(tr("Следить за картой и временем"),body);
    form->addRow(tr("Источники"),m_Region);form->addRow(tr("Период"),m_Period);
    form->addRow(tr("Рельеф"),m_Terrain);form->addRow(tr("Высота над DSM, м"),m_Height);
    form->addRow(tr("AOD 550 нм"),m_AOD);layout->addLayout(form);
    layout->addWidget(m_Near);layout->addWidget(m_Spectral);layout->addWidget(m_Eye);
    layout->addWidget(m_Colour);layout->addWidget(m_OpenEye);layout->addWidget(m_Follow);
    auto *buttons=new QHBoxLayout;
    auto *run=new QPushButton(tr("Рассчитать"),body);auto *cancel=new QPushButton(tr("Отменить"),body);
    buttons->addWidget(run);buttons->addWidget(cancel);layout->addLayout(buttons);
    auto *files=new QHBoxLayout;
    auto *load=new QPushButton(tr("Открыть сценарий"),body);auto *save=new QPushButton(tr("Сохранить результат"),body);
    files->addWidget(load);files->addWidget(save);layout->addLayout(files);
    auto *profiles=new QHBoxLayout;
    auto *loadProfile=new QPushButton(tr("Открыть профиль глаза"),body);
    auto *saveProfile=new QPushButton(tr("Сохранить профиль глаза"),body);
    profiles->addWidget(loadProfile);profiles->addWidget(saveProfile);layout->addLayout(profiles);
    m_Camera=new QCheckBox(tr("Линейная камера / FITS"),body);layout->addWidget(m_Camera);
    auto *cameraGroup=new QGroupBox(tr("Камера: сценарный профиль"),body);auto *cameraForm=new QFormLayout(cameraGroup);
    m_Exposure=new QDoubleSpinBox(cameraGroup);m_Exposure->setDecimals(3);m_Exposure->setRange(.001,120);m_Exposure->setValue(30);
    m_Focal=new QDoubleSpinBox(cameraGroup);m_Focal->setRange(8,5000);m_Focal->setValue(16);
    m_Aperture=new QDoubleSpinBox(cameraGroup);m_Aperture->setRange(1,2000);m_Aperture->setValue(8);
    cameraForm->addRow(tr("Выдержка, с"),m_Exposure);cameraForm->addRow(tr("Фокусное расстояние, мм"),m_Focal);cameraForm->addRow(tr("Апертура, мм"),m_Aperture);
    auto *cameraLoad=new QPushButton(tr("Открыть JSON-профиль"),cameraGroup);
    auto *cameraSave=new QPushButton(tr("Сохранить JSON-профиль"),cameraGroup);
    auto *cameraXML=new QPushButton(tr("Импорт XML камеры Ekos"),cameraGroup);
    cameraForm->addRow(cameraLoad,cameraSave);cameraForm->addRow(cameraXML);
    layout->addWidget(cameraGroup);cameraGroup->setVisible(false);
    m_OpenCamera=new QPushButton(tr("Открыть кадр в FITS Viewer KStars"),body);m_OpenCamera->setEnabled(false);layout->addWidget(m_OpenCamera);
    m_EkosExposure=new QPushButton(tr("Экспозиция Ekos: поток электронов"),body);m_EkosExposure->setEnabled(false);layout->addWidget(m_EkosExposure);
#ifndef LP_HAVE_EKOS_EXPOSURE
    m_EkosExposure->setToolTip(tr("В этой версии/сборке KStars калькулятора нет. Скорости электронов и границы экспозиции сохраняются в camera-ekos.json."));
#endif
    m_Status=new QLabel(tr("Подключение к локальному расчётному ядру…"),body);m_Status->setWordWrap(true);layout->addWidget(m_Status);
    m_Values=new QTableWidget(0,2,body);m_Values->setHorizontalHeaderLabels({tr("Компонент"),tr("Значение")});
    m_Values->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    m_Values->verticalHeader()->hide();m_Values->setEditTriggers(QAbstractItemView::NoEditTriggers);layout->addWidget(m_Values);
    m_Scale=new QLabel(tr("Сценарный масштаб. Порог глаза — ограниченный режим полной адаптации; цвет и сенсор не калиброваны."),body);
    m_Scale->setWordWrap(true);layout->addWidget(m_Scale);
    auto *scroll=new QScrollArea(this);scroll->setWidgetResizable(true);scroll->setWidget(body);setWidget(scroll);
    m_URL=qEnvironmentVariable("LIGHTPOLLUTION_URL","http://127.0.0.1:8765");
    const QUrl url(m_URL);
    if(url.scheme()!="http" || !(url.host()=="127.0.0.1" || url.host()=="localhost" || url.host()=="::1"))
        m_URL="http://127.0.0.1:8765";
    connect(run,&QPushButton::clicked,this,&LightPollutionDock::calculate);
    connect(m_Eye,&QCheckBox::toggled,this,[this](bool enabled){if(enabled)m_Spectral->setChecked(true);});
    connect(m_Colour,&QCheckBox::toggled,this,[this](bool enabled){if(enabled){m_Eye->setChecked(true);m_Spectral->setChecked(true);}});
    connect(m_Camera,&QCheckBox::toggled,this,[this,cameraGroup](bool enabled){cameraGroup->setVisible(enabled);if(enabled)m_Spectral->setChecked(true);});
    connect(m_OpenCamera,&QPushButton::clicked,this,&LightPollutionDock::showCameraFITS);
    connect(m_EkosExposure,&QPushButton::clicked,this,&LightPollutionDock::showEkosExposure);
    auto applyCameraProfile=[this](const QJsonObject &document){
        request("/api/camera-profile/merge",{{"camera",scene().value("camera").toObject()},{"profile",document}},
            [this](const QJsonObject &camera,const QString &error){
                if(!error.isEmpty()){setStatus(error);return;}
                m_Base.insert("camera",camera);m_Camera->setChecked(true);
                const auto optics=camera.value("profile").toObject().value("optics").toObject();
                m_Focal->setValue(optics.value("focal_length_mm").toDouble());m_Aperture->setValue(optics.value("aperture_diameter_mm").toDouble());
                setStatus(tr("Профиль камеры загружен. Реальная QE-кривая, gain e-/ADU и шумы требуют измерений; конфигурация INDI не менялась."));});};
    connect(cameraLoad,&QPushButton::clicked,this,[this,applyCameraProfile](){
        QFile file(QFileDialog::getOpenFileName(this,tr("Профиль камеры"),{},"JSON (*.json)"));
        if(file.open(QIODevice::ReadOnly))applyCameraProfile(QJsonDocument::fromJson(file.readAll()).object());});
    connect(cameraSave,&QPushButton::clicked,this,[this](){
        const QString path=QFileDialog::getSaveFileName(this,tr("Профиль камеры"),"camera-profile.json","JSON (*.json)");if(path.isEmpty())return;
        request("/api/camera-profile/export",{{"camera",scene().value("camera").toObject()}},[this,path](const QJsonObject &document,const QString &error){
            if(!error.isEmpty()){setStatus(error);return;}QFile file(path);if(file.open(QIODevice::WriteOnly))file.write(QJsonDocument(document).toJson());});});
    connect(cameraXML,&QPushButton::clicked,this,[this,applyCameraProfile](){
        QFile file(QFileDialog::getOpenFileName(this,tr("Штатный профиль Ekos"),{},"XML (*.xml)"));if(!file.open(QIODevice::ReadOnly))return;
        bool ok=false;const int gain=QInputDialog::getInt(this,tr("Индекс gain/ISO в таблице Ekos"),tr("Это индекс режима, не e-/ADU"),0,-100,100000,1,&ok);if(!ok)return;
        const auto base=scene().value("camera").toObject().value("profile").toObject();
        request("/api/camera-profile/ekos",{{"xml",QString::fromUtf8(file.readAll())},{"base_profile",base},{"gain",gain},{"read_mode",0}},
            [this,applyCameraProfile](const QJsonObject &document,const QString &error){if(!error.isEmpty()){setStatus(error);return;}applyCameraProfile(document);});});
    connect(loadProfile,&QPushButton::clicked,this,[this](){
        const QString path=QFileDialog::getOpenFileName(this,tr("Профиль глаза и экрана"),{},"JSON (*.json)");
        QFile file(path);if(!file.open(QIODevice::ReadOnly))return;
        const auto profile=QJsonDocument::fromJson(file.readAll()).object();
        request("/api/eye-profile/merge",{{"eye",scene().value("eye").toObject()},{"profile",profile}},
            [this](const QJsonObject &eye,const QString &error){
                if(!error.isEmpty()){setStatus(error);return;}
                m_Base.insert("eye",eye);m_Eye->setChecked(true);
                m_Colour->setChecked(eye.value("appearance").toObject().value("mode").toString()=="experimental_stellar_colour");
                setStatus(tr("Профиль загружен. Место, UTC и геометрия карты сохранены; точность восприятия не подтверждена."));});});
    connect(saveProfile,&QPushButton::clicked,this,[this](){
        const QString path=QFileDialog::getSaveFileName(this,tr("Профиль глаза и экрана"),"eye-profile.json","JSON (*.json)");
        if(path.isEmpty())return;
        request("/api/eye-profile/export",{{"eye",scene().value("eye").toObject()}},
            [this,path](const QJsonObject &profile,const QString &error){
                if(!error.isEmpty()){setStatus(error);return;}
                QFile file(path);if(file.open(QIODevice::WriteOnly))file.write(QJsonDocument(profile).toJson());});});
    connect(m_OpenEye,&QPushButton::clicked,this,&LightPollutionDock::showEyeView);
    connect(cancel,&QPushButton::clicked,this,[this](){m_Follow->setChecked(false);if(!m_Job.isEmpty())request("/api/jobs/"+m_Job+"/cancel",{},[](const QJsonObject &,const QString &){});});
    connect(load,&QPushButton::clicked,this,[this](){const QString path=QFileDialog::getOpenFileName(this,tr("Сценарий"),{},"JSON (*.json)");QFile file(path);if(file.open(QIODevice::ReadOnly))setSceneJSON(QString::fromUtf8(file.readAll()));});
    connect(save,&QPushButton::clicked,this,[this](){if(m_LastResult.isEmpty())return;const QString path=QFileDialog::getSaveFileName(this,tr("Результат"),"lightpollution-result.json","JSON (*.json)");QFile file(path);if(file.open(QIODevice::WriteOnly))file.write(QJsonDocument(m_LastResult).toJson());});
    connect(m_Region,QOverload<int>::of(&QComboBox::currentIndexChanged),this,[this](){m_Period->clear();for(const auto &value:m_Regions.value(m_Region->currentData().toString()).toArray())m_Period->addItem(value.toString());});
    connect(&m_Timer,&QTimer::timeout,this,&LightPollutionDock::tick);m_Timer.start(1000);
    QDBusConnection::sessionBus().registerObject("/LightPollution",this,QDBusConnection::ExportScriptableSlots);
    QTimer::singleShot(0,this,&LightPollutionDock::fetchCatalogue);
}

QJsonObject LightPollutionDock::snapshot() const
{
    auto *data=m_KStars->data();auto *geo=data->geo();auto *point=m_KStars->map()->focus();
    QJsonObject focus{{"azimuth_deg",point->az().Degrees()},{"altitude_deg",point->alt().Degrees()},
        {"ra_jnow_deg",point->ra().Degrees()},{"dec_jnow_deg",point->dec().Degrees()},
        {"fov_deg",m_KStars->map()->fov()},{"object_name",m_KStars->map()->focusObject()?m_KStars->map()->focusObject()->name():QString()}};
    if(m_KStars->map()->focusObject())focus.insert("catalogue_magnitude",m_KStars->map()->focusObject()->mag());
    if(const auto *star=dynamic_cast<const StarObject *>(m_KStars->map()->focusObject()))
    {
        focus.insert("star_spectral_type",star->sptype());
        focus.insert("star_hd",star->getHDIndex());
        const double bv=star->getBVIndex();
        // KStars uses 99.9 as unavailable B-V, not as a physical colour.
        focus.insert("star_b_v",std::isfinite(bv) && bv!=99.9 && bv<30?QJsonValue(bv):QJsonValue());
    }
    auto *horizon=data->skyComposite()->artificialHorizon();
    const bool visible=!horizon || horizon->getHorizon().isVisible(point->az().Degrees(),point->alt().Degrees());
    QJsonArray regions;
    if(horizon)for(const auto *region:*horizon->getHorizon().horizonList()){
        if(!region->enabled() || !region->list() || region->list()->points()->size()<2)continue;
        QJsonArray points;for(const auto &p:*region->list()->points())points.append(QJsonArray{p->az().Degrees(),p->alt().Degrees()});
        regions.append(QJsonObject{{"name",region->region()},{"ceiling",region->ceiling()},{"points",points}});
    }
    const char *horizonKind =
#ifdef LIGHTPOLLUTION_HORIZON_GREAT_CIRCLE
        "KStars_great_circle_v1";
#else
        "KStars_piecewise_linear_v1";
#endif
    return {{"schema_version",2},{"kstars_version",QString(KSTARS_VERSION)},
        {"utc",data->ut().toUTC().toString(Qt::ISODate)},
        {"altitude_convention","geometric_unrefracted"},
        {"location",QJsonObject{{"latitude_deg",geo->lat()->Degrees()},{"longitude_deg",geo->lng()->Degrees()},
                               {"elevation_m",geo->elevation()},{"name",geo->name()}}},
        {"focus",focus},{"artificial_horizon_visible",visible},
        {"artificial_horizon",QJsonObject{{"kind",horizonKind},{"regions",regions}}}};
}

QJsonObject LightPollutionDock::scene() const
{
    QJsonObject value=m_Base;
    value.insert("region",m_Region->currentData().toString());value.insert("period",m_Period->currentText());
    value.insert("observer",QJsonObject{{"height_m",m_Height->value()}});
    auto atmosphere=value.value("atmosphere").toObject();atmosphere.insert("aerosol_optical_depth",m_AOD->value());value.insert("atmosphere",atmosphere);
    if(m_Near->isChecked())value.insert("near_field",value.value("near_field").toObject());else value.remove("near_field");
    if(m_Spectral->isChecked())value.insert("spectrum",value.value("spectrum").toObject());else value.remove("spectrum");
    if(m_Eye->isChecked()){
        auto eye=value.value("eye").toObject();
        eye.insert("appearance",QJsonObject{{"mode",m_Colour->isChecked()?"experimental_stellar_colour":"monochrome"}});
        value.insert("eye",eye);
    }else value.remove("eye");
    if(m_Camera->isChecked()){
        auto camera=value.value("camera").toObject();camera.remove("catalogue");camera.insert("exposure_s",m_Exposure->value());
        auto profile=camera.value("profile").toObject();auto optics=profile.value("optics").toObject();
        optics.insert("focal_length_mm",m_Focal->value());optics.insert("aperture_diameter_mm",m_Aperture->value());
        profile.insert("optics",optics);camera.insert("profile",profile);value.insert("camera",camera);
    }else value.remove("camera");
    if(!m_Terrain->currentData().toString().isEmpty()){auto t=value.value("terrain").toObject();t.insert("dataset",m_Terrain->currentData().toString());value.insert("terrain",t);}else value.remove("terrain");
    if(!value.contains("background"))value.insert("background",QJsonObject{{"twilight","patat2006"}});
    return value;
}

QString LightPollutionDock::contextKey() const
{
    auto frame=snapshot();auto focus=frame.value("focus").toObject();
    focus.insert("azimuth_deg",std::round(focus.value("azimuth_deg").toDouble()*4)/4);
    focus.insert("altitude_deg",std::round(focus.value("altitude_deg").toDouble()*4)/4);
    focus.remove("ra_jnow_deg");focus.remove("dec_jnow_deg");frame.insert("focus",focus);
    auto utc=QDateTime::fromString(frame.value("utc").toString(),Qt::ISODate);utc=QDateTime::fromSecsSinceEpoch(utc.toSecsSinceEpoch()/60*60,Qt::UTC);frame.insert("utc",utc.toString(Qt::ISODate));
    return QString::fromLatin1(QCryptographicHash::hash(QJsonDocument(QJsonObject{{"snapshot",frame},{"scene",scene()}}).toJson(QJsonDocument::Compact),QCryptographicHash::Sha256).toHex());
}

QString LightPollutionDock::snapshotJSON() const{return QString::fromUtf8(QJsonDocument(snapshot()).toJson(QJsonDocument::Compact));}
QString LightPollutionDock::statusJSON() const
{
    QJsonObject summary{{"busy",m_Busy},{"job_id",m_Job},{"message",m_Status->text()},
        {"catalogue_ready",m_Region->count()>0 && !m_Region->currentData().toString().isEmpty() && m_Period->count()>0},
        {"current_scene",scene()},
        {"stale",!m_ResultKey.isEmpty() && m_ResultKey!=contextKey()},
        {"integration",m_LastResult.value("integration")},{"sky",m_LastResult.value("sky")},
        {"request",m_LastResult.value("request")},
        {"eye_display",m_LastResult.value("eye").toObject().value("display")},
        {"eye_preview_ready",m_EyePreviewReady}};
    summary.insert("camera_FITS_ready",m_CameraFITSReady);
    summary.insert("Qt_version",QString(qVersion()));
    summary.insert("camera",m_LastResult.value("camera").toObject().value("exports"));
#ifdef LP_HAVE_EKOS_EXPOSURE
    summary.insert("Ekos_exposure_calculator_compiled",true);
#else
    summary.insert("Ekos_exposure_calculator_compiled",false);
#endif
    return QString::fromUtf8(QJsonDocument(summary).toJson(QJsonDocument::Compact));
}
bool LightPollutionDock::setSceneJSON(const QString &json)
{
    QJsonParseError error;const auto doc=QJsonDocument::fromJson(json.toUtf8(),&error);
    if(error.error!=QJsonParseError::NoError || !doc.isObject())return false;
    m_Base=doc.object();m_Region->setCurrentIndex(m_Region->findData(m_Base.value("region").toString()));
    m_Period->setCurrentText(m_Base.value("period").toString());
    m_Height->setValue(m_Base.value("observer").toObject().value("height_m").toDouble(2));
    m_AOD->setValue(m_Base.value("atmosphere").toObject().value("aerosol_optical_depth").toDouble(.1));
    m_Near->setChecked(m_Base.contains("near_field"));m_Spectral->setChecked(m_Base.contains("spectrum"));
    m_Eye->setChecked(m_Base.contains("eye"));
    m_Colour->setChecked(m_Base.value("eye").toObject().value("appearance").toObject().value("mode").toString()=="experimental_stellar_colour");
    m_Camera->setChecked(m_Base.contains("camera"));const auto camera=m_Base.value("camera").toObject();
    m_Exposure->setValue(camera.value("exposure_s").toDouble(30));const auto optics=camera.value("profile").toObject().value("optics").toObject();
    m_Focal->setValue(optics.value("focal_length_mm").toDouble(16));m_Aperture->setValue(optics.value("aperture_diameter_mm").toDouble(8));
    m_Terrain->setCurrentIndex(std::max(0,m_Terrain->findData(m_Base.value("terrain").toObject().value("dataset").toString())));
    return true;
}
bool LightPollutionDock::savePanelScreenshot(const QString &path){return grab().save(path);}
bool LightPollutionDock::saveWindowScreenshot(const QString &path){return m_KStars->grab().save(path);}
bool LightPollutionDock::saveEyeScreenshot(const QString &path)
{return m_EyeDialog && m_EyeDialog->isVisible() && m_EyeDialog->grab().save(path);}
void LightPollutionDock::showEyeView()
{
    if(m_Busy || !m_OpenEye->isEnabled())return;
    const QString requestedJob=m_Job;
    auto *reply=m_Network.get(QNetworkRequest(QUrl(m_URL+"/api/jobs/"+m_Job+"/export/eye-preview.png")));
    QTimer::singleShot(15000,reply,&QNetworkReply::abort);
    connect(reply,&QNetworkReply::finished,this,[this,reply,requestedJob](){
        if(m_Busy || m_Job!=requestedJob){reply->deleteLater();return;}
        QPixmap picture;
        if(reply->error()!=QNetworkReply::NoError || !picture.loadFromData(reply->readAll()))
        {setStatus(tr("Изображение недоступно; проверьте область модели глаза."));reply->deleteLater();return;}
        auto *dialog=new QDialog(m_KStars);dialog->setAttribute(Qt::WA_DeleteOnClose);
        dialog->setWindowTitle(tr("Модель потока и оптики · вид невооружённым глазом"));
        auto *layout=new QVBoxLayout(dialog);auto *scroll=new QScrollArea(dialog);auto *label=new QLabel;
        // The backend integrated stellar flux at this exact pixel resolution.
        // Rescaling an encoded point image here would change its brightness.
        label->setPixmap(picture);
        scroll->setWidget(label);layout->addWidget(scroll);dialog->resize(picture.width()+40,picture.height()+50);dialog->show();
        m_EyeDialog=dialog;m_EyePreviewReady=true;reply->deleteLater();});
}
void LightPollutionDock::setStatus(const QString &text){m_Status->setText(text);}

QJsonObject LightPollutionDock::cameraCatalogue() const
{
    const auto camera=scene().value("camera").toObject();const auto sensor=camera.value("profile").toObject().value("sensor").toObject();
    const double width=sensor.value("width_px").toDouble(960),height=sensor.value("height_px").toDouble(640),pitch=sensor.value("pixel_pitch_um").toDouble(3.76);
    const double radius=std::atan(std::hypot(width,height)*pitch*.001/(2*m_Focal->value()))*180/M_PI+.3;
    auto *data=m_KStars->data();SkyPoint center=*m_KStars->map()->focus();SkyPoint catalogueCenter=center;
    catalogueCenter.catalogueCoord(data->ut().djd());center.setRA0(catalogueCenter.ra0());center.setDec0(catalogueCenter.dec0());
    QJsonArray files;
    for(const auto &name:{"namedstars.dat","unnamedstars.dat","tycho2.dat","deepstars.dat","USNO-NOMAD-1e8.dat","Henry-Draper.idx","catalogue-provenance.json"}){
        const QString path=KSPaths::locate(QStandardPaths::AppLocalDataLocation,name);
        QFile file(path);if(!file.open(QIODevice::ReadOnly))continue;
        QCryptographicHash hash(QCryptographicHash::Sha256);hash.addData(&file);
        files.append(QJsonObject{{"name",name},{"bytes",double(file.size())},{"sha256",QString::fromLatin1(hash.result().toHex())}});
    }
    QFile metadataFile(KSPaths::locate(QStandardPaths::AppLocalDataLocation,"catalogue-provenance.json"));
    QJsonObject metadata;if(metadataFile.open(QIODevice::ReadOnly))metadata=QJsonDocument::fromJson(metadataFile.readAll()).object();
    const auto expectedFiles=metadata.value("source_files").toObject();
    bool bounded=metadata.value("schema_version").toInt()==1 && expectedFiles.size()==3;
    int matched=0;
    for(const auto &entry:files){const auto file=entry.toObject();const QString name=file.value("name").toString();
        if(!name.endsWith(".dat"))continue;
        if(expectedFiles.value(name).toString()!=file.value("sha256").toString())bounded=false;else ++matched;
    }
    bounded=bounded && matched==3;
    const double maximumPM=metadata.value("maximum_pm_mas_yr").toDouble(-1);
    bounded=bounded && std::isfinite(maximumPM) && maximumPM>=0 && maximumPM<=1e6;
    SkyPoint epochCenter(catalogueCenter.ra0(),catalogueCenter.dec0());
    const double epochPadding=center.angularDistanceTo(&epochCenter).Degrees();
    const double motionPadding=bounded?std::abs(data->ut().djd()-2451545.)/365.25*maximumPM/3600000.:0.;
    const double queryRadius=std::min(180.,radius+epochPadding+motionPadding);
    QList<StarObject *> stars;auto *component=StarComponent::Instance();
    if(component)component->starsInAperture(stars,center,queryRadius,12.);
    std::sort(stars.begin(),stars.end(),[](const StarObject *a,const StarObject *b){return a->mag()<b->mag();});
    QJsonArray rows;QSet<QString> seen;bool truncated=false;
    for(auto *star:stars){
        if(!star || !std::isfinite(star->mag()) || star->mag()>12 || star->mag()<-5)continue;
        star->JITupdate();star->EquatorialToHorizontal(data->lst(),data->geo()->lat());
        if(star->angularDistanceTo(&center).Degrees()>radius)continue;
        // Identity includes every exported physical attribute, not just position/HD.
        const double bv=star->getBVIndex();const QString type=star->sptype();
        const QJsonArray identity{star->ra0().Degrees(),star->dec0().Degrees(),star->pmRA(),star->pmDec(),
            star->parallax(),star->getHDIndex(),star->mag(),bv,type,star->isMultiple(),star->isVariable()};
        const auto hash=QCryptographicHash::hash(QJsonDocument(identity).toJson(QJsonDocument::Compact),QCryptographicHash::Sha256).toHex();
        const QString id=QString("HD%1:%2").arg(star->getHDIndex()).arg(QString::fromLatin1(hash));
        if(seen.contains(id))continue;
        seen.insert(id);
        if(rows.size()>=20000){truncated=true;break;}
        const bool tycho=type.size()>=2 && type[0]!='?' && type[1]=='?' && star->parallax()==0;
        rows.append(QJsonObject{{"id",id},{"name",star->name().left(160)},{"azimuth_deg",star->az().Degrees()},{"altitude_deg",star->alt().Degrees()},
            {"v_mag",star->mag()},{"b_v",std::isfinite(bv)&&bv>=-1&&bv<=4?QJsonValue(bv):QJsonValue()},
            {"photometry",tycho?"Tycho_VT_BTminusVT_inferred":"Johnson_V_BV"},
            {"astrometry",QJsonObject{{"ra_j2000_deg",star->ra0().Degrees()},{"dec_j2000_deg",star->dec0().Degrees()},
                {"pm_ra_cosdec_mas_yr",star->pmRA()},{"pm_dec_mas_yr",star->pmDec()},
                {"hd",star->getHDIndex()},{"multiple",star->isMultiple()}}}});
    }
    return {{"provider","KStars"},{"utc",data->ut().toUTC().toString(Qt::ISODate)},{"magnitude_limit",12.},{"truncated",truncated},{"rows",rows},
        {"provenance",QJsonObject{{"files",files},{"qt_version",qVersion()},
        {"aperture",QJsonObject{{"radius_deg",radius},{"query_radius_deg",queryRadius},{"epoch_rotation_padding_deg",epochPadding},
            {"proper_motion_padding_deg",motionPadding},{"proper_motion_bound_verified",bounded}}},
        {"colour_policy","StarData B-V /100; encoded zero remains unknown; DeepStarData B-V from B,V"},
        {"deduplication","same exported astrometry/photometry/spectral class/flags only; prepared catalogue uses pinned upstream cross-ID"}}}};
}

QString LightPollutionDock::catalogueJSON() const
{return QString::fromUtf8(QJsonDocument(cameraCatalogue()).toJson(QJsonDocument::Compact));}

bool LightPollutionDock::setVerificationHorizonJSON(const QString &json)
{
    // Private test sessions only. Never write KStars' user database.
    if(qEnvironmentVariable("LIGHTPOLLUTION_VERIFICATION")!="1")return false;
    const auto regions=QJsonDocument::fromJson(json.toUtf8()).array();
    auto *component=m_KStars->data()->skyComposite()->artificialHorizon();if(!component)return false;
    for(const auto &value:regions){const auto region=value.toObject();auto line=std::make_shared<LineList>();
        for(const auto &point:region.value("points").toArray()){
            const auto angles=point.toArray();if(angles.size()!=2)return false;
            auto p=std::make_shared<SkyPoint>();p->setAz(angles[0].toDouble());p->setAlt(angles[1].toDouble());line->append(p);
        }
        component->addRegion(region.value("name").toString(),true,line,region.value("ceiling").toBool());
    }
    return true;
}

QString LightPollutionDock::horizonVisibilityJSON(const QString &json) const
{
    const auto points=QJsonDocument::fromJson(json.toUtf8()).array();QJsonArray values;
    auto *component=m_KStars->data()->skyComposite()->artificialHorizon();
    for(const auto &value:points){const auto p=value.toArray();if(p.size()!=2)return "[]";
        values.append(!component || component->getHorizon().isVisible(p[0].toDouble(),p[1].toDouble()));}
    return QString::fromUtf8(QJsonDocument(values).toJson(QJsonDocument::Compact));
}

void LightPollutionDock::showEkosExposure()
{
#ifdef LP_HAVE_EKOS_EXPOSURE
    const auto camera=m_LastResult.value("camera").toObject();
    const auto bridge=camera.value("exports").toObject().value("exposure_bridge").toObject();
    if(bridge.isEmpty() || m_Busy)return;
    if(m_EkosDialog)m_EkosDialog->close();
    auto *dialog=new QDialog(m_KStars);m_EkosDialog=dialog;dialog->setAttribute(Qt::WA_DeleteOnClose);
    dialog->setWindowTitle(tr("Ekos / LightPollution · экспозиция из потока e⁻"));
    auto *layout=new QVBoxLayout(dialog);
    auto *notice=new QLabel(tr("Тот же профиль, что в FITS. Поток уже включает отклик сенсора и оптику.\n"
        "Нативный OptimalSubExposureCalculator Ekos; SQM/Bortle и повторный QE не используются.\n"
        "Это границы сценария. Настройки INDI и очередь Capture не меняются."),dialog);
    notice->setWordWrap(true);layout->addWidget(notice);
    auto *profileLabel=new QLabel(camera.value("config").toObject().value("profile").toObject().value("profile_id").toString(),dialog);
    profileLabel->setWordWrap(true);layout->addWidget(profileLabel);
    auto *tolerance=new QDoubleSpinBox(dialog);tolerance->setRange(.1,100);tolerance->setValue(5);
    auto *form=new QFormLayout;form->addRow(tr("Допустимый рост шума, %"),tolerance);layout->addLayout(form);
    auto *table=new QTableWidget(3,3,dialog);table->setHorizontalHeaderLabels({tr("Участок поля"),tr("Фон e⁻/пиксель/с"),tr("Нижняя граница, с")});
    table->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);table->setEditTriggers(QAbstractItemView::NoEditTriggers);layout->addWidget(table);
    auto *bounds=new QLabel(dialog);bounds->setWordWrap(true);layout->addWidget(bounds);
    auto update=[this,bridge,tolerance,table,bounds](){
        const double rn=bridge.value("read_noise_e").toDouble();const auto rates=bridge.value("sky_rate_e_pixel_s_min_median_max").toArray();
        QJsonArray checks;double worst=0.;
        for(int i=0;i<3;++i){const double rate=rates[i].toDouble();double exposure=0.;
            if(rate>0 && rn>0){
                // One immutable mono profile: gain labels 0/1 share exactly this RN.
                QVector<OptimalExposure::CameraGainReadNoise> noises{{0,rn},{1,rn}};
                QVector<OptimalExposure::CameraGainReadMode> modes{{0,"Frozen LightPollution profile",noises}};
                OptimalExposure::ImagingCameraData sensor(bridge.value("profile_id").toString(),OptimalExposure::SENSORTYPE_MONOCHROME,
                    OptimalExposure::GAIN_SELECTION_TYPE_NORMAL,{0,1},modes,bridge.value("pixel_pitch_um").toDouble(),1.);
                OptimalExposure::OptimalSubExposureCalculator calculator(tolerance->value(),20.,bridge.value("focal_ratio").toDouble(),1.,sensor);
                if(!calculator.setExternalSkyElectronRate(rate))return;
                calculator.setASelectedCameraReadMode(0);calculator.setASelectedGain(0);
                const auto detail=calculator.calculateSubExposureDetail();exposure=detail.getSubExposureTime();
                checks.append(QJsonObject{{"sky_rate_e_pixel_s",rate},{"exposure_s",exposure},
                    {"sky_e",detail.getExposurePollutionElectrons()},{"total_noise_e",detail.getExposureTotalNoise()}});
            }else checks.append(QJsonObject{{"sky_rate_e_pixel_s",rate},{"exposure_s",rate>0?QJsonValue(0.):QJsonValue()}});
            const QString labels[]={tr("Минимум фона"),tr("Медиана"),tr("Максимум фона")};
            table->setItem(i,0,new QTableWidgetItem(labels[i]));table->setItem(i,1,new QTableWidgetItem(QString::number(rate,'g',8)));
            table->setItem(i,2,new QTableWidgetItem(rate>0?QString::number(exposure,'g',8):tr("Нет конечной границы")));worst=std::max(worst,exposure);
        }
        const auto upper=bridge.value("exposure_upper_bound_s_at_90_percent_capacity");
        const bool compatible=rates[0].toDouble()>0 && upper.isDouble() && worst<=upper.toDouble();
        bounds->setText(tr("Верхняя граница при 90% ёмкости: %1 с.\nСовместимый интервал: %2. Насыщение включает звёзды из доступного каталога.")
            .arg(upper.isDouble()?QString::number(upper.toDouble(),'g',8):tr("не определена")).arg(compatible?tr("есть"):tr("нет")));
        m_EkosVerification={{"engine","OptimalExposure::OptimalSubExposureCalculator::calculateSubExposureDetail"},
            {"profile_id",bridge.value("profile_id")},{"noise_increase_percent",tolerance->value()},{"rows",checks},
            {"sky_input_kind","direct_electrons_per_pixel_per_second"},{"SQM_used",false},{"QE_or_focal_ratio_applied_twice",false},
            {"bounds_compatible",compatible},{"hardware_or_capture_queue_changed",false}};
    };
    connect(tolerance,QOverload<double>::of(&QDoubleSpinBox::valueChanged),dialog,[update](double){update();});update();
    dialog->resize(850,430);dialog->show();
#endif
}

QString LightPollutionDock::ekosExposureJSON() const
{return QString::fromUtf8(QJsonDocument(m_EkosVerification).toJson(QJsonDocument::Compact));}

bool LightPollutionDock::saveEkosScreenshot(const QString &path)
{return m_EkosDialog && m_EkosDialog->grab().save(path,"PNG");}

void LightPollutionDock::showCameraFITS()
{
#ifdef HAVE_CFITSIO
    if(m_Busy || !m_OpenCamera->isEnabled() || !m_CameraFiles.isValid())return;
    const QString requestedJob=m_Job;m_CameraFITSReady=false;
    auto *reply=m_Network.get(QNetworkRequest(QUrl(m_URL+"/api/jobs/"+requestedJob+"/export/camera.fits")));
    QTimer::singleShot(15000,reply,&QNetworkReply::abort);
    connect(reply,&QNetworkReply::finished,this,[this,reply,requestedJob](){
        const auto bytes=reply->readAll();
        if(m_Job!=requestedJob || reply->error()!=QNetworkReply::NoError || !bytes.startsWith("SIMPLE")){
            setStatus(tr("Не удалось открыть синтетический FITS."));reply->deleteLater();return;}
        const QString path=m_CameraFiles.path()+"/"+requestedJob+".fits";QFile file(path);
        if(!file.open(QIODevice::WriteOnly) || file.write(bytes)!=bytes.size()){setStatus(tr("Не удалось сохранить временный FITS."));reply->deleteLater();return;}
        file.close();if(!m_FITSViewer)m_FITSViewer=viewerPointer(m_KStars->createFITSViewer());
        disconnect(m_FITSViewer,&FITSViewer::loaded,this,nullptr);
        connect(m_FITSViewer,&FITSViewer::loaded,this,[this,requestedJob](int){if(m_Job==requestedJob)m_CameraFITSReady=true;});
        connect(m_FITSViewer,&FITSViewer::failed,this,[this](const QString &error){setStatus(tr("FITS Viewer: ")+error);});
        m_FITSViewer->loadFile(QUrl::fromLocalFile(path),FITS_NORMAL,FITS_NONE,tr("SIMULATED LightPollution; not a measured RAW"));
        m_FITSViewer->show();m_FITSViewer->raise();reply->deleteLater();});
#endif
}

bool LightPollutionDock::saveCameraScreenshot(const QString &path)
{
#ifdef HAVE_CFITSIO
    return m_FITSViewer && m_FITSViewer->isVisible() && m_CameraFITSReady && m_FITSViewer->grab().save(path);
#else
    Q_UNUSED(path);return false;
#endif
}

void LightPollutionDock::request(const QString &path,const QJsonObject &payload,
    std::function<void(const QJsonObject &,const QString &)> callback,bool post)
{
    QNetworkRequest request(QUrl(m_URL+path));request.setHeader(QNetworkRequest::ContentTypeHeader,"application/json");
    auto *reply=post?m_Network.post(request,QJsonDocument(payload).toJson(QJsonDocument::Compact)):m_Network.get(request);
    QTimer::singleShot(15000,reply,&QNetworkReply::abort);
    connect(reply,&QNetworkReply::finished,this,[reply,callback](){const auto data=QJsonDocument::fromJson(reply->readAll());
        QString error;if(reply->error()!=QNetworkReply::NoError)error=data.object().value("error").toObject().value("message").toString(reply->errorString());
        // Catalogue is an array; keep the generic callback object-based.
        callback(data.isArray()?QJsonObject{{"items",data.array()}}:data.object(),error);reply->deleteLater();});
}
void LightPollutionDock::fetchCatalogue()
{
    request("/api/kstars/capabilities",{},[this](const QJsonObject &capabilities,const QString &error){
        if(!error.isEmpty() || capabilities.value("snapshot_schema").toInt()!=2)
        {setStatus(tr("Несовместимое или недоступное расчётное ядро. Нужен snapshot schema2. ")+error);return;}
    request("/api/catalogue",{},[this](const QJsonObject &data,const QString &error){if(!error.isEmpty()){setStatus(tr("Ядро недоступно: ")+error);return;}
        m_Region->clear();m_Regions=QJsonObject();
        for(const auto &item:data.value("items").toArray()){const auto r=item.toObject();m_Regions.insert(r.value("id").toString(),r.value("periods"));m_Region->addItem(r.value("title").toString(),r.value("id").toString());}
        m_Region->setCurrentIndex(std::max(0,m_Region->findData(m_Base.value("region").toString("petersburg-2025"))));
        if(m_Base.contains("period"))m_Period->setCurrentText(m_Base.value("period").toString());
        setStatus(tr("Готово. Направление, место и UTC берутся из KStars."));},false);
    },false);
}
void LightPollutionDock::calculate()
{
    if(m_Busy)return;
    if(m_Region->count()==0 || m_Period->count()==0 || m_Region->currentData().toString().isEmpty()){
        setStatus(tr("Подождите загрузки регионов и периодов источников."));fetchCatalogue();return;}
    const auto frame=snapshot();const double alt=frame.value("focus").toObject().value("altitude_deg").toDouble();
    if(alt<10 || alt>90){setStatus(tr("Количественный режим поддерживает высоты 10–90°. Переместите центр карты."));return;}
    auto requestScene=scene();if(m_Camera->isChecked()){auto camera=requestScene.value("camera").toObject();camera.insert("catalogue",cameraCatalogue());requestScene.insert("camera",camera);}
    m_SubmittedKey=contextKey();m_Busy=true;m_CancelSent=false;m_Job.clear();m_OpenEye->setEnabled(false);m_EyePreviewReady=false;m_OpenCamera->setEnabled(false);m_CameraFITSReady=false;setStatus(tr("Расчёт снимка состояния KStars…"));
    request("/api/kstars/jobs",{{"scene",requestScene},{"snapshot",frame}},[this](const QJsonObject &data,const QString &error){if(!error.isEmpty() || data.value("id").toString().isEmpty()){m_Busy=false;setStatus(error.isEmpty()?tr("Ядро не вернуло идентификатор задания."):error);return;}m_Job=data.value("id").toString();poll();});
}
void LightPollutionDock::tick()
{
    if(!isVisible())return;
    auto frame=snapshot();auto f=frame.value("focus").toObject();auto l=frame.value("location").toObject();
    m_Position->setText(QString("%1\n%2°, %3° · азимут %4°, высота %5°").arg(frame.value("utc").toString()).arg(l.value("latitude_deg").toDouble(),0,'f',4).arg(l.value("longitude_deg").toDouble(),0,'f',4).arg(f.value("azimuth_deg").toDouble(),0,'f',2).arg(f.value("altitude_deg").toDouble(),0,'f',2));
    const QString key=contextKey();if(key!=m_CurrentKey){m_CurrentKey=key;m_StableTicks=0;if(!m_LastResult.isEmpty())setStatus(tr("Параметры изменились: показан предыдущий снимок, пересчёт ожидается."));}else ++m_StableTicks;
    if(m_Busy){poll();if(m_Follow->isChecked() && m_SubmittedKey!=key && !m_CancelSent && !m_Job.isEmpty()){m_CancelSent=true;request("/api/jobs/"+m_Job+"/cancel",{},[](const QJsonObject &,const QString &){});}return;}
    if(m_Follow->isChecked() && m_StableTicks>=1 && key!=m_SubmittedKey)calculate();
}
void LightPollutionDock::poll()
{
    if(m_Job.isEmpty() || m_PollPending)return;
    m_PollPending=true;
    request("/api/jobs/"+m_Job,{},[this](const QJsonObject &job,const QString &error){m_PollPending=false;
        if(!error.isEmpty()){m_Busy=false;setStatus(error);return;}const auto state=job.value("state").toString();
        if(state=="completed"){m_PollPending=true;request("/api/jobs/"+m_Job+"/result",{},[this](const QJsonObject &value,const QString &problem){m_PollPending=false;m_Busy=false;if(problem.isEmpty())showResult(value);else setStatus(problem);},false);}
        else if(state=="error" || state=="cancelled"){m_Busy=false;setStatus(state=="error"?job.value("error").toObject().value("message").toString():tr("Предыдущий расчёт отменён."));}
        else setStatus(job.value("progress").toObject().value("phase").toString(tr("В очереди…")));},false);
}
void LightPollutionDock::showResult(const QJsonObject &value)
{
    m_ResultKey=m_SubmittedKey;
    m_OpenEye->setEnabled(value.value("eye").toObject().value("display").toObject().value("available").toBool());
    const bool cameraAvailable=value.value("camera").toObject().value("exports").toObject().value("available").toBool();
#ifdef HAVE_CFITSIO
    m_OpenCamera->setEnabled(cameraAvailable);
#endif
#ifdef LP_HAVE_EKOS_EXPOSURE
    m_EkosExposure->setEnabled(cameraAvailable);
#endif
    m_LastResult=value;const auto rows=value.value("sky").toObject().value("directions").toArray();if(rows.isEmpty())return;const auto d=rows.first().toObject();
    const QList<QPair<QString,QString>> fields={{tr("Город, B/Bref"),"artificial_relative"},{tr("Прямой городской"),"artificial_direct_relative"},{tr("Отражённый городской"),"artificial_reflected_relative"},{tr("Луна, B/Bref"),"moon_relative"},{tr("Ночной естественный"),"natural_relative"},{tr("Сумерки, B/Bref"),"twilight_relative"},{tr("Сумма, V mag/arcsec²"),"total_v_mag_arcsec2"}};
    m_Values->setRowCount(fields.size());for(int i=0;i<fields.size();++i){m_Values->setItem(i,0,new QTableWidgetItem(fields[i].first));const auto v=d.value(fields[i].second);m_Values->setItem(i,1,new QTableWidgetItem(v.isDouble()?QString::number(v.toDouble(),'g',7):QString::fromUtf8("—")));}
    const auto catalogue=value.value("camera").toObject().value("catalogue").toObject();
    if(!catalogue.isEmpty()){
        const QList<QPair<QString,QString>> extra={{tr("Каталог: записей"),"rows_received"},
            {tr("Слабейшая входная величина"),"faintest_received_v_mag"},
            {tr("Каталожный цвет: записей"),"native_colour_rows"},
            {tr("Преобразование Tycho недоступно"),"photometry_unavailable_rows"}};
        const int begin=m_Values->rowCount();m_Values->setRowCount(begin+extra.size());
        for(int i=0;i<extra.size();++i){m_Values->setItem(begin+i,0,new QTableWidgetItem(extra[i].first));
            const auto n=catalogue.value(extra[i].second);m_Values->setItem(begin+i,1,new QTableWidgetItem(n.isDouble()?QString::number(n.toDouble(),'g',7):QString::fromUtf8("—")));}
    }
    m_Scale->setText(tr("Масштаб: ")+value.value("calibration").toObject().value("origin").toString()+tr(". Высота KStars сохранена как справочная; AGL задаётся отдельно. Для глаза доступен порог полной адаптации; экранный вид и сенсор не калиброваны."));
    setStatus(m_SubmittedKey!=contextKey()?tr("Получен предыдущий снимок; карта уже изменилась."):(d.value("supported").toBool()?tr("Готово для снимка состояния: ")+value.value("request").toObject().value("time").toString():tr("Сумма недоступна: ")+d.value("status").toString()));
}
