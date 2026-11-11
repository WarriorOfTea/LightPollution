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
#include "version.h"

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
#include <QHeaderView>
#include <QJsonArray>
#include <QJsonDocument>
#include <QLabel>
#include <QNetworkReply>
#include <QPushButton>
#include <QTableWidget>
#include <QUrl>
#include <QVBoxLayout>
#include <cmath>
#include <algorithm>

LightPollutionDock::LightPollutionDock(KStars *parent) : QDockWidget("LightPollution", parent), m_KStars(parent)
{
    setObjectName("LightPollutionDock");
    setMinimumWidth(360);
    auto *body=new QWidget(this);
    auto *layout=new QVBoxLayout(body);
    m_Position=new QLabel(body); m_Position->setWordWrap(true); layout->addWidget(m_Position);
    auto *form=new QFormLayout;
    m_Region=new QComboBox(body); m_Period=new QComboBox(body); m_Terrain=new QComboBox(body);
    m_Terrain->addItem(tr("Без высот"), "");
    m_Terrain->addItem(tr("Петербург · Copernicus DSM"), "petersburg-copdem90");
    m_Terrain->addItem(tr("Мадрид · Copernicus DSM"), "madrid-copdem90");
    m_Terrain->addItem(tr("Тест: плоская поверхность"), "demo-flat");
    m_Height=new QDoubleSpinBox(body);m_Height->setRange(.5,100);m_Height->setDecimals(1);m_Height->setValue(2);
    m_AOD=new QDoubleSpinBox(body);m_AOD->setRange(0,2);m_AOD->setSingleStep(.05);m_AOD->setValue(.1);
    m_Near=new QCheckBox(tr("Конечные ближние ячейки"),body);m_Near->setChecked(true);
    m_Spectral=new QCheckBox(tr("Спектральный сценарий"),body);
    m_Eye=new QCheckBox(tr("Порог невооружённого глаза"),body);
    m_OpenEye=new QPushButton(tr("Открыть монохромный вид"),body);m_OpenEye->setEnabled(false);
    m_Follow=new QCheckBox(tr("Следить за картой и временем"),body);
    form->addRow(tr("Источники"),m_Region);form->addRow(tr("Период"),m_Period);
    form->addRow(tr("Рельеф"),m_Terrain);form->addRow(tr("Высота над DSM, м"),m_Height);
    form->addRow(tr("AOD 550 нм"),m_AOD);layout->addLayout(form);
    layout->addWidget(m_Near);layout->addWidget(m_Spectral);layout->addWidget(m_Eye);
    layout->addWidget(m_OpenEye);layout->addWidget(m_Follow);
    auto *buttons=new QHBoxLayout;
    auto *run=new QPushButton(tr("Рассчитать"),body);auto *cancel=new QPushButton(tr("Отменить"),body);
    buttons->addWidget(run);buttons->addWidget(cancel);layout->addLayout(buttons);
    auto *files=new QHBoxLayout;
    auto *load=new QPushButton(tr("Открыть сценарий"),body);auto *save=new QPushButton(tr("Сохранить результат"),body);
    files->addWidget(load);files->addWidget(save);layout->addLayout(files);
    m_Status=new QLabel(tr("Подключение к локальному расчётному ядру…"),body);m_Status->setWordWrap(true);layout->addWidget(m_Status);
    m_Values=new QTableWidget(0,2,body);m_Values->setHorizontalHeaderLabels({tr("Компонент"),tr("Значение")});
    m_Values->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    m_Values->verticalHeader()->hide();m_Values->setEditTriggers(QAbstractItemView::NoEditTriggers);layout->addWidget(m_Values);
    m_Scale=new QLabel(tr("Сценарный масштаб. Порог глаза — ограниченный режим полной адаптации; цвет и сенсор не калиброваны."),body);
    m_Scale->setWordWrap(true);layout->addWidget(m_Scale);setWidget(body);
    m_URL=qEnvironmentVariable("LIGHTPOLLUTION_URL","http://127.0.0.1:8765");
    const QUrl url(m_URL);
    if(url.scheme()!="http" || !(url.host()=="127.0.0.1" || url.host()=="localhost" || url.host()=="::1"))
        m_URL="http://127.0.0.1:8765";
    connect(run,&QPushButton::clicked,this,&LightPollutionDock::calculate);
    connect(m_Eye,&QCheckBox::toggled,this,[this](bool enabled){if(enabled)m_Spectral->setChecked(true);});
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
    return {{"schema_version",1},{"kstars_version",QString(KSTARS_VERSION)},
        {"utc",data->ut().toUTC().toString(Qt::ISODate)},
        {"altitude_convention","geometric_unrefracted"},
        {"location",QJsonObject{{"latitude_deg",geo->lat()->Degrees()},{"longitude_deg",geo->lng()->Degrees()},
                               {"elevation_m",geo->elevation()},{"name",geo->name()}}},
        {"focus",focus},{"artificial_horizon_visible",visible}};
}

QJsonObject LightPollutionDock::scene() const
{
    QJsonObject value=m_Base;
    value.insert("region",m_Region->currentData().toString());value.insert("period",m_Period->currentText());
    value.insert("observer",QJsonObject{{"height_m",m_Height->value()}});
    auto atmosphere=value.value("atmosphere").toObject();atmosphere.insert("aerosol_optical_depth",m_AOD->value());value.insert("atmosphere",atmosphere);
    if(m_Near->isChecked())value.insert("near_field",value.value("near_field").toObject());else value.remove("near_field");
    if(m_Spectral->isChecked())value.insert("spectrum",value.value("spectrum").toObject());else value.remove("spectrum");
    if(m_Eye->isChecked())value.insert("eye",value.value("eye").toObject());else value.remove("eye");
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
        {"stale",!m_SubmittedKey.isEmpty() && m_SubmittedKey!=contextKey()},
        {"integration",m_LastResult.value("integration")},{"sky",m_LastResult.value("sky")},
        {"request",m_LastResult.value("request")},
        {"eye_display",m_LastResult.value("eye").toObject().value("display")},
        {"eye_preview_ready",m_EyePreviewReady}};
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
    auto *reply=m_Network.get(QNetworkRequest(QUrl(m_URL+"/api/jobs/"+m_Job+"/export/eye-view.png")));
    QTimer::singleShot(15000,reply,&QNetworkReply::abort);
    connect(reply,&QNetworkReply::finished,this,[this,reply,requestedJob](){
        if(m_Busy || m_Job!=requestedJob){reply->deleteLater();return;}
        QPixmap picture;
        if(reply->error()!=QNetworkReply::NoError || !picture.loadFromData(reply->readAll()))
        {setStatus(tr("Изображение недоступно; проверьте область модели глаза."));reply->deleteLater();return;}
        auto *dialog=new QDialog(m_KStars);dialog->setAttribute(Qt::WA_DeleteOnClose);
        dialog->setWindowTitle(tr("Монохромная иллюстрация · сохранённый сценарий"));
        auto *layout=new QVBoxLayout(dialog);auto *scroll=new QScrollArea(dialog);auto *label=new QLabel;
        label->setPixmap(picture.scaled(1200,800,Qt::KeepAspectRatio,Qt::SmoothTransformation));
        scroll->setWidget(label);layout->addWidget(scroll);dialog->resize(1230,840);dialog->show();
        m_EyeDialog=dialog;m_EyePreviewReady=true;reply->deleteLater();});
}
void LightPollutionDock::setStatus(const QString &text){m_Status->setText(text);}

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
        if(!error.isEmpty() || capabilities.value("snapshot_schema").toInt()!=1)
        {setStatus(tr("Несовместимое или недоступное расчётное ядро. Нужен snapshot schema 1. ")+error);return;}
    request("/api/catalogue",{},[this](const QJsonObject &data,const QString &error){if(!error.isEmpty()){setStatus(tr("Ядро недоступно: ")+error);return;}
        m_Region->clear();m_Regions=QJsonObject();
        for(const auto &item:data.value("items").toArray()){const auto r=item.toObject();m_Regions.insert(r.value("id").toString(),r.value("periods"));m_Region->addItem(r.value("title").toString(),r.value("id"));}
        m_Region->setCurrentIndex(std::max(0,m_Region->findData(m_Base.value("region").toString("petersburg-2025"))));
        if(m_Base.contains("period"))m_Period->setCurrentText(m_Base.value("period").toString());
        setStatus(tr("Готово. Направление, место и UTC берутся из KStars."));},false);
    },false);
}
void LightPollutionDock::calculate()
{
    if(m_Busy)return;
    if(m_Region->count()==0){fetchCatalogue();return;}
    const auto frame=snapshot();const double alt=frame.value("focus").toObject().value("altitude_deg").toDouble();
    if(alt<10 || alt>90){setStatus(tr("Количественный режим поддерживает высоты 10–90°. Переместите центр карты."));return;}
    m_SubmittedKey=contextKey();m_Busy=true;m_CancelSent=false;m_Job.clear();m_OpenEye->setEnabled(false);m_EyePreviewReady=false;setStatus(tr("Расчёт снимка состояния KStars…"));
    request("/api/kstars/jobs",{{"scene",scene()},{"snapshot",frame}},[this](const QJsonObject &data,const QString &error){if(!error.isEmpty() || data.value("id").toString().isEmpty()){m_Busy=false;setStatus(error.isEmpty()?tr("Ядро не вернуло идентификатор задания."):error);return;}m_Job=data.value("id").toString();poll();});
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
    m_OpenEye->setEnabled(value.value("eye").toObject().value("display").toObject().value("available").toBool());
    m_LastResult=value;const auto rows=value.value("sky").toObject().value("directions").toArray();if(rows.isEmpty())return;const auto d=rows.first().toObject();
    const QList<QPair<QString,QString>> fields={{tr("Город, B/Bref"),"artificial_relative"},{tr("Прямой городской"),"artificial_direct_relative"},{tr("Отражённый городской"),"artificial_reflected_relative"},{tr("Луна, B/Bref"),"moon_relative"},{tr("Ночной естественный"),"natural_relative"},{tr("Сумерки, B/Bref"),"twilight_relative"},{tr("Сумма, V mag/arcsec²"),"total_v_mag_arcsec2"}};
    m_Values->setRowCount(fields.size());for(int i=0;i<fields.size();++i){m_Values->setItem(i,0,new QTableWidgetItem(fields[i].first));const auto v=d.value(fields[i].second);m_Values->setItem(i,1,new QTableWidgetItem(v.isDouble()?QString::number(v.toDouble(),'g',7):QString::fromUtf8("—")));}
    m_Scale->setText(tr("Масштаб: ")+value.value("calibration").toObject().value("origin").toString()+tr(". Высота KStars сохранена как справочная; AGL задаётся отдельно. Для глаза доступен порог полной адаптации; экранный вид и сенсор не калиброваны."));
    setStatus(m_SubmittedKey!=contextKey()?tr("Получен предыдущий снимок; карта уже изменилась."):(d.value("supported").toBool()?tr("Готово для снимка состояния: ")+value.value("request").toObject().value("time").toString():tr("Сумма недоступна: ")+d.value("status").toString()));
}
