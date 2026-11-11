// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <QDockWidget>
#include <QPointer>
#include <QJsonObject>
#include <QNetworkAccessManager>
#include <QTimer>
#include <functional>

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QLabel;
class QPushButton;
class QTableWidget;
class KStars;
class QDialog;

/** Native KStars dock; physical calculations stay in the versioned local backend. */
class LightPollutionDock : public QDockWidget
{
    Q_OBJECT
    Q_CLASSINFO("D-Bus Interface", "org.lightpollution.KStars")
public:
    explicit LightPollutionDock(KStars *parent);
public slots:
    Q_SCRIPTABLE QString snapshotJSON() const;
    Q_SCRIPTABLE QString statusJSON() const;
    Q_SCRIPTABLE bool setSceneJSON(const QString &json);
    Q_SCRIPTABLE void calculate();
    Q_SCRIPTABLE bool savePanelScreenshot(const QString &path);
    Q_SCRIPTABLE bool saveWindowScreenshot(const QString &path);
    Q_SCRIPTABLE bool saveEyeScreenshot(const QString &path);
    Q_SCRIPTABLE void showEyeView();
private:
    QJsonObject snapshot() const;
    QJsonObject scene() const;
    QString contextKey() const;
    void tick();
    void poll();
    void showResult(const QJsonObject &value);
    void fetchCatalogue();
    void setStatus(const QString &message);
    void request(const QString &path, const QJsonObject &payload,
                 std::function<void(const QJsonObject &, const QString &)> callback, bool post=true);
    KStars *m_KStars;
    QNetworkAccessManager m_Network;
    QTimer m_Timer;
    QComboBox *m_Region, *m_Period, *m_Terrain;
    QDoubleSpinBox *m_Height, *m_AOD;
    QCheckBox *m_Near, *m_Follow, *m_Spectral, *m_Eye, *m_Colour;
    QPushButton *m_OpenEye;
    QLabel *m_Position, *m_Status, *m_Scale;
    QTableWidget *m_Values;
    QJsonObject m_Base, m_LastResult, m_Regions;
    QString m_URL, m_Job, m_SubmittedKey, m_CurrentKey;
    bool m_Busy=false, m_PollPending=false, m_CancelSent=false;
    int m_StableTicks=0;
    bool m_EyePreviewReady=false;
    QPointer<QDialog> m_EyeDialog;
};
