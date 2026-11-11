'use strict';
const $ = id => document.getElementById(id);
const regions = JSON.parse($('regions-data').textContent);
const defaults = {reference_wavelength_nm:550,rayleigh_scale_height_m:8000,aerosol_scale_height_m:1200,aerosol_optical_depth:.1,aerosol_single_scattering_albedo:.9,aerosol_g:.8,rayleigh_multiplier:1,top_m:80000};
let active = null, jobs = [], referenceAtmosphere = {}, currentResult = null;
let photometricCalibration = null, anchorObserver = null, referenceMix = {'LED-B3':1};
function calibrationNote(){
  $('calibration-note').textContent=photometricCalibration?`Калибровка ${photometricCalibration.id.slice(0,12)} · контрольных ночей ${photometricCalibration.validation?.nights??'—'}. Область применения проверяется при расчёте.`:'Калибровка по измерениям не подключена.';
  $('anchor').disabled=!!photometricCalibration;
}
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num = (value,d=3) => value === null || value === undefined ? '—' : Number(value).toLocaleString('ru-RU',{maximumFractionDigits:d});
const supportLabel = status => ({ok:'Расчёт доступен',paranal_empirical_scenario_not_local_validation:'Сценарий Paranal; местной проверки нет',twilight_solar_domain_gap:'Высота Солнца вне модели сумерек',twilight_outside_zenith_region:'Сумерки: взгляд ниже 50°',twilight_disabled:'Модель сумерек выключена',moon_separation_below_10deg:'Ближе 10° к Луне',possible_lunar_eclipse:'Возможное лунное затмение',multiple_model_restrictions:'Несколько ограничений моделей'}[status]||status);
const states = {queued:'В очереди',running:'Выполняется',completed:'Готово',cancelled:'Отменено',error:'Ошибка'};
function message(text) { $('message').textContent=text; $('message').className='error'; $('message').hidden=!text; }
async function api(path,body) {
  const response=await fetch(path,body === undefined ? {} : {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const data=await response.json();
  if (!response.ok) throw new Error(data.error?.message || 'Не удалось выполнить запрос.');
  return data;
}
function entry(){return regions.find(r=>r.id===$('region').value);}
function selectRegion(reset=true){
  const r=entry(); if(!r)return;
  $('period').innerHTML=r.periods.map(p=>`<option value="${esc(p)}">${esc(p==='synthetic'?'Учебные источники':p)}</option>`).join('');
  $('region-note').textContent=r.description+(r.kind==='synthetic'?'':` Локально ${r.local_files_present}/${r.required_files} файлов.`);
  if(reset){$('latitude').value=r.observer.latitude_deg;$('longitude').value=r.observer.longitude_deg;$('height').value=r.observer.height_m;$('time').value=r.default_time.replace('Z','');$('offset').value='Z';photometricCalibration=null;anchorObserver=null;calibrationNote();}
}
function readForm(){
  const atmosphere={reference_wavelength_nm:550};
  document.querySelectorAll('[data-atmo]').forEach(el=>atmosphere[el.dataset.atmo]=Number(el.value));
  const data={schema_version:2,region:$('region').value,period:$('period').value,mode:$('mode').value,
    observer:{latitude_deg:Number($('latitude').value),longitude_deg:Number($('longitude').value),height_m:Number($('height').value)},
    directions:$('directions').value.trim().split(/\n+/).map(line=>line.trim().split(/[,;\s]+/).map(Number)),
    atmosphere,reference_atmosphere:referenceAtmosphere,profile:$('profile').value,
    anchor_v_mag_arcsec2:Number($('anchor').value),natural_v_mag_arcsec2:Number($('natural').value),download_missing:$('download-missing').checked};
  if(data.mode==='dated')data.time=$('time').value+$('offset').value;else data.moon={mode:'none'};
  if(photometricCalibration)data.photometric_calibration=photometricCalibration;
  if(anchorObserver)data.anchor_observer=anchorObserver;
  if($('spectrum-mode').value==='on')data.spectrum={mix:JSON.parse($('spectral-mix').value),reference_mix:referenceMix,angstrom_exponent:Number($('angstrom').value),stars:$('spectral-stars').value};
  if($('background-mode').value!=='legacy')data.background={mode:$('background-mode').value,airglow_fraction:Number($('airglow-fraction').value),airglow_height_m:Number($('airglow-height').value),twilight:$('twilight-mode').value};
  if($('surface-enabled').checked)data.surface={snow_fraction:Number($('snow-fraction').value),bare_albedo:Number($('bare-albedo').value),snow_albedo:Number($('snow-albedo').value),direct_upward_fraction:Number($('direct-upward').value),direct_horizontal_fraction:Number($('direct-horizontal').value)};
  return data;
}
function fill(data){
  if(data.schema_version!==2)throw new Error('Откройте scene.json планировщика схемы 2. Старые сцены схемы 1 доступны через observe.');
  if(!regions.some(r=>r.id===data.region))throw new Error('Этот регион отсутствует в локальном каталоге.');
  $('region').value=data.region;selectRegion(false);$('period').value=data.period;
  $('latitude').value=data.observer.latitude_deg;$('longitude').value=data.observer.longitude_deg;$('height').value=data.observer.height_m;
  $('mode').value=data.mode||'dated';$('dated-controls').hidden=$('mode').value==='undated';
  if(data.time){const instant=new Date(data.time);if(!Number.isNaN(instant.valueOf()))$('time').value=instant.toISOString().slice(0,19);$('offset').value='Z';}
  $('directions').value=(data.directions||[[0,90],[0,45],[90,45],[180,45],[270,45]]).map(v=>v.join(',')).join('\n');
  document.querySelectorAll('[data-atmo]').forEach(el=>el.value=data.atmosphere?.[el.dataset.atmo]??defaults[el.dataset.atmo]);
  referenceAtmosphere=data.reference_atmosphere||{};$('anchor').value=data.anchor_v_mag_arcsec2??20.5;$('natural').value=data.natural_v_mag_arcsec2??21.6;
  $('profile').value=data.profile||'standard';$('download-missing').checked=!!data.download_missing;
  photometricCalibration=data.photometric_calibration||null;anchorObserver=data.anchor_observer||null;calibrationNote();
  $('spectrum-mode').value=data.spectrum?'on':'off';$('spectral-mix').value=JSON.stringify(data.spectrum?.mix||{'LED-B3':1});
  referenceMix=data.spectrum?.reference_mix||{'LED-B3':1};$('angstrom').value=data.spectrum?.angstrom_exponent??1.3;$('spectral-stars').value=data.spectrum?.stars||'catalogue_continuum';
  $('spectral-reference-note').textContent=`Опорная смесь ${JSON.stringify(referenceMix)}. Смена текущей смеси не восстанавливает прежнюю V-яркость.`;
  $('background-mode').value=data.background?.mode||'legacy';$('airglow-fraction').value=data.background?.airglow_fraction??.6;$('airglow-height').value=data.background?.airglow_height_m??90000;$('twilight-mode').value=data.background?.twilight||'none';
  $('surface-enabled').checked=!!data.surface;$('snow-fraction').value=data.surface?.snow_fraction??0;$('bare-albedo').value=data.surface?.bare_albedo??.15;$('snow-albedo').value=data.surface?.snow_albedo??.8;$('direct-upward').value=data.surface?.direct_upward_fraction??.15;$('direct-horizontal').value=data.surface?.direct_horizontal_fraction??0;
}
function labelJob(job){return `${job.request.time||'Без даты'} · ${job.request.period} · AOD ${num(job.request.atmosphere?.aerosol_optical_depth??.1,2)}`;}
async function refreshJobs(){
  jobs=await api('/api/jobs');
  $('history').innerHTML=jobs.length?jobs.slice(0,12).map(j=>`<div class="history-row"><div>${esc(regions.find(r=>r.id===j.request.region)?.title||j.request.region)}<small>${esc(labelJob(j))} · ${esc(states[j.state])}</small></div><button class="secondary" data-job="${j.id}">${j.state==='completed'?'Открыть':'Статус'}</button></div>`).join(''):'Пока нет расчётов';
  $('history').querySelectorAll('[data-job]').forEach(b=>b.onclick=()=>openJob(b.dataset.job).catch(e=>message(e.message)));
  const complete=jobs.filter(j=>j.state==='completed');
  for(const name of ['compare-left','compare-right']){const selected=$(name).value;$(name).innerHTML=complete.map(j=>`<option value="${j.id}">${esc(labelJob(j))}</option>`).join('');if(complete.some(j=>j.id===selected))$(name).value=selected;}
  if(complete.length>1&&$('compare-left').value===$('compare-right').value)$('compare-right').value=complete.find(j=>j.id!==$('compare-left').value).id;
  $('compare').disabled=complete.length<2;
}
function resultHTML(value,id){
  const rows=value.sky.directions,first=rows[0],ob=value.observation,qa=value.data_provenance.QA;
  const metric=(title,v,unit)=>`<div class="metric"><label>${title}</label><strong>${num(v)}</strong><small>${unit}</small></div>`;
  const directions=rows.map(r=>`<tr><td>${num(r.azimuth_deg,1)}° / ${num(r.altitude_deg,1)}°</td><td>${num(r.artificial_relative)}</td><td>${num(r.moon_relative)}</td><td>${num(r.natural_relative)}</td><td>${num(r.twilight_relative??(r.background_supported===false?null:0))}</td><td>${num(r.total_v_mag_arcsec2)}</td><td>${esc(supportLabel(r.status))}</td></tr>`).join('');
  const stars=[...value.stars.rows].sort((a,b)=>a.observed_v_mag-b.observed_v_mag).slice(0,40);
  const stellar=stars.map(s=>`<tr><td>${esc(s.name||'HR '+s.hr)}</td><td>${num(s.azimuth_deg,1)}°</td><td>${num(s.altitude_deg,1)}°</td><td>${num(s.observed_v_mag,2)}</td></tr>`).join('');
  const moon=ob.moon,geometry=ob.geometry;
  const context=geometry?`Солнце ${num(geometry.sun.altitude_deg,1)}° · Луна ${num(moon.altitude_deg,1)}° · освещено ${num(100*moon.illuminated_fraction,1)}%`:'Без даты: положение звёзд не назначено, Луна явно отключена';
  const qaText=qa.common_pixels!==undefined?`Общая QA-область: ${num(qa.common_pixels,0)} / ${num(qa.reference_land_domain_pixels,0)} (${num(100*qa.common_fraction_of_reference_land,2)}%).`:'Учебные искусственные источники. Спутниковые измерения не использованы.';
  const calibrationText=value.calibration.origin==='provided_calibration'?`Подключена калибровка по измерениям ${esc(value.calibration.provided_calibration_id?.slice(0,12))}. Её контроль ограничен указанными ночами и площадками.`:`Общая сценарная опора: искусственный зенит ${num(value.calibration.anchor_v_mag_arcsec2,1)} V mag/arcsec² в фиксированной точке ${num(value.calibration.anchor_observer?.latitude_deg,3)}°, ${num(value.calibration.anchor_observer?.longitude_deg,3)}°. Это не измеренная яркость данной ночи.`;
  const spectral=value.spectral?`<h3>Спектральный отклик города</h3><p class="hint">Состав ламп: ${esc(JSON.stringify(value.spectral.source_mix))}. Отражение поверхности при включении имеет серое альбедо; рельеф и спектры Луны/естественного фона не восстановлены.</p><div class="table-wrap"><table><thead><tr><th>Направление</th><th>Дневной отклик, мккд/м²</th><th>Ночной отклик, мккд/м²</th><th>S/P</th></tr></thead><tbody>${value.spectral.directions.map(r=>`<tr><td>${num(r.azimuth_deg,1)}° / ${num(r.altitude_deg,1)}°</td><td>${num(r.photopic_luminance_cd_m2*1e6,2)}</td><td>${num(r.scotopic_luminance_cd_m2*1e6,2)}</td><td>${num(r.scotopic_photopic_ratio,3)}</td></tr>`).join('')}</tbody></table></div><p class="hint">Это взвешенные световые величины; адаптация и обнаружение глазом ещё не рассчитываются. Спектры звёзд: измеренные ${num(value.spectral.stellar_summary.measured,0)}, континуум по B−V ${num(value.spectral.stellar_summary.catalogue_continuum,0)}, недоступны ${num(value.spectral.stellar_summary.unavailable,0)}.</p>`:'';
  const surface=value.observation.surface_model;const surfaceText=surface?`Сценарий снега: покрытие ${num(100*surface.config.snow_fraction,1)}% освещённой поверхности источников, альбедо ${num(surface.balance.effective_albedo,3)}. Уходящая вверх мощность ×${num(surface.balance.upward_power_over_snow_free,3)} при тех же лампах; усиление фона зависит также от направления.`:'Разделение прямого/отражённого света не включено.';
  return `<section class="panel"><div class="section-title"><div><span class="eyebrow">РЕЗУЛЬТАТ СЦЕНАРИЯ</span><h2>${esc(value.data_provenance.region.title)}</h2></div><button id="repeat" class="secondary">Изменить сценарий</button></div>
    <div class="meta"><span>${esc(value.request.time||'Условный безлунный сценарий')}</span><span>Источники: ${esc(value.request.period)}</span><span class="tag">${num(value.runtime.elapsed_seconds,2)} с</span>${value.cache.scene_hit?'<span class="tag">Из кэша</span>':''}</div>
    <div class="scenario-note">${calibrationText} Мощность источников при смене точки и атмосферы фиксирована.</div>
    <p class="hint">${esc(context)}</p>
    <div class="metrics">${metric('Суммарный фон · первое направление',first.total_v_mag_arcsec2,'V mag/arcsec² · меньше число — ярче')}${metric('Город · первое направление',first.artificial_relative,'B/Bref · опора V=22')}${metric('Луна · первое направление',first.moon_relative,'B/Bref · та же опора')}</div>
    <h3>Фон по направлениям</h3><div class="table-wrap"><table><thead><tr><th>Азимут / высота</th><th>Город, B/Bref</th><th>Луна, B/Bref</th><th>Ночной естеств., B/Bref</th><th>Сумерки, B/Bref</th><th>Сумма, V mag/arcsec²</th><th>Статус</th></tr></thead><tbody>${directions}</tbody></table></div>
    <p class="hint">${esc(qaText)} Источники ближе 3 км и за пределами региона не включены; рельеф не восстановлен. ${esc(surfaceText)}</p>
    ${spectral}<details><summary>Каталожные звёзды (${num(value.stars.rows.length,0)})</summary><p class="hint">Показаны до 40 ярчайших звёзд над нижней границей поля 10°. Это геометрия и экстинкция; обнаружение глазом ещё не оценивается.</p><div class="table-wrap"><table><thead><tr><th>Звезда</th><th>Азимут</th><th>Высота</th><th>V после ослабления</th></tr></thead><tbody>${stellar}</tbody></table></div></details>
    <div class="exports">${[['scene.json','Сохранить сценарий'],['bundle.zip','Все результаты ZIP'],['directions.csv','Фон CSV'],['stars.csv','Звёзды CSV'],['result.json','Полный JSON']].map(([f,l])=>`<a href="/api/jobs/${id}/export/${f}">${l}</a>`).join('')}</div></section>`;
}
async function openJob(id){
  const job=await api('/api/jobs/'+id);
  if(job.state==='completed'){
    currentResult=await api(`/api/jobs/${id}/result`);$('empty').hidden=true;$('result').hidden=false;$('result').innerHTML=resultHTML(currentResult,id);
    $('repeat').onclick=()=>{fill(currentResult.request);window.scrollTo({top:0,behavior:'smooth'});};
  }else if(job.state==='error'){message(job.error?.message||'Расчёт не завершён.');}
  else if(job.state==='cancelled'){message('Расчёт отменён. Завершённые численные блоки можно использовать при повторе.');}
  else {active=id;poll(id);}
}
async function poll(id){
  try{
    $('progress-panel').hidden=false;$('cancel').disabled=false;
    while(active===id){
      const job=await api('/api/jobs/'+id),p=job.progress||{};
      $('phase').textContent=p.phase||states[job.state];
      $('progress-count').textContent=p.total?`${p.completed} / ${p.total}`:states[job.state];
      if(p.total){$('progress').max=p.total;$('progress').value=p.completed;}else $('progress').removeAttribute('value');
      if(['completed','cancelled','error'].includes(job.state)){
        active=null;$('cancel').disabled=true;$('calculate').disabled=false;
        await refreshJobs();await openJob(id);break;
      }
      await new Promise(resolve=>setTimeout(resolve,700));
    }
  }catch(e){active=null;$('calculate').disabled=false;$('cancel').disabled=true;message(e.message);}
}
$('region').innerHTML=regions.map(r=>`<option value="${r.id}">${esc(r.title)}</option>`).join('');selectRegion();
$('region').onchange=()=>selectRegion();$('mode').onchange=()=>$('dated-controls').hidden=$('mode').value==='undated';
$('scenario').onsubmit=async e=>{e.preventDefault();message('');$('calculate').disabled=true;try{const job=await api('/api/jobs',readForm());active=job.id;await refreshJobs();poll(job.id);}catch(err){$('calculate').disabled=false;message(err.message);}};
$('cancel').onclick=async()=>{if(active){await api('/api/jobs/'+active+'/cancel',{});$('cancel').disabled=true;}};
$('refresh').onclick=()=>refreshJobs().catch(e=>message(e.message));
$('import').onchange=async e=>{try{fill(JSON.parse(await e.target.files[0].text()));message('');}catch(err){message(err.message);}e.target.value='';};
$('calibration-import').onchange=async e=>{try{const v=JSON.parse(await e.target.files[0].text());if(v.schema_version!==1||!v.id||!v.scope||!(v.source_coefficient>0))throw new Error('Нужен файл fit-photometry.');photometricCalibration=v;calibrationNote();message('');}catch(err){message(err.message);}e.target.value='';};
$('calibration-clear').onclick=()=>{photometricCalibration=null;calibrationNote();};
$('compare').onclick=async()=>{message('');try{const r=await api('/api/compare',{left:$('compare-left').value,right:$('compare-right').value});$('comparison').innerHTML=`<div class="table-wrap"><table><thead><tr><th>Направление</th><th>Яркость справа / слева</th><th>Разность mag справа − слева</th></tr></thead><tbody>${r.directions.map(v=>`<tr><td>${num(v.azimuth_deg,1)}° / ${num(v.altitude_deg,1)}°</td><td>${num(v.right_over_left,4)}</td><td>${num(v.right_minus_left_mag,4)}</td></tr>`).join('')}</tbody></table></div><p class="hint">Общая сценарная опора. Это изменение расчёта при выбранных условиях, не подтверждённая точность.</p>`;}catch(err){message(err.message);}};
refreshJobs().then(()=>{const last=jobs.find(j=>j.state==='completed');if(last)return openJob(last.id);}).catch(e=>message(e.message));
