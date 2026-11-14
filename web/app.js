'use strict';
const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const paths = {
  grid:'<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/>',
  layers:'<path d="m12 3 9 5-9 5-9-5 9-5Z"/><path d="m3 12 9 5 9-5M3 16l9 5 9-5"/>',
  spark:'<path d="m12 3 2.5 6.5L21 12l-6.5 2.5L12 21l-2.5-6.5L3 12l6.5-2.5L12 3Z"/>',
  database:'<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v7c0 4 16 4 16 0V5M4 12v7c0 4 16 4 16 0v-7"/>',
  settings:'<path d="m10 3-1 3-3 1-3 3 2 2-1 4 4 1 2 4 3-1 3-3 4-1-1-4 2-3-3-2-2-3-3 1-3-1Z"/><circle cx="12" cy="12" r="3"/>',
  plus:'<path d="M12 5v14M5 12h14"/>',
  flask:'<path d="M9 3h6M10 3v6L4 19c-.6 1 .2 2 1 2h14c1 0 1.6-1 1-2L14 9V3M7 15h10"/>',
  gauge:'<path d="M4 19a10 10 0 1 1 16 0M12 12l5-5M5 12H3M12 3v2M19 12h2"/><circle cx="12" cy="12" r="2"/>',
  target:'<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1"/>',
  coin:'<ellipse cx="12" cy="12" rx="8" ry="9"/><path d="M12 7v10M9 9c1-2 6-1 5 1-1 2-5 1-4 4 1 1 4 1 5-1"/>',
  download:'<path d="M12 3v12m-5-5 5 5 5-5M4 16v5h16v-5"/>',
  info:'<circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7v.1"/>',
  compare:'<path d="M5 4v16M19 4v16M5 7h10m-3-3 3 3-3 3M19 17H9m3-3-3 3 3 3"/>',
  clock:'<circle cx="12" cy="12" r="9"/><path d="M12 6v6l4 2"/>',
  menu:'<path d="M4 7h16M4 12h16M4 17h16"/>',
  archive:'<path d="M4 5h16v4H4zM6 9v10h12V9M10 13h4"/>',
  trash:'<path d="M5 7h14M9 7V5h6v2M7 7l1 13h8l1-13M10 11v6M14 11v6"/>',
  restore:'<path d="M4 12a8 8 0 1 0 3-6.2M4 4v5h5"/>'
};
function icon(name) { return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${paths[name] || paths.info}</svg>`; }
$$('[data-icon]').forEach(el => el.innerHTML = icon(el.dataset.icon));
const statuses = {queued:'等待中',running:'研究中',completed:'已完成',failed:'调用失败',cancelled:'已停止',budget_stopped:'预算停止',interrupted:'服务重启中断'};
let state = null, run = null, selectedId = localStorage.getItem('rl-selected-run') || '', page = 'workspace';
let evidenceExperiment = '', evidenceClaim = '', evidenceKey = '', lastRunStamp = '', refreshing = false, lastBudgetStamp = 0;
let modelStatus = {available:false,models:[]};
const percent = n => n == null ? '—' : `${(n*100).toFixed(1)}%`;
const number = n => new Intl.NumberFormat('en-US').format(n || 0);
const time = n => n >= 60 ? `${Math.floor(n/60)}m ${Math.round(n%60)}s` : `${Math.round(n || 0)}s`;
const delta = n => `${n >= 0 ? '+' : ''}${(n*100).toFixed(1)} pp`;
const operatorTask = name => name === 'win_bias_mean' || name === 'lose_gelu_pool';
const modeLabel = r => r.domain === 'operator' ? '旧版算子研究' : '科研 Agent';

async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || `请求失败 ${response.status}`);
  return result;
}
let toastTimer;
function toast(message) {
  $('#toast').textContent = message; $('#toast').classList.add('visible');
  clearTimeout(toastTimer); toastTimer = setTimeout(() => $('#toast').classList.remove('visible'), 4800);
}
function empty(title, message, mark='↻') { return `<div class="empty-state"><div class="empty-mark">${mark}</div><strong>${esc(title)}</strong><p>${esc(message)}</p></div>`; }
function goto(name) {
  page = ['workspace','session','memory','data','archive'].includes(name) ? name : 'workspace';
  $$('.page').forEach(el => el.classList.toggle('hidden',el.id !== `page-${page}`));
  $$('.nav-item').forEach(el => el.classList.toggle('active',el.dataset.page === page));
  $('#page-label').textContent = {workspace:'研究工作台',session:'研究对话',memory:'记忆',data:'材料与工具',archive:'归档会话'}[page];
  history.replaceState(null,'',`#${page}`);
}
$$('[data-page]').forEach(el => el.addEventListener('click',() => goto(el.dataset.page)));
$$('[data-goto]').forEach(el => el.addEventListener('click',() => goto(el.dataset.goto)));
window.addEventListener('hashchange',() => goto(location.hash.slice(1)));
$$('.close-dialog').forEach(el => el.addEventListener('click',() => el.closest('dialog').close()));
$$('dialog').forEach(dialog => dialog.addEventListener('click',e => {if (e.target === dialog) {const r=dialog.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)dialog.close();}}));

async function refresh(force=false) {
  if (refreshing) return;
  refreshing = true;
  try {
    state = await api('/api/state');
    if (!state.runs.some(r => r.id === selectedId)) selectedId = state.active_run || state.runs.find(r=>r.domain==='research')?.id || '';
    const oldOptions = $('#run-select').innerHTML;
    const options = state.runs.length ? state.runs.map(r => `<option value="${esc(r.id)}">${esc(r.task)} · ${esc(modeLabel(r))} · ${esc(r.id.slice(9,15))}</option>`).join('') : '<option value="">暂无运行</option>';
    if (oldOptions !== options) $('#run-select').innerHTML = options;
    $('#run-select').value = selectedId;
    const summary = state.runs.find(r => r.id === selectedId);
    if (selectedId && (force || !run || run.id !== selectedId || summary?.status === 'running' || summary?.status === 'queued' || run.status !== summary?.status)) {
      run = await api(`/api/runs/${selectedId}`); lastBudgetStamp = Date.now();
    }
    if (!selectedId) run = null;
    renderTop(); renderMetrics(); renderMemory(); renderData();
      const stamp = run ? `${run.id}:${run.updated_at}` : 'empty';
    if (force || stamp !== lastRunStamp) { lastRunStamp = stamp; renderRun(); }
    renderBudget();
  } catch(error) { $('#notice').textContent = `本地服务连接异常：${error.message}`; $('#notice').classList.remove('hidden'); }
  finally { refreshing=false; }
}
function renderTop() {
  const notice=run?.error||run?.stop_reason||'';
  $('#notice').textContent=notice;$('#notice').classList.toggle('hidden',!notice);
  $$('.new-run').forEach(el=>el.disabled=!!state.active_run);
}
function metric(label,value,unit,bottom,which='target',extra='') {
  return `<article class="metric-card"><div class="metric-top"><span>${label}</span><span class="metric-icon">${icon(which)}</span></div><div class="metric-value">${value}<small>${unit}</small></div><div class="metric-bottom">${extra}${bottom}</div></article>`;
}
function renderMetrics() {
  if (run?.domain === 'operator') {
    const speed = run.operator_experiments?.length ? `${Number(run.verdict?.best_speedup || 0).toFixed(2)}×` : '—';
    $('#metrics').innerHTML = metric('同窗加速比', speed, '', run.verdict?.reason_code || run.skip_reason || '等待裁决', 'target') +
      metric('本轮模型 token', '0', '', '算子研究不调用模型', 'coin') +
      metric('已完成候选实验', String(run.operator_experiments?.length || 0), run ? ` / ${run.max_experiments}` : '', '研究层决定是否做实验', 'flask') +
      metric('方法记忆', `v${state.research_memory.active}`, '', '记忆页面可查看来源与版本', 'spark');
    return;
  }
  $('#metrics').innerHTML='';
}
function renderRun(){if(run?.domain==='operator')renderOperatorRun();}
function renderOperatorRun() {
  const active=['running','queued'].includes(run.status);
  const exps=run.operator_experiments || [];
  const reason=run.verdict?.reason_code || run.skip_reason || (active ? '正在测量' : '尚无裁决');
  $('#diagnostics-panel').classList.add('hidden');
  $('#experiment-counter').textContent = `${exps.length} 个实验`;
  $('#run-meta').innerHTML=`<span class="status-dot ${active?'running':run.status==='failed'?'failed':''}"></span><span>${esc(statuses[run.status])}</span><span>·</span><span>算子研究</span><span>·</span><span>任务 ${esc(run.task)}</span><span class="spacer"></span><button id="show-calls">裁决记录</button>`;
  $('#show-calls').addEventListener('click',()=>showDetail('算子裁决', {verdict:run.verdict, skip_reason:run.skip_reason, events:run.events}));
  const brief=run.brief?.question?`<div class="hypothesis-source"><strong>这次的问题</strong><br>${esc(run.brief.question)}<br>${esc(run.brief.falsify_if||'')}</div>`:'';
  $('#hypothesis').innerHTML=`<span class="hypothesis-tag">算子研究</span><div class="hypothesis-text">${esc(run.verdict?.detail || run.skip_reason || run.brief?.question || '研究层正在判断这条结构值不值得做实验。')}</div>${brief}<div class="hypothesis-source">${esc(run.verdict?.next_action || '正确性和同窗计时由门控完成，不调用 Ollama。')}</div>`;
  if (!exps.length) {
    $('#chart').innerHTML=`<div class="chart-empty"><span>${esc(reason)}</span></div>`;
    $('#experiments').innerHTML=empty('没有进入实验', '证据层在实验前判定这条结构赢不了。');
  } else {
    $('#chart').innerHTML=`<div class="chart-empty"><span>同窗加速比 ${Number(run.verdict?.best_speedup || 0).toFixed(2)}×</span></div>`;
    $('#experiments').innerHTML=exps.map(e=>`<div class="experiment-row"><div class="experiment-number">${esc(e.exp_id)}</div><div><div class="experiment-title">${esc((e.rule_chain || []).join(' → ') || e.family || '候选')}</div><div class="experiment-detail"><span>${esc(e.note || e.origin || '')}</span></div></div><div class="experiment-outcome">${e.speedup_same_window==null?'—':`${Number(e.speedup_same_window).toFixed(2)}×`}<small>${e.target_met?'达到 1.10×':'未达 1.10×'}</small></div></div>`).join('');
  }
  $('#evidence-experiment').innerHTML='';
  $('#evidence-claim').innerHTML='';
  $('#evidence').innerHTML=`<div class="hypothesis-source"><strong>${esc(reason)}</strong><br>${esc(run.verdict?.detail || run.skip_reason || '等待这一轮测量结束。')} 云端调用 0 次。</div>`;
}
function renderBudget(){
  if(run?.domain==='operator')$('#budget').textContent=`算子执行 ${time(run.budget?.elapsed)}，模型调用 0 次`;
}
function renderMemory(){}
function renderData(){}
function showDetail(title,value){$('#detail-title').textContent=title;$('#detail-json').textContent=JSON.stringify(value,null,2);$('#detail-dialog').showModal();}
async function selectRun(id){selectedId=id;localStorage.setItem('rl-selected-run',id);evidenceKey='';evidenceClaim='';evidenceExperiment='';goto('session');await refresh(true);}
$('#run-select').addEventListener('change',e=>selectRun(e.target.value));
$$('.new-run').forEach(el=>el.addEventListener('click',()=>{document.getElementById('composer').scrollIntoView({block:'center'});$('#brief-input').focus();}));
async function checkModels(){const badge=$('#ollama-status');badge.innerHTML='<span class="loader"></span>检查 Ollama…';try{modelStatus=await api('/api/ollama');badge.classList.toggle('offline',!modelStatus.available);badge.innerHTML=`<span class="small-dot"></span>${modelStatus.available?'Ollama 已连接':'Ollama 未连接'}`;$('#model-options').innerHTML=modelStatus.models.filter(m=>!m.name.toLowerCase().includes('cloud')).map(m=>`<option value="${esc(m.name)}">${esc(m.details?.parameter_size||'')}</option>`).join('');const cloud=state?.settings?.cloud||{};$('#model-details').textContent=modelStatus.available?`检测到 ${modelStatus.models.length} 个本地模型。当前本地默认 ${state?.settings.model||'未设置'}。${cloud.enabled?`云端已启用：${cloud.model}`:'云端未启用。'}`:`本地服务未连接：${modelStatus.error}。请运行 ollama serve，再检测服务。`;}catch(e){badge.textContent='连接检测失败';toast(e.message);}}
function fillCloudSettings(){fillRoutingSettings();const c=state?.settings?.cloud||{},w=state?.settings?.web||{},form=$('#settings-form');$('#model-source').value=state?.settings?.model_source||'local';$('#cloud-enabled').checked=!!c.enabled;form.querySelector('[name=cloud_base_url]').value=c.base_url||'';form.querySelector('[name=cloud_model]').value=c.model||'';form.querySelector('[name=cloud_api_key]').value='';$('#cloud-details').textContent=c.api_key_set?'已保存 API Key（页面不回显明文）。':'尚未保存 API Key；启用云端模型前需填写。';$('#web-enabled').checked=!!w.enabled;form.querySelector('[name=web_provider]').value=w.search_provider||'searxng';form.querySelector('[name=web_endpoint]').value=w.search_endpoint||'';form.querySelector('[name=web_api_key]').value='';form.querySelector('[name=web_max_results]').value=w.max_results||5;const lim=state?.settings?.limits||{};form.querySelector('[name=limit_tokens]').value=lim.max_tokens||400000;form.querySelector('[name=limit_seconds]').value=lim.max_seconds||7200;form.querySelector('[name=limit_steps]').value=lim.max_steps||40;describeSearchConfig();}
let roleCatalog=[];
async function renderRoleModels(){const saved=state?.settings?.roles||{};if(!roleCatalog.length){try{const data=await api('/api/roles');roleCatalog=data.roles||[];}catch(e){roleCatalog=[];}}const cloud=state?.settings?.cloud||{};$('#role-models').innerHTML=roleCatalog.map(r=>{const c=saved[r.role]||{};return `<div class="role-row" data-role="${esc(r.role)}"><span>${esc(r.label)}</span><select><option value="">沿用本轮模型</option><option value="local">本地</option><option value="cloud">云端</option></select><input placeholder="${esc(cloud.enabled?(cloud.model||'云端模型'):'模型名')}" value="${esc(c.model||'')}"></div>`;}).join('');document.querySelectorAll('#role-models .role-row').forEach(row=>{const conf=saved[row.dataset.role]||{};row.querySelector('select').value=conf.model_source||'';});}
function openSettings(){if(state){$('#settings-form [name=ollama_url]').value=state.settings.ollama_url;$('#model-name').value=state.settings.model;fillCloudSettings();}$('#settings-dialog').showModal();checkModels();renderRoleModels();}
$('#settings-button').addEventListener('click',openSettings);$('#settings-shortcut').addEventListener('click',openSettings);$('#refresh-models').addEventListener('click',checkModels);
$('#check-cloud').addEventListener('click',async()=>{const form=$('#settings-form'),note=$('#cloud-details');note.textContent='检测已保存的云端配置…';try{const result=await api('/api/cloud/test',{cloud:{base_url:form.querySelector('[name=cloud_base_url]').value,api_key:form.querySelector('[name=cloud_api_key]').value}});note.textContent=result.available?`云端已连接，可用模型 ${result.models.length} 个${result.models[0]?`（如 ${result.models[0].name}）`:''}。`:`云端未连接：${result.error}`;}catch(error){note.textContent='检测失败：'+error.message;}});
$('#check-web').addEventListener('click',async()=>{const form=$('#settings-form'),note=$('#web-details');note.textContent='检测已保存的联网搜索配置…';try{const result=await api('/api/web/test',{query:'research agent evaluation'});note.textContent=`${result.provider} 可用，返回 ${result.results.length} 条结果${result.results[0]?`（如 ${result.results[0].title||result.results[0].url}）`:''}。`;}catch(error){note.textContent='联网检测失败：'+error.message;}});
$('#settings-form').addEventListener('submit',async e=>{e.preventDefault();const raw=Object.fromEntries(new FormData(e.target));const roles={};document.querySelectorAll('#role-models .role-row').forEach(row=>{const source=row.querySelector('select').value;if(source)roles[row.dataset.role]={model_source:source,model:row.querySelector('input').value.trim()};});try{const body={ollama_url:raw.ollama_url,model:raw.model,model_source:raw.model_source,roles,routing:readRoutingSettings(),cloud:{enabled:$('#cloud-enabled').checked,base_url:raw.cloud_base_url,model:raw.cloud_model,api_key:raw.cloud_api_key},web:{enabled:$('#web-enabled').checked,search_provider:raw.web_provider,search_endpoint:raw.web_endpoint,search_api_key:raw.web_api_key,max_results:Number(raw.web_max_results||5),fetch_max_chars:6000},limits:{max_tokens:Number(raw.limit_tokens||400000),max_seconds:Number(raw.limit_seconds||7200),max_steps:Number(raw.limit_steps||40)}};await api('/api/settings',body);$('#settings-dialog').close();toast('设置已保存。');await refresh(true);await checkModels();}catch(error){toast(error.message);}});
$('#draft-brief').addEventListener('click',()=>startGeneralResearch());
$$('[data-prompt]').forEach(el=>el.addEventListener('click',()=>{$('#brief-input').value=el.dataset.prompt;$('#brief-input').focus();}));
$('#brief-input').addEventListener('keydown',event=>{if(event.key==='Enter'&&(event.metaKey||event.ctrlKey))startGeneralResearch();});
goto(location.hash.slice(1)||'workspace');
refresh(true).then(checkModels);
setInterval(()=>refresh(),2200);
setInterval(()=>renderBudget(),1000);

/* Sidebar: collapsible, remembered locally; on small screens it closes after
   navigation so the page is immediately visible. */
const SIDEBAR_KEY='researchloop-sidebar-collapsed';
let sidebarCollapsed=false;
try{sidebarCollapsed=localStorage.getItem(SIDEBAR_KEY)==='1';}catch(_){}
function applySidebar(collapsed){
  document.body.classList.toggle('sidebar-collapsed',collapsed);
  const button=$('#sidebar-toggle');
  if(button){button.setAttribute('aria-expanded',String(!collapsed));button.setAttribute('aria-label',collapsed?'展开侧栏':'收起侧栏');}
}
function setSidebar(collapsed){sidebarCollapsed=collapsed;try{localStorage.setItem(SIDEBAR_KEY,collapsed?'1':'0');}catch(_){}applySidebar(collapsed);}
applySidebar(sidebarCollapsed);
$('#sidebar-toggle')?.addEventListener('click',()=>setSidebar(!sidebarCollapsed));
document.addEventListener('keydown',event=>{if(event.key==='Escape'&&!sidebarCollapsed&&window.matchMedia('(max-width:700px)').matches)setSidebar(true);});
$$('.nav-item').forEach(el=>el.addEventListener('click',()=>{if(window.matchMedia('(max-width:700px)').matches&&!sidebarCollapsed)setSidebar(true);}));

/* Sidebar research archive: one entry per session, like a conversation list. */
function sidebarSessionDate(value){
  const date=new Date(value), now=new Date(), yesterday=new Date(now);
  yesterday.setDate(now.getDate()-1);
  if(Number.isNaN(date.getTime()))return {group:'更早',label:'',full:''};
  const today=date.toDateString()===now.toDateString();
  return {group:today?'今天':date.toDateString()===yesterday.toDateString()?'昨天':'更早',
    label:today?date.toLocaleTimeString('zh-CN',{hour:'2-digit',minute:'2-digit',hour12:false}):date.toLocaleDateString('zh-CN',{month:'2-digit',day:'2-digit'}),
    full:date.toLocaleString('zh-CN',{hour12:false})};
}
function renderSidebarSessions(){
  const box=$('#sidebar-sessions');if(!box)return;
  const runs=(state?.runs||[]).filter(r=>r.domain==='research'&&!r.archived);
  const labels={queued:'等待中',running:'研究中',pausing:'正在暂停',paused:'已暂停',completed:'已完成',inconclusive:'证据不足',incomplete:'未完成',budget_stopped:'预算耗尽',cancelled:'已停止',interrupted:'已中断',failed:'执行失败'};
  const count=$('#sidebar-session-count');if(count){count.textContent=runs.length;count.title=`${runs.length} 个未归档研究会话`;}
  let group='';
  const content=runs.length?runs.map(r=>{
    const date=sidebarSessionDate(r.created_at),active=r.id===selectedId&&page==='session';
    const title=String(r.question||r.task||'未命名研究').trim(),status=labels[r.status]||'状态未知';
    const heading=group!==date.group?`<div class="session-group" role="presentation">${esc(date.group)}</div>`:'';group=date.group;
    return `${heading}<div class="session-row${active?' active':''}" role="listitem"><button type="button" class="session-link${active?' active':''}" data-session="${esc(r.id)}" ${active?'aria-current="page"':''} title="${esc(title)}" aria-label="${esc(title)}，${esc(status)}"><span class="session-name">${esc(title)}</span><span class="session-meta"><span class="session-state"><span class="session-dot ${esc(r.status)}" aria-hidden="true"></span>${esc(status)}</span><time datetime="${esc(r.created_at||'')}" title="${esc(date.full)}">${esc(date.label)}</time></span></button><button type="button" class="session-archive" data-archive="${esc(r.id)}" title="归档会话" aria-label="归档：${esc(title)}">${icon('archive')}</button></div>`;
  }).join(''):`<div class="session-empty"><span class="session-empty-icon">${icon('layers')}</span><strong>还没有研究记录</strong><span>从一个问题开始新的研究</span></div>`;
  if(box.dataset.content===content)return;
  const scroll=box.scrollTop,focused=document.activeElement;
  const focusKey=box.contains(focused)?{session:focused.dataset.session,archive:focused.dataset.archive}:null;
  box.innerHTML=content;box.dataset.content=content;box.scrollTop=scroll;
  $$('[data-session]').forEach(el=>el.onclick=()=>selectRun(el.dataset.session));
  $$('[data-archive]').forEach(button=>button.onclick=async event=>{
    event.stopPropagation();
    try{await api(`/api/runs/${button.dataset.archive}/archive`,{archived:true});toast('会话已归档，可在“归档”页恢复');await refresh(true);}
    catch(error){toast(error.message);}
  });
  if(focusKey){const target=$$('[data-session], [data-archive]').find(el=>focusKey.session?el.dataset.session===focusKey.session:el.dataset.archive===focusKey.archive);target?.focus({preventScroll:true});}
}
renderSidebarSessions();
const baseRefresh=refresh;
refresh=async function(force=false){const result=await baseRefresh(force);renderSidebarSessions();return result;};
$('#session-back')?.addEventListener('click',()=>goto('workspace'));

statuses.inconclusive='证据不足';

/* Archive page: open, restore or delete archived sessions. */
function renderArchive(){
  const box=$('#archive-list');
  if(!box)return;
  const runs=(state?.runs||[]).filter(r=>r.domain==='research'&&r.archived);
  box.innerHTML=runs.length?runs.map(r=>`<article class="archive-card"><div class="archive-main"><span class="session-dot ${esc(r.status)}"></span><strong>${esc(r.task||'未命名研究')}</strong><small>${esc((r.archived_at||'').slice(0,16).replace('T',' '))} 归档 · ${esc(statuses[r.status]||r.status)}</small></div><div class="archive-actions"><button type="button" class="secondary-button" data-open="${esc(r.id)}">打开</button><button type="button" class="secondary-button" data-restore="${esc(r.id)}">恢复</button><button type="button" class="secondary-button danger" data-delete="${esc(r.id)}">删除</button></div></article>`).join(''):empty('没有归档会话','在会话页点“归档”后，这里会出现；删除会移到 runs/.trash。');
  $$('[data-open]').forEach(el=>el.onclick=()=>selectRun(el.dataset.open));
  $$('[data-restore]').forEach(el=>el.onclick=async()=>{try{await api(`/api/runs/${el.dataset.restore}/archive`,{archived:false});toast('会话已恢复');await refresh(true);}catch(error){toast(error.message);}});
  $$('[data-delete]').forEach(el=>el.onclick=async()=>{
    if(!window.confirm('删除后会话会移到 runs/.trash，不再出现在列表。确定删除？'))return;
    try{await api(`/api/runs/${el.dataset.delete}/delete`,{});toast('已移入回收目录');await refresh(true);}catch(error){toast(error.message);}
  });
}
const baseArchiveRefresh=refresh;
refresh=async function(force=false){const result=await baseArchiveRefresh(force);if(page==='archive')renderArchive();renderSidebarSessions();return result;};
const baseArchiveGoto=goto;
goto=function(name){baseArchiveGoto(name);if(name==='archive')renderArchive();renderSidebarSessions();};
$('#session-archive')?.addEventListener('click',async()=>{
  if(!run){toast('先打开一个研究会话');return;}
  try{await api(`/api/runs/${run.id}/archive`,{archived:true});toast('会话已归档');await refresh(true);goto('archive');}
  catch(error){toast(error.message);}
});

/* Budget ceilings come from settings: keep the composer's input bounds in sync. */
function applyBudgetLimits(){
  const limits=(state&&state.settings&&state.settings.limits)||{};
  const tokens=$('#research-tokens'),seconds=$('#research-seconds'),steps=$('#research-steps');
  if(tokens){tokens.max=limits.max_tokens||400000;tokens.min=1000;}
  if(seconds){seconds.max=limits.max_seconds||7200;seconds.min=10;}
  if(steps){steps.max=limits.max_steps||40;steps.min=1;}
}
const baseLimitRefresh=refresh;
refresh=async function(force=false){const result=await baseLimitRefresh(force);applyBudgetLimits();return result;};
applyBudgetLimits();

function fillRoutingSettings(){const conf=state?.settings?.routing||{};$('#routing-enabled').checked=!!conf.enabled;$('#routing-profiles').innerHTML=['light','strong'].map(tier=>{const p=conf.profiles?.[tier]||{};return `<label class="routing-profile" data-tier="${tier}"><span>${tier==='light'?'轻量档':'高能力档'}</span><select aria-label="${tier==='light'?'轻量档':'高能力档'}来源"><option value="local" ${p.model_source!=='cloud'?'selected':''}>本地 Ollama</option><option value="cloud" ${p.model_source==='cloud'?'selected':''}>云端 API</option></select><input aria-label="${tier==='light'?'轻量档':'高能力档'}模型名称" value="${esc(p.model||'')}" placeholder="模型名；留空沿用角色配置"></label>`;}).join('');}
function readRoutingSettings(){const profiles={};$$('[data-tier]').forEach(row=>{profiles[row.dataset.tier]={model_source:row.querySelector('select').value,model:row.querySelector('input').value.trim()};});return {enabled:$('#routing-enabled').checked,profiles};}

function describeSearchConfig(){
  const form=$('#settings-form'),provider=form.querySelector('[name=web_provider]').value,endpoint=form.querySelector('[name=web_endpoint]').value;
  let host='';try{host=new URL(endpoint).hostname;}catch(_){}
  const hasKey=!!form.querySelector('[name=web_api_key]').value.trim()||!!state?.settings?.web?.search_api_key_set;
  $('#web-details').textContent=provider==='searxng'&&(host==='api.tavily.com'||host.endsWith('.tavily.com'))
    ?'配置不匹配：这个地址属于 Tavily。请选择 Tavily，并填写独立的搜索 API Key；模型 API Key 不会用于搜索。'
    :provider==='tavily'&&!hasKey?'尚未配置 Tavily 搜索 Key。它与云端模型 Key 分开管理；未配置时 Agent 会选择其他可用来源。'
    :provider==='searxng'?'SearXNG 需要实际可用的搜索实例地址；不能填写模型 API 或 Tavily 地址。'
    :'已保存搜索服务配置；连接状态请通过“检测联网”确认。';
}
['web_provider','web_endpoint','web_api_key'].forEach(name=>$('#settings-form').querySelector(`[name=${name}]`).addEventListener('input',describeSearchConfig));
