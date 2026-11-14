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
  clock:'<circle cx="12" cy="12" r="9"/><path d="M12 6v6l4 2"/>'
};
function icon(name) { return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${paths[name] || paths.info}</svg>`; }
$$('[data-icon]').forEach(el => el.innerHTML = icon(el.dataset.icon));
const statuses = {queued:'等待中',running:'研究中',completed:'已完成',failed:'调用失败',cancelled:'已停止',budget_stopped:'预算停止',interrupted:'服务重启中断'};
const strategies = {baseline:'固定 BM25',acronym:'语料内缩写扩展',prf:'伪相关反馈扩展'};
const gates = {all:'全部问题',has_acronym:'存在已知缩写',unique_acronym:'缩写映射唯一',ambiguous_acronym:'缩写有歧义',short_query:'短查询',has_negation:'含否定词',has_number:'含数值',rare_terms:'含未登录词'};
const gateLabel = gate => typeof gate==='string' ? (gates[gate]||gate) : Object.entries(gate||{}).map(([op,list])=>list.map(f=>gates[f]||f).join(op==='all'?' 且 ':' 或 ')).join('');
const scopeLabel = e => e.result_scope==='screen'?'开发筛选':'完整探索';
let state = null, run = null, selectedId = localStorage.getItem('rl-selected-run') || '', page = 'workspace';
let evidenceExperiment = '', evidenceClaim = '', evidenceKey = '', lastRunStamp = '', refreshing = false, lastBudgetStamp = 0;
let modelStatus = {available:false,models:[]};
const percent = n => n == null ? '—' : `${(n*100).toFixed(1)}%`;
const number = n => new Intl.NumberFormat('en-US').format(n || 0);
const time = n => n >= 60 ? `${Math.floor(n/60)}m ${Math.round(n%60)}s` : `${Math.round(n || 0)}s`;
const delta = n => `${n >= 0 ? '+' : ''}${(n*100).toFixed(1)} pp`;
const modeLabel = r => r.mode === 'search' ? '随机搜索对照' : r.mode === 'carry' ? '配置沿用对照' : r.memory_version ? `Agent · 经验 v${r.memory_version}` : r.memory_selection?.requested_version ? `Agent · 拒绝经验 v${r.memory_selection.requested_version}` : 'Agent · 无历史经验';

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
  page = ['workspace','records','memory','data'].includes(name) ? name : 'workspace';
  $$('.page').forEach(el => el.classList.toggle('hidden',el.id !== `page-${page}`));
  $$('.nav-item').forEach(el => el.classList.toggle('active',el.dataset.page === page));
  $('#page-label').textContent = {workspace:'研究工作台',records:'实验记录',memory:'研究经验',data:'数据与评估'}[page];
  history.replaceState(null,'',`#${page}`);
  if (page === 'data') renderFinal();
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
    if (!state.runs.some(r => r.id === selectedId)) selectedId = state.active_run || state.runs.find(r=>r.mode==='agent'&&r.status==='completed')?.id || state.runs[0]?.id || '';
    const oldOptions = $('#run-select').innerHTML;
    const options = state.runs.length ? state.runs.map(r => `<option value="${esc(r.id)}">${esc(r.task)} · ${esc(modeLabel(r))} · ${esc(r.id.slice(9,15))}</option>`).join('') : '<option value="">暂无运行</option>';
    if (oldOptions !== options) $('#run-select').innerHTML = options;
    $('#run-select').value = selectedId;
    const summary = state.runs.find(r => r.id === selectedId);
    if (selectedId && (force || !run || run.id !== selectedId || summary?.status === 'running' || summary?.status === 'queued' || run.status !== summary?.status)) {
      run = await api(`/api/runs/${selectedId}`); lastBudgetStamp = Date.now();
    }
    if (!selectedId) run = null;
    renderTop(); renderMetrics(); renderRecords(); renderMemory(); renderData();
    const stamp = run ? `${run.id}:${run.updated_at}` : 'empty';
    if (force || stamp !== lastRunStamp) { lastRunStamp = stamp; renderRun(); }
    renderBudget();
  } catch(error) { $('#notice').textContent = `本地服务连接异常：${error.message}`; $('#notice').classList.remove('hidden'); }
  finally { refreshing=false; }
}
function renderTop() {
  $('#run-count').textContent = state.runs.length;
  $('#memory-count').textContent = state.memory.versions.length;
  $('#active-memory').textContent = `当前版本 v${state.memory.active}`;
  $('#dataset-summary').textContent = state.dataset ? `${number(state.dataset.corpus_count)} 篇摘要 · ${state.research_tasks?.C?'C / D 各 24 道探索 + 12 道验收 · 8 道固定开发筛选':'A / B 为已见开发任务'} · 文档隔离` : '数据尚未准备，前往「数据与评估」下载官方公开数据。';
  const taskInput=$('#run-form [name=task]');
  if(taskInput.dataset.version!=='2'){taskInput.innerHTML='<option value="C">C · 第二版来源任务（24+12）</option><option value="D">D · 第二版迁移任务（24+12）</option><option value="A">A · 已见开发任务（40+20）</option><option value="B">B · 已见开发任务（40+20）</option>';taskInput.dataset.version='2';}
  $('#test-seal').textContent = state.final_test_revealed ? '最终测试已开放' : '最终测试已封存';
  let notice = '';
  if (state.preparing) notice = state.preparing.message;
  else if (state.final_test_revealed) notice = '最终测试已开放。之后的新运行仍可研究，但不能再把同一测试结果作为未见数据的证明。';
  else if (run?.error) notice = run.error;
  else if (run && ['budget_stopped','cancelled','interrupted'].includes(run.status)) notice = run.stop_reason;
  $('#notice').classList.toggle('hidden',!notice); $('#notice').textContent = notice;
  $$('.new-run').forEach(el => el.disabled = !!state.active_run || !!state.preparing || !state.dataset);
}
function metric(label,value,unit,bottom,which='target',extra='') {
  return `<article class="metric-card"><div class="metric-top"><span>${label}</span><span class="metric-icon">${icon(which)}</span></div><div class="metric-value">${value}<small>${unit}</small></div><div class="metric-bottom">${extra}${bottom}</div></article>`;
}
function renderMetrics() {
  const budget=run?.budget;
  const guard=run?.guard;
  const best=guard ? (guard.accepted ? guard.candidate.metrics : guard.baseline.metrics) : run?.experiments.filter(e=>e.result_scope!=='screen').reduce((b,e)=>Math.max(b,e.result.metrics.complete_at_5),0);
  const metricValue=guard ? percent(best.complete_at_5) : run?.experiments.length ? percent(best) : '—';
  const d=guard ? (guard.accepted ? guard.delta.complete_at_5 : 0) : null;
  $('#metrics').innerHTML = metric(guard?'验收集证据命中@5':'探索集暂定命中@5', metricValue, '',guard?'相对固定基线':'等待独立验收','target',d===null?'':`<span class="delta-tag ${d<0?'negative':''}">${delta(d)}</span>`) +
    metric('本轮模型 token',budget?number(budget.total_tokens):'—','',budget?.unknown_reserved?'包含未返回 usage 的预留量':run?.mode==='search'?'确定性对照 · 无模型调用':'规划、反思与重试均计入','coin') +
    metric('已完成候选实验',run?String(Math.max(0,run.experiments.length-1)):'0',run?`/ ${run.max_experiments}`:'',run?`${modeLabel(run)}`:'从一个可检验的假设开始','flask')+
    metric('当前经验版本',`v${state.memory.active}`,'',`${state.memory.versions.length} 个历史版本 · 支持回退`,'spark');
}
function renderRun() {
  const experiments=run?.experiments || [];
  renderDiagnostics();
  if(experiments.length&&!experiments.some(e=>e.id===evidenceExperiment))evidenceExperiment=experiments.at(-1).id;
  $('#experiment-counter').textContent = `${experiments.length} 个实验`;
  if(!run) {
    $('#run-meta').innerHTML='';
    $('#chart').innerHTML='<div class="chart-empty"><span>研究曲线将从第一次真实测量开始</span></div>';
    $('#experiments').innerHTML=empty('还没有运行记录','开始研究，或先运行不依赖模型的随机搜索对照。');
    $('#hypothesis').innerHTML='<div class="hypothesis-empty">等待研究任务。<br>系统会从基线失败中寻找值得验证的问题。</div>';
    $('#evidence-experiment').innerHTML='';$('#evidence-claim').innerHTML='';
    $('#evidence').innerHTML=empty('证据会出现在这里','每个实验保留逐题结果、原始句子与标注证据。','⌕');
    return;
  }
  const active=['running','queued'].includes(run.status);
  $('#run-meta').innerHTML=`<span class="status-dot ${active?'running':run.status==='failed'?'failed':''}"></span><span>${esc(statuses[run.status])}</span><span>·</span><span>${esc(modeLabel(run))}</span><span>·</span><span>任务 ${esc(run.task)}</span><span>· seed ${run.seed}</span><span class="spacer"></span>${active?'<button id="stop-run">停止</button>':''}<button id="show-calls">原始调用</button><a href="/api/runs/${esc(run.id)}/export">导出结果 ↗</a>${run.memory_candidate?'<button id="publish-memory">发布经验</button>':''}`;
  $('#stop-run')?.addEventListener('click',async()=>{try{await api(`/api/runs/${run.id}/cancel`,{});toast('已请求停止，当前模型或工具调用结束后生效。');}catch(e){toast(e.message);}});
  $('#show-calls').addEventListener('click',()=>showDetail('模型调用与完整 usage',run.calls));
  $('#publish-memory')?.addEventListener('click',async()=>{try{const v=await api('/api/memory/promote',{run_id:run.id});toast(`经验 v${v.version} 已发布。迁移收益仍需在 B 上检验。`);await refresh(true);}catch(e){toast(e.message);}});
  renderChart(experiments);
  $('#experiments').innerHTML=experiments.map(e=>`<div class="experiment-row ${e.id==='e0'?'baseline':''} ${e.id===evidenceExperiment?'selected':''}" role="button" tabindex="0" data-experiment="${esc(e.id)}" aria-label="查看 ${esc(e.id)} 实验证据"><div class="experiment-number">${esc(e.id)}</div><div><div class="experiment-title">${esc(e.hypothesis)}</div><div class="experiment-detail"><span>${esc(strategies[e.config.strategy])}</span><span>· ${esc(gateLabel(e.config.gate))}</span>${e.config.normalization==='preserve_constraints'?'<span>保留否定与数值</span>':''}<span>${scopeLabel(e)} ${e.result.count} 题</span><span class="mono">${e.result.seconds.toFixed(2)}s</span>${e.regressed_ids?.length?`<span>退化 ${e.regressed_ids.length} 题</span>`:''}</div></div><div class="experiment-outcome">${percent(e.result.metrics.complete_at_5)}<small>${e.id==='e0'?'固定基线':e.result_scope==='screen'?'仅筛选 · 已否决':e.accepted_on_explore?'完整探索暂优':'未替换当前最优'}</small></div></div>`).join('') || empty('正在准备基线','首次加载语料索引可能需要数秒。');
  $$('.experiment-row').forEach(el=>{const choose=()=>{evidenceExperiment=el.dataset.experiment;evidenceClaim='';evidenceKey='';renderEvidenceSelectors();showExperimentHypothesis(experiments.find(e=>e.id===evidenceExperiment));$$('.experiment-row').forEach(row=>row.classList.toggle('selected',row===el));};el.addEventListener('click',choose);el.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();choose();}});});
  const current=experiments.at(-1);
  const lastEvent=run.events.at(-1);
  const hypothesis=run.events.filter(e=>e.type==='hypothesis').at(-1)?.decision;
  const question=experiments.find(e=>e.id===evidenceExperiment) || hypothesis || current;
  $('#hypothesis').innerHTML=question?`<span class="hypothesis-tag">${esc(active && lastEvent?.type==='planning'?'正在规划下一项':question.id==='e0'?'代码固定基线':run.mode==='agent'?'本地模型提出':'确定性对照')}</span><div class="hypothesis-text">${esc(question.hypothesis)}</div>${question.falsify_if?`<div class="hypothesis-source"><strong>如何否证</strong><br>${esc(question.falsify_if)}</div>`:`<div class="hypothesis-source">${esc(run.mode==='agent'?(active?'等待模型根据失败案例提出机制解释。':`本轮停止理由：${run.stop_reason}`):'此模式验证检索工具和评估流程，不生成模型研究经验。')}</div>`}`:'<div class="hypothesis-empty">正在加载语料、建立固定检索基线。</div>';
  if(active && lastEvent?.type==='planning')$('#hypothesis').insertAdjacentHTML('beforeend','<div class="hypothesis-source"><span class="loader"></span> 正在等待本地模型…</div>');
  renderEvidenceSelectors();
}
function renderChart(experiments) {
  experiments=experiments.filter(e=>e.result_scope!=='screen');
  if(!experiments.length){$('#chart').innerHTML='<div class="chart-empty"><span>等待第一次测量</span></div>';return;}
  const width=620,height=145,left=37,right=17,top=18,bottom=27;
  const values=experiments.map(e=>e.result.metrics.complete_at_5);
  const lo=Math.max(0,Math.floor((Math.min(...values)-.09)*10)/10),hi=Math.min(1,Math.ceil((Math.max(...values)+.09)*10)/10);
  const x=i=>left+(width-left-right)*i/Math.max(1,experiments.length-1),y=v=>top+(height-top-bottom)*(1-(v-lo)/(hi-lo||1));
  const ticks=[lo,(lo+hi)/2,hi];let best=-1;
  const current=values.map((v,i)=>`${x(i)},${y(v)}`).join(' ');
  const bestPoints=values.map((v,i)=>{best=Math.max(best,v);return `${x(i)},${y(best)}`;}).join(' ');
  const dots=values.map((v,i)=>`<circle cx="${x(i)}" cy="${y(v)}" r="3.7" fill="#8d7ca8" stroke="white" stroke-width="2"><title>${esc(experiments[i].id)}: ${percent(v)}</title></circle><text class="chart-axis" x="${x(i)}" y="${height-4}" text-anchor="middle">${esc(experiments[i].id)}</text>`).join('');
  $('#chart').innerHTML=`<svg class="research-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="探索集真实测量曲线，独立验收结果另行报告">${ticks.map(v=>`<line class="chart-grid" x1="${left}" x2="${width-right}" y1="${y(v)}" y2="${y(v)}"/><text class="chart-axis" x="${left-9}" y="${y(v)+3}" text-anchor="end">${Math.round(v*100)}%</text>`).join('')}<polyline points="${bestPoints}" fill="none" stroke="#7e9c67" stroke-width="2"/><polyline points="${current}" fill="none" stroke="#b1a0c4" stroke-width="1.8" stroke-dasharray="4 4"/>${dots}</svg><div class="chart-legend"><span><i class="legend-dot"></i>探索集当前最好</span><span><i class="legend-dot purple"></i>本次候选测量</span><span>独立验收仅在研究结束后进行</span></div>`;
}
function renderBudget() {
  if(!run?.budget){$('#budget').innerHTML='<div class="budget-item"><div class="budget-label">模型 token <strong>0 / —</strong></div><div class="progress-track"></div></div><div class="budget-item"><div class="budget-label">墙钟时间 <strong>尚未开始</strong></div><div class="progress-track"></div></div><div class="budget-footer"><span class="small-dot"></span>云端调用 0 次 · 无隐式升级</div>';return;}
  const b=run.budget;
  const elapsed=b.elapsed_seconds+(['running','queued'].includes(run.status)?Math.max(0,(Date.now()-Date.parse(run.updated_at))/1000):0);
  $('#budget').innerHTML=`<div class="budget-item"><div class="budget-label">模型 token <strong>${number(b.total_tokens)} / ${number(b.token_limit)}</strong></div><div class="progress-track"><div class="progress-fill" style="width:${Math.min(100,b.total_tokens/b.token_limit*100)}%"></div></div></div><div class="budget-item"><div class="budget-label">墙钟时间 <strong>${time(elapsed)} / ${time(b.seconds_limit)}</strong></div><div class="progress-track"><div class="progress-fill purple" style="width:${Math.min(100,elapsed/b.seconds_limit*100)}%"></div></div></div><div class="budget-footer"><span class="small-dot"></span>${b.model_calls} 次模型调用 · 云端 ${b.cloud_calls} 次${b.unknown_reserved?' · 含未知用量预留':''}</div>`;
}
function renderEvidenceSelectors() {
  if(!run?.experiments.length)return;
  const experiments=run.experiments;
  if(!experiments.some(e=>e.id===evidenceExperiment))evidenceExperiment=experiments.at(-1).id;
  $('#evidence-experiment').innerHTML=experiments.map(e=>`<option value="${esc(e.id)}">${esc(e.id)} · ${e.result_scope==='screen'?'筛选 · ':''}${esc(strategies[e.config.strategy])}</option>`).join('');
  $('#evidence-experiment').value=evidenceExperiment;
  const rows=experiments.find(e=>e.id===evidenceExperiment).result.rows;
  if(!rows.some(r=>r.claim_id===evidenceClaim))evidenceClaim=(rows.find(r=>!r.complete_at_5)||rows[0])?.claim_id || '';
  $('#evidence-claim').innerHTML=rows.map(r=>`<option value="${esc(r.claim_id)}">${r.complete_at_5?'✓':'○'} #${esc(r.claim_id)} · ${esc(r.claim.slice(0,43))}</option>`).join('');
  $('#evidence-claim').value=evidenceClaim;
  renderEvidence();
}
function showExperimentHypothesis(experiment){if(!experiment)return;$('#hypothesis').innerHTML=`<span class="hypothesis-tag">实验 ${esc(experiment.id)} · ${experiment.id==='e0'?'代码固定基线':run.mode==='agent'?'本地模型提出':'确定性对照'}</span><div class="hypothesis-text">${esc(experiment.hypothesis)}</div>${experiment.falsify_if?`<div class="hypothesis-source"><strong>如何否证</strong><br>${esc(experiment.falsify_if)}</div>`:''}<div class="hypothesis-source">${experiment.id==='e0'?'固定分句、固定 BM25，作为所有实验的比较起点。':`相对父实验改善 ${experiment.improved_ids?.length||0} 题，退化 ${experiment.regressed_ids?.length||0} 题。结论限定于探索集。`}</div><button class="small-link" id="experiment-details">查看配置与决策依据 ↗</button>`;$('#experiment-details').addEventListener('click',()=>showDetail(`实验 ${experiment.id} · 配置与依据`,Object.fromEntries(Object.entries(experiment).filter(([k])=>k!=='result')).valueOf()));}
$('#evidence-experiment').addEventListener('change',e=>{evidenceExperiment=e.target.value;evidenceClaim='';renderEvidenceSelectors();showExperimentHypothesis(run.experiments.find(item=>item.id===evidenceExperiment));});
$('#evidence-claim').addEventListener('change',e=>{evidenceClaim=e.target.value;renderEvidence();});
async function renderEvidence() {
  const experiment=run?.experiments.find(e=>e.id===evidenceExperiment);
  const row=experiment?.result.rows.find(r=>r.claim_id===evidenceClaim);
  if(!row)return;
  const key=`${run.id}:${evidenceExperiment}:${evidenceClaim}`;
  if(key===evidenceKey)return;evidenceKey=key;
  const goldIds=new Set(row.gold_sets.flat());
  const card=(h,gold=false)=>`<article class="evidence-card ${gold?'gold':''}"><div class="evidence-card-top"><span class="mono">${esc(h.id)}</span>${h.label?`<span>${esc(h.label)}</span>`:''}${gold?'<span class="pill">标注证据</span>':''}</div><h4>${esc(h.title)}</h4><p>${esc(h.text)}</p></article>`;
  $('#evidence').innerHTML=`<div class="claim-box"><span class="claim-label">RESEARCH QUERY</span><div class="claim-text">${esc(row.claim)}<span class="claim-result">${scopeLabel(experiment)} ${experiment.result.count} 题中的一题 · ${row.complete_at_5?'完整证据命中':'完整证据未命中'} · 证据覆盖 ${percent(row.coverage_at_5)} · nDCG@5 ${row.ndcg_at_5.toFixed(3)}${row.failure_type?` · 失败分类 ${esc(row.failure_type)}`:''}</span></div></div>${row.expansion.length?`<div class="expansion-note">本次扩展：${esc(row.expansion.join(' · '))}。词语来自语料，原查询保留。</div>`:''}<div class="evidence-grid"><div class="evidence-column"><h3>检索结果 · Top 5</h3>${row.hits.map(h=>card(h,goldIds.has(h.id))).join('')||'<p class="body-muted">未返回句子</p>'}</div><div class="evidence-column"><h3>人工标注的支持 / 反驳证据</h3><div id="gold-evidence" class="evidence-loading">正在读取原始证据…</div></div></div>`;
  try {
    const data=await api(`/api/runs/${run.id}/evidence/${row.claim_id}`);
    if(evidenceKey!==key)return;
    $('#gold-evidence').classList.remove('evidence-loading');
    $('#gold-evidence').innerHTML=data.gold.map(h=>card(h,true)).join('');
  }catch(e){if(evidenceKey===key)$('#gold-evidence').textContent=e.message;}
}
function renderRecords() {
  if(!state.runs.length){$('#records').innerHTML=empty('实验档案还是空的','每次真实运行都会自动保存，不会生成示意性能数字。');return;}
  $('#records').innerHTML=`<div class="table-wrap"><table><thead><tr><th>运行 / 任务</th><th>执行方式</th><th>状态</th><th>验收命中@5</th><th>模型 token</th><th>耗时 / 实验</th><th>预算 / 种子</th><th></th></tr></thead><tbody>${state.runs.map(r=>`<tr><td class="mono">${esc(r.id.slice(0,15))}<small>任务 ${esc(r.task)}</small></td><td>${esc(modeLabel(r))}<small>${esc(r.model||'代码执行，无模型')}</small></td><td><span class="status-badge ${esc(r.status)}">${esc(statuses[r.status])}</span></td><td>${r.guard?percent(r.guard.metrics.complete_at_5):'—'}<small>${r.guard?(r.guard.accepted?'候选通过':'已回退到基线'):'尚未独立验收'}</small></td><td class="mono">${r.budget?number(r.budget.total_tokens):'—'}<small>${r.budget?.unknown_reserved?'含未知 usage 的预留量':''}</small></td><td>${time(r.budget?.elapsed_seconds)} / ${r.experiment_count} 次</td><td class="mono">${number(r.budget?.token_limit)} / ${r.budget?.seconds_limit||'—'}s<small>seed ${r.seed}</small></td><td><button class="small-link" data-open-run="${esc(r.id)}">查看 ↗</button></td></tr>`).join('')}</tbody></table></div><p class="comparison-note">请仅比较同任务、同模型、同预算和同种子的冷 / 热启动运行。零 token 的随机搜索是工具对照，不代表 Agent 已完成自主研究。</p>`;
  $$('[data-open-run]').forEach(el=>el.addEventListener('click',()=>selectRun(el.dataset.openRun)));
}
function renderMemoryLegacy() {
  const versions=[...state.memory.versions].reverse();
  $('#memory-list').innerHTML=versions.map(v=>`<article class="panel memory-card"><div class="memory-card-head"><h2>v${v.version}</h2><span class="pill ${state.memory.active===v.version?'':'subtle'}">${state.memory.active===v.version?'当前启用':'历史版本'}</span><span class="pill subtle">迁移仍待验证</span>${state.memory.active!==v.version?`<button class="secondary-button" data-activate="${v.version}">切换到此版本</button>`:''}</div><p>${esc(v.lesson)}</p><p class="body-muted">适用范围：英文科学摘要。${esc(v.limitations)}</p><div class="memory-evidence"><span>来自 <button class="small-link" data-memory-run="${esc(v.source_run)}">${esc(v.source_run)} ↗</button></span><span>父版本 v${v.parent_version}</span><span>验收命中差异 ${delta(v.guard_delta.complete_at_5)}</span><span>${v.evidence.length} 项实验依据</span></div></article>`).join('')+`<article class="panel memory-card"><div class="memory-card-head"><h2>v0</h2><span class="pill subtle">无历史经验</span>${state.memory.active!==0?'<button class="secondary-button" data-activate="0">回退到 v0</button>':'<span class="pill">当前启用</span>'}</div><p>从固定 BM25 基线开始，根据本轮探索集反馈决定实验。用于冷启动和经验消融对照。</p></article>`;
  $$('[data-activate]').forEach(el=>el.addEventListener('click',async()=>{try{await api('/api/memory/activate',{version:Number(el.dataset.activate)});toast(`已切换到 v${el.dataset.activate}，历史记录保持可查。`);await refresh(true);}catch(e){toast(e.message);}}));
  $$('[data-memory-run]').forEach(el=>el.addEventListener('click',()=>selectRun(el.dataset.memoryRun)));
}
function renderDataLegacy() {
  const d=state.dataset;
  $('#prepare-button').disabled=!!d||!!state.active_run||!!state.preparing;
  $('#prepare-button').textContent=d?'公开数据已就绪':state.preparing?'数据准备中…':'准备公开数据';
  if(!d){$('#data-content').innerHTML=empty('先建立可检查的研究环境','下载 SciFact 原始数据、许可文件，并冻结按文档隔离的数据划分。');return;}
  $('#data-content').innerHTML=`<div class="data-grid"><article class="panel data-card"><div class="section-label">SOURCE TASK A</div><h2>积累研究经验</h2><div class="data-number">${d.splits.A.explore} <span class="body-muted">+ ${d.splits.A.guard}</span></div><p>探索题 + 独立验收题。只有探索反馈进入迭代规划，验收后不能继续修改本轮配置。</p></article><article class="panel data-card"><div class="section-label">TRANSFER TASK B</div><h2>检验经验迁移</h2><div class="data-number">${d.splits.B.explore} <span class="body-muted">+ ${d.splits.B.guard}</span></div><p>新问题与新证据文档。冷启动和带经验运行使用同一任务，跨运行不继承结果缓存。</p></article><article class="panel data-card"><div class="section-label">SEALED FINAL TEST</div><h2>最终统一验收</h2><div class="data-number">${d.splits.test}</div><p>从官方开发集保留。与 A / B 的标注证据文档分离，所有候选冻结后一次性开放结果。</p></article></div><section class="panel data-metadata"><h2>SciFact · 固定句子级证据</h2><p>${number(d.corpus_count)} 篇摘要；返回最多 5 个唯一原始句子。主指标为完整命中任一可接受证据集合的比例，辅助报告最佳集合覆盖率与 nDCG@5。仅纳入有标注证据的主张，未标注内容不等于已证明不相关。</p><p>标注：CC BY 4.0 · 摘要：ODC-By 1.0。<a href="https://github.com/allenai/scifact/blob/master/LICENSE.md" target="_blank" rel="noreferrer">查看原始许可 ↗</a></p><div class="section-label">SOURCE ARCHIVE SHA256</div><code>${esc(d.sha256)}</code></section>`;
  $('#finalize-button').disabled=!!state.active_run||state.final_test_revealed||!state.runs.some(r=>r.status==='completed');
  $('#finalize-button').textContent=state.final_test_revealed?'已冻结，结果只读':'冻结并统一验收';
}
async function renderFinal() {
  try{const report=await api('/api/final');if(report.sealed){$('#final-results').innerHTML='<div class="call-summary" style="padding-top:15px">尚未开放最终测试。当前所有研究与验收均不读取其反馈。</div>';return;}
    $('#final-results').innerHTML=`<div class="table-wrap"><table><thead><tr><th>冻结方案</th><th>完整命中@5</th><th>证据覆盖@5</th><th>nDCG@5</th><th>样本数</th></tr></thead><tbody>${Object.entries(report.results).map(([id,r])=>`<tr><td>${esc(id==='baseline'?'固定 BM25 基线':id)}</td><td>${percent(r.metrics.complete_at_5)}</td><td>${percent(r.metrics.coverage_at_5)}</td><td>${r.metrics.ndcg_at_5.toFixed(3)}</td><td>${r.count}</td></tr>`).join('')}</tbody></table></div><p class="comparison-note">此处仅适用于已冻结的运行。数据量较小，结果不能外推到其他领域或一般科研能力。</p>`;
  }catch(e){toast(e.message);}
}
function showDetail(title,value){$('#detail-title').textContent=title;$('#detail-json').textContent=JSON.stringify(value,null,2);$('#detail-dialog').showModal();}
function renderDiagnostics(){
  const panel=$('#diagnostics-panel'),d=run?.diagnostics;
  panel.classList.toggle('hidden',!d);if(!d)return;
  const counts=d.failure_counts,groups=d.feature_groups;
  const items=[['初始召回缺失',counts.recall||0,'标注证据句未进入初始前 50'],['排序不足',counts.ranking||0,'证据集合在初始前 50，最终前 5 不完整'],['部分召回',counts.partial_recall||0,'初始前 50 仅包含部分证据'],['可观察的约束',`${groups.has_negation?.count||0} / ${groups.has_number?.count||0}`,'含否定词问题 / 含数值问题']];
  const m=run.memory_selection,c=run.evaluation_counts,f=run.candidate_filter;
  panel.innerHTML=`<div class="panel-header"><div><div class="section-label">DIAGNOSE BEFORE EXPERIMENTING</div><h2>基线失败诊断 <span class="counter">开发数据 ${d.count} 题</span></h2></div><span class="pill subtle">诊断用金标 · 策略门控不用金标</span></div><div class="diagnostic-grid">${items.map(([label,value,note])=>`<div><span>${label}</span><strong>${value}</strong><small>${note}</small></div>`).join('')}</div><div class="diagnostic-notes">${m?`<p><strong>经验检查：</strong>${esc(m.reason)}${m.profile_distance!=null?` · 特征距离 ${m.profile_distance}`:''}</p>`:''}${f?`<p>当前候选生成 ${f.generated} 项，移除等价或不触发干预 ${f.equivalent_or_inactive_removed} 项，保留 ${f.remaining} 项。此规则为所有对照组共用。</p>`:''}${c?`<p>评估账本：${c.screen} 次开发筛选 / ${c.full} 次完整探索 / ${c.guard} 次验收；同轮复用 ${c.reused_rows} 条已测结果。筛选结果不进入完整探索曲线。</p>`:''}</div>`;
}
function renderMemory(){
  renderMemoryLegacy();const versions=[...state.memory.versions].reverse();
  $$('#memory-list .memory-card').slice(0,versions.length).forEach((card,i)=>{const v=versions[i];const node=document.createElement('div');node.className='memory-rule-note';node.textContent=v.schema_version>=3?`适用检查：仅看查询特征，平均差异上限 ${v.applicability.max_mean_distance}；策略必须在目标问题上可触发。完整依据保存在来源运行。`:'旧版经验没有特征依据：Agent 会拒绝直接注入，仍可用于配置沿用对照。';$('.memory-evidence',card).before(node);});
}
function renderData(){
  renderDataLegacy();if(!state.dataset||!state.research_tasks?.C)return;
  const tasks=state.research_tasks;
  $('#data-content .data-grid').innerHTML=['C','D'].map(name=>`<article class="panel data-card"><div class="section-label">${name==='C'?'SOURCE':'TRANSFER'} TASK ${name}</div><h2>${name==='C'?'诊断与经验来源':'新任务配对验证'}</h2><div class="data-number">${tasks[name].explore} <span class="body-muted">+ ${tasks[name].guard}</span></div><p>探索 + 一次独立验收。固定 ${tasks[name].screen} 道开发筛选题。${state.task_exposure?.[name]?'已有真实运行，后续分析时属于已见任务。':'尚未运行；不按检索效果选择问题。'}</p></article>`).join('')+`<article class="panel data-card"><div class="section-label">SEALED FINAL TEST</div><h2>最终统一验收</h2><div class="data-number">${state.dataset.splits.test}</div><p>保留测试继续封存。旧 A/B 仅作开发诊断；E/F 各 24+12 题预留，未开放指标。</p></article>`;
  const note=document.createElement('p');note.className='task-registry-note';note.textContent=`任务注册哈希：${state.task_registry_hash}。按证据文档连通分组排除旧任务与最终测试；任务清单及后续暴露记录见本地 docs/experiments。`;
  $('#data-content .data-metadata').append(note);
}
async function selectRun(id){selectedId=id;localStorage.setItem('rl-selected-run',id);evidenceKey='';evidenceClaim='';evidenceExperiment='';goto('workspace');await refresh(true);}
$('#run-select').addEventListener('change',e=>selectRun(e.target.value));
$$('.new-run').forEach(el=>el.addEventListener('click',()=>{
  $('#memory-form-hint').textContent=state.memory.active?`当前 v${state.memory.active}，只作为可被否决的研究先验`:'当前 v0，尚无已发布经验';
  const currentMemory=state.memory.versions.find(v=>v.version===state.memory.active);
  $('#memory-input').checked=false;$('#run-form [name=task]').value=currentMemory?.schema_version>=3?'D':'C';updateMode();$('#run-dialog').showModal();
}));
function updateMode(){const mode=$('#mode-input').value;$('#memory-input').disabled=mode!=='agent'||!state?.memory.active;if(mode==='search')$('#memory-input').checked=false;if(mode==='carry')$('#memory-input').checked=true;$('#mode-description').textContent={agent:'模型负责假设与实验决策；数值计算和验收由代码完成。失败调用与重试也计入预算。',search:'随机选择合法单因素候选，全部测量来自真实检索。此模式不调用模型，也不能发布 Agent 经验。',carry:'仅验证来源任务的最佳配置，用于区分「复用方案」与「研究能力提升」。不调用模型。'}[mode];}
$('#mode-input').addEventListener('change',updateMode);
$('#run-form').addEventListener('submit',async e=>{e.preventDefault();const form=new FormData(e.target);const body=Object.fromEntries(form);body.use_memory=$('#memory-input').checked;for(const k of ['max_experiments','token_limit','seconds_limit','seed'])body[k]=Number(body[k]);body.memory_version=state.memory.active;const button=$('#run-form [type=submit]');button.disabled=true;try{const created=await api('/api/runs',body);$('#run-dialog').close();await selectRun(created.id);toast('研究已启动，结果和全部调用消耗将自动保存。');}catch(error){toast(error.message);}finally{button.disabled=false;}});
async function checkModels(){const badge=$('#ollama-status');badge.innerHTML='<span class="loader"></span>检查 Ollama…';try{modelStatus=await api('/api/ollama');badge.classList.toggle('offline',!modelStatus.available);badge.innerHTML=`<span class="small-dot"></span>${modelStatus.available?'Ollama 已连接':'Ollama 未连接'}`;$('#model-options').innerHTML=modelStatus.models.filter(m=>!m.name.toLowerCase().includes('cloud')).map(m=>`<option value="${esc(m.name)}">${esc(m.details?.parameter_size||'')}</option>`).join('');$('#model-details').textContent=modelStatus.available?`检测到 ${modelStatus.models.length} 个模型。当前默认 ${state?.settings.model||'未设置'}。所有推理保持本地，云端接口关闭。`:`本地服务未连接：${modelStatus.error}。请运行 ollama serve，再检测服务。`;}catch(e){badge.textContent='连接检测失败';toast(e.message);}}
function openSettings(){if(state){$('#settings-form [name=ollama_url]').value=state.settings.ollama_url;$('#model-name').value=state.settings.model;}$('#settings-dialog').showModal();checkModels();}
$('#settings-button').addEventListener('click',openSettings);$('#settings-shortcut').addEventListener('click',openSettings);$('#refresh-models').addEventListener('click',checkModels);
$('#settings-form').addEventListener('submit',async e=>{e.preventDefault();try{await api('/api/settings',Object.fromEntries(new FormData(e.target)));$('#settings-dialog').close();toast('本地模型设置已保存。');await refresh(true);await checkModels();}catch(error){toast(error.message);}});
$('#prepare-button').addEventListener('click',async()=>{try{await api('/api/prepare',{});toast('开始准备数据，所有文件写入当前项目的 data 目录。');await refresh(true);}catch(e){toast(e.message);}});
$('#finalize-button').addEventListener('click',()=>$('#confirm-dialog').showModal());
$('#confirm-finalize').addEventListener('click',async()=>{const button=$('#confirm-finalize');button.disabled=true;button.textContent='正在统一验收…';try{await api('/api/finalize',{});$('#confirm-dialog').close();toast('方案已冻结，最终测试结果已保存。');await refresh(true);await renderFinal();}catch(e){toast(e.message);}finally{button.disabled=false;button.textContent='确认冻结并验收';}});
goto(location.hash.slice(1)||'workspace');
refresh(true).then(checkModels);
setInterval(()=>refresh(),2200);
setInterval(renderBudget,1000);
