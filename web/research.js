/* Question-driven workspace. Legacy domain runs remain readable in the archive. */
statuses.incomplete='证据不足 / 未完成';
const previousTop=renderTop, previousMetrics=renderMetrics, previousRun=renderRun, previousMemory=renderMemory, previousData=renderData;
const memoryArchive=document.createElement('details');memoryArchive.className='legacy-archive';memoryArchive.innerHTML='<summary>旧版检索经验档案</summary>';
$('#memory-list').before(memoryArchive);memoryArchive.append($('#memory-list'));
memoryArchive.insertAdjacentHTML('beforebegin','<div id="research-memory-list"></div>');
const toolPage=$('#page-data'), toolArchive=document.createElement('details');toolArchive.className='legacy-archive';toolArchive.innerHTML='<summary>旧版SciFact数据与封存评估</summary>';
for(const child of [...toolPage.children])if(!child.classList.contains('page-heading'))toolArchive.append(child);
toolArchive.append($('#prepare-button'));toolPage.append(toolArchive);
toolArchive.insertAdjacentHTML('beforebegin','<div id="research-tool-list" class="research-tools"></div>');
$('#page-data h1').textContent='材料与工具';
$('#page-data .subtitle').textContent='查看工具的输入要求和使用范围。';
$('.protocol-callout strong').textContent='评价研究过程，而不只看报告是否写完';
$('.protocol-callout p').textContent='记录证据、可证伪假设、真实实验与独立复核。比较历史经验时保持同问题、材料、模型和预算；复核完成也可能得到否定结论。';
$('#log-source').insertAdjacentHTML('afterbegin','<option value="research">科研 Agent</option>');

renderTop=function(){
  previousTop();
  $('#dataset-summary').textContent=`${state.research_tools?.length||0} 项工具可用 · 当前会话 ${run?.assets?.length||0} 份材料`;
  $('#test-seal').classList.add('hidden');
  if(page==='data')$('#page-label').textContent='材料与工具';
  $('#draft-brief').disabled=!!state.active_run;
  const modelSelect=$('#research-model');
  const cloud=state.settings?.cloud||{};
  if(modelStatus?.models?.length||(cloud.enabled&&cloud.model)){
    const names=(modelStatus?.models||[]).filter(m=>!m.name.toLowerCase().includes('cloud')).map(m=>m.name);
    const options=[...names.map(name=>`<option value="${esc(name)}">${esc(name)} · 本地</option>`),(cloud.enabled&&cloud.model)?`<option value="cloud:${esc(cloud.model)}">${esc(cloud.model)} · 云端</option>`:''].join('');
    if(modelSelect.innerHTML!==options){const selected=modelSelect.value;modelSelect.innerHTML=options;const fallback=cloud.enabled&&cloud.model?`cloud:${cloud.model}`:(names.includes('deepseek-r1:14b')?'deepseek-r1:14b':state.settings.model);modelSelect.value=modelSelect.dataset.ready&&options.includes(`value="${selected}"`)?selected:(options.includes(`value="${fallback}"`)?fallback:(names[0]||''));modelSelect.dataset.ready='1';}
  }
};
renderMetrics=function(){
  if(run&&run.domain!=='research'){previousMetrics();return;}
  const experiments=run?.experiments||[], verified=experiments.filter(e=>e.verification);
  $('#metrics').innerHTML=metric('已复核实验',String(verified.length),'',`${verified.filter(e=>e.verification.check.passed).length} 项满足判据，否定结果同样保留`,'target')+
    metric('本轮模型 token',run?.budget?number(run.budget.total_tokens):'0','','规划、修正与报告决策均计入','coin')+
    metric('真实实验',String(experiments.length),'',`${run?.steps?.length||0} 步工具行动 · 不预设实验顺序`,'flask')+
    metric('研究经验',`v${run?.memory_version||0}`,'','仅作待验证先验，不继承历史成绩','spark');
};
renderRun=function(){
  if(run&&run.domain!=='research'){previousRun();return;}
  if(!run){
    $('#session-title').textContent='研究对话';
    $('#session-question').textContent='从工作台开始一个新的研究，或从左侧研究记录里打开已有会话。';
    $('#session-status').textContent='未选择';
  }else{
    $('#session-title').textContent=(run.question||run.task||'研究对话').slice(0,60);
    $('#session-question').textContent=run.question||'';
    $('#session-status').textContent=statuses[run.status]||run.status;
  }
  $('#diagnostics-panel').classList.add('hidden');
  $('#evidence-experiment').innerHTML='';$('#evidence-claim').innerHTML='';
  if(!run){
    $('#run-meta').innerHTML='';$('#experiment-counter').textContent='等待研究问题';
    $('#chart').innerHTML='<div class="research-guide"><span>研究问题</span> → <span>证据与假设</span> → <span>实验与分析</span> → <span>复核与修正</span></div>';
    $('#experiments').innerHTML=empty('从一个可检验的问题开始','Agent会根据工具的实际返回决定下一步，报告包含执行代码、测量结果、成本与局限。');
    $('#hypothesis').innerHTML='<div class="hypothesis-empty">等待问题与材料。<br>假设由Agent提出，判据在实验之前登记。</div>';
    $('#evidence').innerHTML=empty('真实证据会汇集到这里','文献来源、上传材料与实验ID均可追溯。没有验证时不会显示完成。');return;
  }
  const active=['running','queued'].includes(run.status);
  $('#experiment-counter').textContent=`${run.steps.length} 步行动 · ${run.experiments.length} 项实验`;
  $('#run-meta').innerHTML=`<span class="status-dot ${active?'running':''}"></span><span>${esc(statuses[run.status]||run.status)}</span><span>· 科研 Agent</span><span class="spacer"></span><button id="research-raw">完整记录</button>${active?'<button id="research-cancel">停止</button>':`<a href="/api/runs/${run.id}/report">下载报告</a>`}<a href="/api/runs/${run.id}/export">复现包 ↗</a>${run.status==='completed'&&run.report?.lesson?'<button id="research-publish">发布方法经验</button>':''}`;
  $('#research-raw').onclick=()=>showDetail('研究问题、决策和原始工具结果',run);
  $('#research-cancel')?.addEventListener('click',async()=>{await api(`/api/runs/${run.id}/cancel`,{});toast('当前调用结束后停止，已有证据保留。');});
  $('#research-publish')?.addEventListener('click',async()=>{try{const value=await api('/api/memory/promote',{run_id:run.id});toast(`已发布科研经验 v${value.version}，迁移收益仍待验证。`);await refresh(true);}catch(e){toast(e.message);}});
  const h=run.hypotheses.at(-1);
  $('#hypothesis').innerHTML=`<span class="hypothesis-tag">${h?esc(h.id):'收集证据中'}</span><div class="hypothesis-text">${esc(h?.statement||run.question)}</div><div class="hypothesis-source">${h?`预测：${esc(h.prediction)}<br>控制：${esc(h.controls)}<br>冻结判据：${esc(h.metric)} ${esc(h.relation)} ${esc(h.threshold)}`:'模型将先检查条件、材料和可用工具，不预设结论。'}</div>`;
  $('#chart').innerHTML=`<div class="research-guide">${esc(run.question)}</div>`;
  $('#experiments').innerHTML=run.steps.length?run.steps.map(s=>`<details class="research-step"><summary><span class="pill subtle">${esc(s.id)} · ${esc(s.tool)}</span><strong>${esc(s.reason)}</strong><span class="${s.status==='failed'?'negative':''}">${s.status==='failed'?'失败，等待修正':'已执行'}</span></summary><pre>${esc(JSON.stringify({arguments:s.args,observation:s.observation},null,2))}</pre></details>`).join(''):empty('正在规划第一步','模型正在结合问题、预算与工具拟定研究行动。');
  renderResearchEvidence(run);
};
function renderResearchEvidence(value){
  const box=$('#evidence'),stamp=JSON.stringify([value.id,value.experiments,value.sources,value.assets]);
  if(box._evidenceStamp===stamp)return;box._evidenceStamp=stamp;
  const opened=new Set($$('#evidence details[open]').map(el=>el.dataset.evidenceId));
  const experiments=(value.experiments||[]).map(e=>`<article class="evidence-card"><div class="evidence-card-heading"><span class="source-id">${esc(e.id)}</span><h3>实验 · ${esc(e.hypothesis_id)}</h3><span class="evidence-kind">${e.verification?'已复核':'未复核'}</span></div><p>${e.result?`探索结果：${esc(JSON.stringify(e.result.metrics))}`:esc(e.error||'执行中')}</p>${e.verification?`<p>复核结果：${esc(JSON.stringify(e.verification.result.metrics))}</p><p>${e.verification.check.passed?'满足预注册判据':'未满足预注册判据'}</p>`:''}<details data-evidence-id="${esc(e.id)}" ${opened.has(e.id)?'open':''}><summary>查看实验代码</summary><pre>${esc(e.code||'未记录代码')}</pre></details></article>`);
  const sources=(value.sources||[]).map(p=>{
    const text=String(p.abstract||p.text||p.content||''),scope=p.evidence_scope||'证据范围未标注';let url='',host='';
    try{const parsed=new URL(p.url);if(['https:','http:'].includes(parsed.protocol)){url=parsed.href;host=parsed.hostname.replace(/^www\./,'');}}catch(_){}
    return `<article class="evidence-card"><div class="evidence-card-heading"><span class="source-id">${esc(p.id)}</span><span class="evidence-kind" title="${esc(scope)}">${esc(scope)}</span></div><h3>${esc(p.title||'未命名来源')}</h3><p class="source-excerpt">${esc(text?text.slice(0,260)+(text.length>260?'…':''):'未获取摘要或正文，当前只有元数据。')}</p>${text.length>260?`<details data-evidence-id="${esc(p.id)}" ${opened.has(p.id)?'open':''}><summary>展开完整摘录</summary><p>${esc(text)}</p></details>`:''}<div class="source-footer"><span>${esc(host||'材料来源')}</span>${url?`<a href="${esc(url)}" target="_blank" rel="noreferrer">打开来源 ↗</a>`:''}</div></article>`;
  });
  const assets=(value.assets||[]).map(a=>`<article class="evidence-card"><div class="evidence-card-heading"><span class="source-id">${esc(a.id)}</span><span class="evidence-kind">上传材料</span></div><h3>${esc(a.name)}</h3><p>${number(a.bytes)} 字节</p><details><summary>文件校验值</summary><pre>${esc(a.sha256||'未记录')}</pre></details></article>`);
  const cards=[...experiments,...sources,...assets];box.innerHTML=cards.length?`<div class="research-evidence">${cards.join('')}</div>`:empty('尚无可展示的证据','可在执行记录中查看取证过程与失败原因。');
}
renderRecords=function(){
  if(!state.runs.length){$('#records').innerHTML=empty('没有研究记录','提交研究问题后，真实决策、实验和报告会自动保存。');return;}
  $('#records').innerHTML=`<div class="table-wrap"><table><thead><tr><th>研究问题 / 任务</th><th>方式</th><th>状态</th><th>验证</th><th>模型token</th><th>耗时 / 实验</th><th></th></tr></thead><tbody>${state.runs.map(r=>`<tr><td>${esc(r.question||r.task)}<small>${esc(r.id)}</small></td><td>${esc(modeLabel(r))}</td><td>${esc(statuses[r.status]||r.status)}</td><td>${r.domain==='research'?`${r.verified_count} 项复核`:r.domain==='operator'?'旧版算子门控':r.guard?percent(r.guard.metrics.complete_at_5):'未验收'}</td><td>${number(r.budget?.total_tokens||0)}</td><td>${time(r.budget?.elapsed_seconds??r.budget?.elapsed)} / ${r.experiment_count} 项</td><td><button class="small-link" data-research-open="${esc(r.id)}">打开 ↗</button></td></tr>`).join('')}</tbody></table></div>`;
  $$('[data-research-open]').forEach(el=>el.onclick=()=>selectRun(el.dataset.researchOpen));
};
renderMemory=function(){
  previousMemory();
  const memory=state.research_memory||{active:0,versions:[]};
  $('#active-memory').textContent=`科研经验 v${memory.active}`;$('#memory-count').textContent=memory.versions.length;
  $('#research-memory-list').innerHTML=[...memory.versions].reverse().map(v=>`<article class="panel memory-card"><h2>v${v.version} · ${v.version===memory.active?'当前启用':'历史版本'}</h2><p>${esc(v.lesson)}</p><p class="body-muted">来源问题：${esc(v.question)}<br>${esc(v.limitations)}</p><button class="secondary-button" data-research-memory="${v.version}">切换到此版本</button></article>`).join('')+`<article class="panel memory-card"><h2>v0 · 无历史经验</h2><p>每次研究均依据当前问题与真实反馈。经验是方法先验，不能替代本轮实验。</p><button class="secondary-button" data-research-memory="0">回退到 v0</button></article>`;
  $$('[data-research-memory]').forEach(el=>el.onclick=async()=>{try{await api('/api/research/memory/activate',{version:Number(el.dataset.researchMemory)});await refresh(true);}catch(e){toast(e.message);}});
};
renderData=function(){previousData();$('#research-tool-list').innerHTML=(state.research_tools||[]).map(t=>`<article class="panel memory-card"><h3>${esc(t.name)}</h3><p>${esc(t.description)}</p></article>`).join('');};
let pendingFiles=[];
const TEXT_SUFFIX=/\.(csv|txt|md|json)$/i, DOC_SUFFIX=/\.(pdf|docx|pptx)$/i;
const MAX_TEXT_BYTES=500000, MAX_DOC_BYTES=20*1024*1024;
function base64Bytes(buffer){const bytes=new Uint8Array(buffer);let binary='';const chunk=0x8000;for(let i=0;i<bytes.length;i+=chunk){binary+=String.fromCharCode.apply(null,bytes.subarray(i,i+chunk));}return btoa(binary);}
function renderAssetChips(){
  $('#research-asset-list').innerHTML=pendingFiles.map((file,index)=>`<span class="asset-chip"><b>${esc(file.name)}</b><small>${Math.ceil(file.size/1024)}KB</small><button type="button" data-remove-file="${index}" aria-label="移除 ${esc(file.name)}">×</button></span>`).join('');
  $$('[data-remove-file]').forEach(button=>button.onclick=()=>{pendingFiles.splice(Number(button.dataset.removeFile),1);renderAssetChips();});
}
async function assetPayload(file){
  if(TEXT_SUFFIX.test(file.name)){if(file.size>MAX_TEXT_BYTES)throw new Error(`${file.name} 超过 500KB`);return {name:file.name,content:await file.text()};}
  if(DOC_SUFFIX.test(file.name)){if(file.size>MAX_DOC_BYTES)throw new Error(`${file.name} 超过 20MB`);return {name:file.name,content_base64:base64Bytes(await file.arrayBuffer())};}
  throw new Error(`不支持的材料类型：${file.name}`);
}
async function startGeneralResearch(){
  const question=$('#brief-input').value.trim();if(question.length<8){toast('请先描述一个具体研究问题。');return;}
  const button=$('#draft-brief');button.disabled=true;
  try{
    if(pendingFiles.length>8)throw new Error('最多8份材料');
    const assets=[];for(const file of pendingFiles)assets.push(await assetPayload(file));
    const pick=$('#research-model').value;
    const isCloud=pick.startsWith('cloud:');
    const created=await api('/api/research',{question,assets,model:isCloud?pick.slice(6):pick,model_source:isCloud?'cloud':'local',workflow_mode:$('#research-workflow').value,roles:state.settings?.roles||{},token_limit:Number($('#research-tokens').value),seconds_limit:Number($('#research-seconds').value),max_steps:Number($('#research-steps').value),use_memory:$('#research-use-memory').checked});
    pendingFiles=[];renderAssetChips();
    await selectRun(created.id);toast('研究已启动；材料已随本次运行保存。');
  }catch(error){toast(error.message);}finally{button.disabled=!!state?.active_run;}
}
$('#attach-button').addEventListener('click',()=>$('#research-files').click());
$('#research-files').addEventListener('change',()=>{
  for(const file of $('#research-files').files){
    if(!TEXT_SUFFIX.test(file.name)&&!DOC_SUFFIX.test(file.name)){toast(`不支持的材料类型：${file.name}`);continue;}
    if(pendingFiles.some(item=>item.name===file.name&&item.size===file.size))continue;
    if(pendingFiles.length>=8){toast('最多8份材料');break;}
    pendingFiles.push(file);
  }
  $('#research-files').value='';renderAssetChips();
});

/* Random topic generator: the three fixed examples are gone; every click composes
   a testable simulation question that fits the current tool surface. */
const TOPIC_PARTS={
  units:['总体均值','中位数','分位数','回归系数','相关系数','分类准确率'],
  settings:['离群值比例 0.02 至 0.10','样本量 30 与 300','缺失值比例 5% 至 20%','异方差噪声','非正态重尾分布','重复测量误差'],
  methods:['均值填补与中位数填补','普通最小二乘与稳健回归','简单随机抽样与分层抽样','固定阈值与自适应阈值'],
  metrics:['平均绝对误差','均方误差','置信区间覆盖率','排序稳定性'],
  templates:[
    '在模拟情景中比较{method}在{setting}下的{metric}差异。先注册可证伪假设与数值判据，重复模拟至少200次，实际运行并用新的随机种子独立复核，说明结论的适用条件。',
    '研究{setting}如何影响{unit}的{metric}。没有真实数据时明确采用模拟，控制生成过程，设计对照实验并由固定代码复核。',
    '比较两种估计{unit}的方法（{method}）在{setting}下的{metric}，给出预注册判据并独立重跑复核，不允许用同一批随机数当作验证。'
  ]
};
function randomTopic(){
  const pick=list=>list[Math.floor(Math.random()*list.length)];
  const template=pick(TOPIC_PARTS.templates);
  return template.replace('{method}',pick(TOPIC_PARTS.methods)).replace('{setting}',pick(TOPIC_PARTS.settings)).replace('{unit}',pick(TOPIC_PARTS.units)).replace('{metric}',pick(TOPIC_PARTS.metrics));
}
$('#random-topic').addEventListener('click',()=>{const input=$('#brief-input');input.value=randomTopic();input.dispatchEvent(new Event('input'));input.focus();});

/* --- workflow orchestration, decision chain and session metrics --- */
statuses.done='阶段完成';
const ROLE_LABELS={planner:'规划',experimenter:'实验',verifier:'复核',reporter:'报告'};
function roleLabel(role){return ROLE_LABELS[role]||role||'—';}
let sessionTab='overview', sessionViewId='', chainView={runId:'',turn:'all',status:'all',query:''};
const TOOL_LABELS={get_memory_context:'读取研究经验',inspect:'检查材料',design_workflow:'设计研究流程',plan:'制定研究方案',literature:'检索文献',platform_search:'检索平台',web_search:'搜索网页',web_fetch:'读取网页',extract_document:'提取文档',read:'读取材料',hypothesis:'注册假设',experiment:'运行实验',verify:'复核实验',reflect:'评估结果',finish:'形成结论',invalid:'决策失败',remember_fact:'保存事实',recall_facts:'查找事实'};
function setSessionTab(name){
  sessionTab=['overview','execution','evidence'].includes(name)?name:'overview';
  $$('[data-session-tab]').forEach(button=>{const selected=button.dataset.sessionTab===sessionTab;button.setAttribute('aria-selected',String(selected));button.tabIndex=selected?0:-1;});
  $$('#page-session .session-view').forEach(panel=>panel.classList.toggle('hidden',panel.id!==`session-view-${sessionTab}`));
}
function renderSessionLayout(value){
  if(!value)return;
  if(sessionViewId!==value.id){sessionViewId=value.id;setSessionTab(value.domain==='research'?'overview':'execution');$('.session-raw-records').open=value.domain!=='research';}
  $('#session-title').textContent=value.question||value.task||'研究会话';
  const created=new Date(value.created_at);
  $('#session-question').textContent=[Number.isNaN(created.getTime())?'':created.toLocaleString('zh-CN',{month:'long',day:'numeric',hour:'2-digit',minute:'2-digit',hour12:false}),value.turns?.length?`${value.turns.length} 轮对话`:'历史运行',value.id].filter(Boolean).join(' · ');
  $('#session-step-count').textContent=value.steps?.length||0;
  $('#session-evidence-count').textContent=(value.sources?.length||0)+(value.assets?.length||0)+(value.experiments?.length||0);
  $('#session-report').href=`/api/runs/${value.id}/report`;$('#session-export').href=`/api/runs/${value.id}/export`;
  const pending=['running','queued'].includes(value.status);$('#session-report').classList.toggle('hidden',pending||value.domain!=='research');$('#session-archive').disabled=pending;
  if(value.domain==='research')$('#experiments').innerHTML='';
}
function renderWorkflow(run){
  const panel=$('#workflow-panel');
  if(!run||run.domain!=='research'||run.workflow_mode!=='plan'){panel.classList.add('hidden');panel.innerHTML='';return;}
  panel.classList.remove('hidden');
  const stages=run.workflow?.stages||[],turn=run.turns?.at(-1)?.id,agents=(run.agents||[]).filter(a=>!a.turn||!turn||a.turn===turn),calls=(run.calls||[]).filter(c=>!c.turn||!turn||c.turn===turn);
  const byAgent={};calls.forEach(c=>{if(c.agent)byAgent[c.agent]=(byAgent[c.agent]||0)+1;});
  panel.innerHTML=`<div class="panel-header"><h2>本轮研究流程</h2><span class="workflow-source">${esc(run.workflow?.source==='agent_tool'?'Agent 设计':run.workflow?.source==='model'?'模型规划':run.workflow?.source==='model_retry'?'模型重试':'默认规划')} · ${stages.length} 阶段</span></div>${run.workflow?.reason?`<p class="workflow-reason">${esc(run.workflow.reason)}</p>`:''}<div class="workflow-stages">${stages.map((s,i)=>{const a=agents.filter(x=>x.id==='g'+(i+1)).at(-1)||{};return `<article class="workflow-stage${a.status==='running'?' active':''}"><div class="workflow-stage-heading"><span class="workflow-index">${String(i+1).padStart(2,'0')}</span><h3>${esc(roleLabel(s.role))}</h3><span class="workflow-stage-status">${esc(statuses[a.status]||'未开始')}</span></div><p>${esc(s.goal)}</p>${s.why?`<details class="stage-why"><summary>设计依据</summary><p>${esc(s.why)}</p></details>`:''}<small>${esc(a.model||'尚未调用模型')} · ${byAgent[a.id]||0} 次调用</small></article>`;}).join('')}</div>`;
}
function renderChain(run){
  const panel=$('#chain-panel');
  if(!run||run.domain!=='research'){panel.classList.add('hidden');return;}
  panel.classList.remove('hidden');
  const calls=run.calls||[],steps=run.steps||[],box=$('#chain');
  if(chainView.runId!==run.id){chainView={runId:run.id,turn:'all',status:'all',query:''};$('#chain-status').value='all';$('#chain-search').value='';box._chainStamp='';}
  const turnNames=[...new Set([...(run.turns||[]).map(t=>t.id),...steps.map(s=>s.turn).filter(Boolean)])];
  const options='<option value="all">全部轮次</option>'+turnNames.map(t=>`<option value="${esc(t)}">第 ${esc(t.replace(/^t/,''))} 轮</option>`).join('');
  if($('#chain-turn').innerHTML!==options){$('#chain-turn').innerHTML=options;$('#chain-turn').value=chainView.turn;}
  const scopedSteps=steps.filter(s=>chainView.turn==='all'||s.turn===chainView.turn);
  const scopedCalls=calls.filter(c=>chainView.turn==='all'||c.turn===chainView.turn);
  const filtered=scopedSteps.filter(s=>(chainView.status==='all'||s.status===chainView.status)&&(!chainView.query||JSON.stringify(s).toLowerCase().includes(chainView.query.toLowerCase())));
  const known=scopedCalls.reduce((total,c)=>total+(c.input_tokens||0)+(c.output_tokens||0),0),unknown=scopedCalls.reduce((total,c)=>total+(c.unknown_reserved||0),0);
  const failures=scopedSteps.filter(s=>s.status==='failed').length;
  $('#chain-meta').textContent=`${scopedSteps.length} 步 · ${failures} 项失败 · ${scopedCalls.length} 次模型调用`;
  $('#chain-usage-note').textContent=`${chainView.turn==='all'?'会话累计':'本轮'}模型用量 ${number(known+unknown)} token${unknown?`（含 ${number(unknown)} 未知用量预留）`:''} · 每步耗时为工具执行时间`;
  const stamp=JSON.stringify([run.id,chainView,filtered]);if(box._chainStamp===stamp)return;
  const open=new Set($$('details.chain-step[open]').map(el=>el.dataset.stepKey));
  box._chainStamp=stamp;
  box.innerHTML=filtered.length?filtered.map((s,index)=>{
    const key=`${s.turn||'legacy'}:${s.id}`,failed=s.status==='failed',seconds=typeof s.elapsed_seconds==='number'?s.elapsed_seconds:null;
    const duration=s.tool==='invalid'?'未执行工具':seconds==null?'耗时未记录':`工具 ${seconds<1?seconds.toFixed(2):seconds.toFixed(1)}s`;
    const status=failed?'失败':s.status==='completed'?'已执行':statuses[s.status]||s.status||'状态未记录';
    const rawReason=s.reason||'',reason=/^(模型自主选择|控制器执行确定性检查|按通用研究协议进入)/.test(rawReason)?'':rawReason,error=typeof s.observation?.error==='string'?s.observation.error:'';
    return `<details class="chain-step ${esc(s.status||'')}" data-step-key="${esc(key)}" ${open.has(key)?'open':''}><summary class="chain-summary"><span class="chain-index">${esc(s.id?.replace(/^s/,'')||String(index+1))}</span><span class="chain-content"><span class="chain-heading"><strong class="chain-tool" title="${esc(s.tool)}">${esc(TOOL_LABELS[s.tool]||s.tool)}</strong><span class="chain-status ${failed?'failed':''}">${esc(status)}</span><span class="chain-agent" title="${esc(s.agent||'')}">${esc(s.source==='controller'?'控制器':roleLabel(s.role))}</span></span><span class="chain-reason">${esc(reason)}</span>${error?`<span class="chain-error">${esc(error.slice(0,220))}</span>`:''}</span><span class="chain-duration">${esc(duration)}</span><span class="chain-chevron" aria-hidden="true">⌄</span></summary><div class="chain-body"><div class="chain-io"><section><h3>输入参数</h3><pre>${esc(JSON.stringify(s.args??{},null,2))}</pre></section><section><h3>执行结果</h3><pre>${esc(JSON.stringify(s.observation??{},null,2))}</pre></section></div><div class="chain-detail-meta"><span>${esc(s.id)}${s.turn?` · ${esc(s.turn)}`:''} · ${esc(s.tool)}</span><button type="button" class="small-link" data-chain-raw="${esc(key)}">完整步骤</button></div></div></details>`;
  }).join(''):empty(scopedSteps.length?'没有匹配的步骤':'尚无执行记录',scopedSteps.length?'调整轮次、状态或关键词。':'开始研究后，实际工具调用会显示在这里。');
  $$('[data-chain-raw]').forEach(button=>button.onclick=()=>showDetail('步骤记录',steps.find(s=>`${s.turn||'legacy'}:${s.id}`===button.dataset.chainRaw)));
}
function renderBudgetResearch(){
  if(!run||run.domain!=='research')return;
  const b=run.budget,box=$('#budget');
  if(!b){if(!box.querySelector('.budget-pending'))box.innerHTML='<div class="budget-item"><div class="budget-label">等待首个模型调用</div></div>';return;}
  if(!box.querySelector('#budget-calls')){
    box.innerHTML=`<div class="budget-item"><div class="budget-label">模型 token <strong id="budget-token-text">—</strong></div><div class="progress-track"><div class="progress-fill" id="budget-token-bar" style="width:0%"></div></div></div><div class="budget-item"><div class="budget-label">墙钟时间 <strong id="budget-time-text">—</strong></div><div class="progress-track"><div class="progress-fill purple" id="budget-time-bar" style="width:0%"></div></div></div><div class="budget-metrics"><div class="budget-row"><span>模型调用</span><strong id="budget-calls">—</strong></div><div class="budget-row"><span>生成速度</span><strong id="budget-speed">—</strong></div><div class="budget-row"><span>缓存命中</span><strong id="budget-cached">—</strong></div><div class="budget-row"><span>上下文用量</span><strong id="budget-context">—</strong></div><div class="budget-row"><span>云端调用</span><strong id="budget-cloud">—</strong></div><div class="budget-row"><span>工具耗时</span><strong id="budget-tools">—</strong></div></div>`;
  }
  const turn=run.turns?.at(-1)?.id;
  const calls=(run.calls||[]).filter(call=>!turn||call.turn===turn);
  const out=calls.reduce((sum,call)=>sum+(call.output_tokens||0),0);
  const wall=calls.reduce((sum,call)=>sum+(call.wall_seconds||0),0);
  const cached=b.cached_tokens||0;
  const context=(calls[calls.length-1]||{}).context_tokens||0;
  const speed=wall?out/wall:0;
  const set=(id,value)=>{const el=document.getElementById(id);if(el&&el.textContent!==value)el.textContent=value;};
  const bar=(id,percent)=>{const el=document.getElementById(id);if(!el)return;const width=Math.min(100,Math.max(0,percent))+'%';if(el.style.width!==width)el.style.width=width;};
  set('budget-token-text',`${number(b.total_tokens)} / ${number(b.token_limit)}`);
  bar('budget-token-bar',b.total_tokens/b.token_limit*100);
  set('budget-time-text',`${time(Math.max(b.elapsed_seconds||0,wall))} / ${time(b.seconds_limit)}`);
  bar('budget-time-bar',(b.elapsed_seconds||0)/b.seconds_limit*100);
  set('budget-calls',`${b.model_calls||0} 次`);
  set('budget-speed',`${speed.toFixed(1)} tok/s`);
  set('budget-cached',`${number(cached)} token`);
  set('budget-context',calls.at(-1)?.context_tokens==null?'未返回':`${number(context)} token`);
  set('budget-cloud',`${b.cloud_calls||0} 次`);
  const toolSeconds=(run.steps||[]).filter(step=>!turn||step.turn===turn).reduce((sum,step)=>sum+(step.elapsed_seconds||0),0);
  set('budget-tools',`${toolSeconds.toFixed(2)} s`);
}
const baseResearchRun=renderRun;
renderRun=function(){baseResearchRun();renderWorkflow(run);renderChain(run);};
const baseBudget=renderBudget;
renderBudget=function(){if(run&&run.domain==='research'){renderBudgetResearch();return;}baseBudget();};

/* Multi-turn: the session page shows every turn and lets the user continue. */
function renderSessionTurns(run){
  const box=$('#session-turns');
  if(!box)return;
  renderSessionLayout(run);
  if(!run||run.domain!=='research'){box.innerHTML='';return;}
  const turns=run.turns?.length?run.turns:[{question:run.question,status:run.status,conclusion:run.report?.conclusion||run.stop_reason||run.error||''}];
  const stamp=JSON.stringify([run.id,turns]);if(box._turnStamp===stamp)return;
  const historyOpen=box.querySelector('.session-turn-history')?.open;box._turnStamp=stamp;
  const card=(turn,index)=>{
    const status=statuses[turn.status]||turn.status||'',question=index>0||turn.question!==run.question;
    return `<article class="turn-block">${question?`<div class="turn-question"><span class="turn-role">追问</span><p>${esc(turn.question||'')}</p></div>`:''}<div class="turn-answer"><div class="turn-answer-heading"><strong>${index===turns.length-1?'研究结果':'历史结果'}</strong><span>第 ${index+1} 轮</span><span class="pill subtle">${esc(status)}</span></div><p class="${turn.conclusion?'':'muted'}">${esc(turn.conclusion||(['running','queued'].includes(turn.status)?'研究正在进行，执行记录中会显示最新步骤。':'本轮尚未形成结论。'))}</p></div></article>`;
  };
  box.innerHTML=(turns.length>1?`<details class="session-turn-history" ${historyOpen?'open':''}><summary>此前 ${turns.length-1} 轮对话</summary>${turns.slice(0,-1).map(card).join('')}</details>`:'')+card(turns.at(-1),turns.length-1);
}
const baseTurnsRun=renderRun;
renderRun=function(){baseTurnsRun();renderSessionTurns(run);};
$$('[data-session-tab]').forEach(button=>{
  button.addEventListener('click',()=>setSessionTab(button.dataset.sessionTab));
  button.addEventListener('keydown',event=>{if(!['ArrowLeft','ArrowRight','Home','End'].includes(event.key))return;event.preventDefault();const tabs=$$('[data-session-tab]');let index=tabs.indexOf(button);index=event.key==='Home'?0:event.key==='End'?tabs.length-1:(index+(event.key==='ArrowRight'?1:-1)+tabs.length)%tabs.length;setSessionTab(tabs[index].dataset.sessionTab);tabs[index].focus();});
});
$('#chain-turn').addEventListener('change',event=>{chainView.turn=event.target.value;renderChain(run);});
$('#chain-status').addEventListener('change',event=>{chainView.status=event.target.value;renderChain(run);});
$('#chain-search').addEventListener('input',event=>{chainView.query=event.target.value.trim();renderChain(run);});
$('#followup-send')?.addEventListener('click',async()=>{
  const question=$('#followup-input').value.trim();
  if(!run){toast('先打开一个研究会话');return;}
  if(question.length<8){toast('追问至少 8 个字符');return;}
  const button=$('#followup-send');button.disabled=true;
  try{
    await api(`/api/runs/${run.id}/continue`,{question});
    $('#followup-input').value='';
    toast('已在本会话中开始下一轮研究。');
    await refresh(true);
  }catch(error){toast(error.message);}finally{button.disabled=false;}
});
