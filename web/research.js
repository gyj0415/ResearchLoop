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
$('#page-data .subtitle').textContent='当前支持文献元数据、文本与CSV材料、可执行数值实验和固定判据复核。扩展工具不改变研究主循环。';
$('.protocol-callout strong').textContent='评价研究过程，而不只看报告是否写完';
$('.protocol-callout p').textContent='记录证据、可证伪假设、真实实验与独立复核。比较历史经验时保持同问题、材料、模型和预算；复核完成也可能得到否定结论。';
$('#log-source').insertAdjacentHTML('afterbegin','<option value="research">科研 Agent</option>');

renderTop=function(){
  previousTop();
  $('#dataset-summary').textContent=`${state.research_tools?.length||0} 类工具 · 文献检索 / 材料读取 / Python实验 / 独立复核 · 缺少工具或证据时明确停止`;
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
  const report=run.report;
  $('#evidence').innerHTML=`${report?`<div class="research-conclusion"><h3>Agent分析与局限</h3><p>${esc(report.conclusion)}</p><small>引用：${esc(report.citations.join(', '))}。以以下实测记录为准。</small></div>`:''}<div class="research-evidence">${run.experiments.map(e=>`<article><h3>${esc(e.id)} · ${esc(e.hypothesis_id)}</h3><p>${e.result?`探索实测：${esc(JSON.stringify(e.result.metrics))}`:esc(e.error||'执行中')}</p><p>${e.verification?`独立重跑：${esc(JSON.stringify(e.verification.result.metrics))} · ${e.verification.check.passed?'满足预注册判据':'未满足预注册判据'}`:'尚未独立复核'}</p><details><summary>冻结代码与判据</summary><pre>${esc(e.code)}</pre></details></article>`).join('')}${run.sources.map(p=>`<article><h3>${esc(p.id)} · ${esc(p.title)}</h3><p>${esc(p.abstract||'无摘要，不能据此验证论文结论。')}</p><a href="${esc(p.url)}" target="_blank" rel="noreferrer">DOI来源 ↗</a></article>`).join('')}${run.assets.map(a=>`<article><h3>${esc(a.id)} · ${esc(a.name)}</h3><p>${number(a.bytes)}字节 · SHA256 ${esc(a.sha256.slice(0,16))}…</p></article>`).join('')}</div>${!run.experiments.length&&!run.sources.length&&!run.assets.length?empty('尚无研究证据','查看上方研究行动；预算耗尽时会保留未完成状态。'):''}`;
};
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
function renderWorkflow(run){
  const panel=$('#workflow-panel');
  if(!run||run.domain!=='research'||run.workflow_mode!=='plan'){panel.classList.add('hidden');panel.innerHTML='';return;}
  panel.classList.remove('hidden');
  const stages=run.workflow?.stages||[],agents=run.agents||[],calls=run.calls||[];
  const byAgent={};calls.forEach(c=>{if(c.agent)byAgent[c.agent]=(byAgent[c.agent]||0)+1;});
  const planner=agents.find(a=>a.id==='g0');
  panel.innerHTML=`<div class="panel-header"><div><div class="section-label">WORKFLOW ORCHESTRATION</div><h2>工作流编排 · 子 Agent</h2></div><span class="pill subtle">${esc(run.workflow?.source==='model'?'模型规划':'默认规划')} · ${stages.length} 阶段</span></div><div class="workflow-stages">${planner?`<article class="workflow-stage"><h3>g0 · 规划</h3><p>生成工作流：${esc(stages.map(s=>roleLabel(s.role)).join(' → ')||'—')}</p><small>${esc(planner.model||'')} · ${esc(planner.model_source||'')} · ${byAgent.g0||0} 次模型调用</small></article>`:''}${stages.map((s,i)=>{const a=agents.find(x=>x.id==='g'+(i+1))||{};return `<article class="workflow-stage${a.status==='running'?' active':''}"><h3>${esc(a.id||('g'+(i+1)))} · ${esc(roleLabel(s.role))}</h3><p>${esc(s.goal)}</p><small>${esc(a.model||'—')} · ${esc(a.model_source||'—')} · ${byAgent[a.id]||0} 次调用 · 上限 ${s.max_steps} 步</small></article>`;}).join('')}</div>${run.workflow?.reason?`<p class="form-note" style="padding:0 22px 16px">${esc(run.workflow.reason)}</p>`:''}`;
}
function renderChain(run){
  const panel=$('#chain-panel');
  if(!run||run.domain!=='research'){panel.classList.add('hidden');return;}
  panel.classList.remove('hidden');
  const calls=run.calls||[],steps=run.steps||[];
  const byAgent={};calls.forEach(c=>{const key=c.agent||'main';byAgent[key]=byAgent[key]||{calls:0,tokens:0};byAgent[key].calls++;byAgent[key].tokens+=(c.input_tokens||0)+(c.output_tokens||0);});
  const b=run.budget||{};
  $('#chain-meta').textContent=`${steps.length} 步 · ${calls.length} 次模型调用 · ${number(b.total_tokens||0)} token`;
  $('#chain').innerHTML=steps.length?steps.map(s=>{const m=byAgent[s.agent]||{};const secs=s.elapsed_seconds!=null?`${s.elapsed_seconds.toFixed(2)}s`:'';return `<div class="chain-step ${esc(s.status||'')}"><div class="chain-head"><span class="pill subtle">${esc(s.id)}</span><span class="chain-tool">${esc(s.tool)}</span>${s.agent?`<span class="chain-agent">${esc(s.agent)} · ${esc(roleLabel(s.role))}</span>`:''}<strong>${esc((s.reason||'').slice(0,90))}</strong><span class="chain-metrics">${s.source==='controller'?'<span>控制器</span>':''}${secs?`<span>${secs}</span>`:''}${m.tokens?`<span>${number(m.tokens)} tok</span>`:''}</span></div><div class="chain-body"><details><summary>参数与观察</summary><pre>${esc(JSON.stringify({args:s.args,observation:s.observation},null,2))}</pre></details></div></div>`;}).join(''):empty('尚无决策步骤','启动研究后，每一步工具选择、参数、观察和耗时都会出现在这里。');
}
function renderBudgetResearch(){
  const b=run?.budget;
  if(!b){$('#budget').innerHTML='<div class="budget-item"><div class="budget-label">等待首个模型调用</div></div>';return;}
  const calls=run.calls||[];const out=calls.reduce((a,c)=>a+(c.output_tokens||0),0);const wall=calls.reduce((a,c)=>a+(c.wall_seconds||0),0);
  const cached=b.cached_tokens||0;const context=(calls[calls.length-1]||{}).context_tokens||0;
  const speed=wall?out/wall:0;
  $('#budget').innerHTML=`<div class="budget-item"><div class="budget-label">模型 token <strong>${number(b.total_tokens)} / ${number(b.token_limit)}</strong></div><div class="progress-track"><div class="progress-fill" style="width:${Math.min(100,b.total_tokens/b.token_limit*100)}%"></div></div></div><div class="budget-item"><div class="budget-label">墙钟时间 <strong>${time(Math.max(b.elapsed_seconds||0,wall))} / ${time(b.seconds_limit)}</strong></div><div class="progress-track"><div class="progress-fill purple" style="width:${Math.min(100,(b.elapsed_seconds||0)/b.seconds_limit*100)}%"></div></div></div><div class="budget-footer"><span class="small-dot"></span>${b.model_calls||0} 次调用 · ${speed.toFixed(1)} tok/s · 缓存命中 ${number(cached)} · 最近上下文 ${number(context)}${b.cloud_calls?` · 云端 ${b.cloud_calls} 次`:''}</div>`;
}
const baseResearchRun=renderRun;
renderRun=function(){baseResearchRun();renderWorkflow(run);renderChain(run);renderBudgetResearch();};
const baseBudget=renderBudget;
renderBudget=function(){if(run&&run.domain==='research'){renderBudgetResearch();return;}baseBudget();};
