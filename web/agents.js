/* Session permissions, capacity-tier orchestration, child traces and recovery. */
Object.assign(TOOL_LABELS,{delegate_research:'委派并行子任务',read_subagents:'读取子任务',select_model:'选择模型档位',list_files:'查看工作目录',read_file:'读取文件',write_file:'写入文件',run_command:'执行命令',command_experiment:'运行工程实验'});
const tierName=tier=>({light:'轻量',strong:'高能力',auto:'自动'}[tier]||tier||'自动');
const categoryName=category=>({workspace_write:'本会话的工作区文件写入',command_powershell:'本会话的 PowerShell 命令',command_cmd:'本会话的 CMD 命令',command_bash:'本会话的 Bash 命令'}[category]||category);
const accessNote=document.createElement('p');accessNote.className='form-note access-note';
$('#research-access').closest('.composer-box').append(accessNote);
function updateAccessNote(){accessNote.textContent=$('#research-access').value==='full'?'完全访问会自动允许本会话的文件写入和命令。命令以本机用户身份执行。':'文件读写位于独立工作目录；写入和命令执行前请求批准。可按单次或本会话同类操作授权。';}
$('#research-access').addEventListener('change',updateAccessNote);updateAccessNote();

function renderApprovals(value){
  const box=$('#session-approvals');if(!box)return;
  const pending=(value?.approval_requests||[]).filter(r=>r.status==='pending');
  const stamp=JSON.stringify([value?.id,pending]);if(box._stamp===stamp)return;box._stamp=stamp;
  box.innerHTML=pending.map(request=>{const info=request.details,file=info.operation==='write_file';return `<section class="approval-card"><div class="approval-heading"><strong>${file?'写入文件':'执行 '+esc(info.shell)+' 命令'}${request.subagent?' · '+esc(request.subagent):''}</strong><span>${esc(categoryName(request.category))}</span></div><p class="approval-path">${esc(info.path||info.cwd)}${info.overwrites?' · 将覆盖已有文件':''}</p><pre>${esc(file?info.content:info.command)}</pre><div class="approval-actions"><button type="button" class="primary-button" data-approve="${esc(request.id)}" data-decision="once">批准一次</button><button type="button" class="secondary-button" data-approve="${esc(request.id)}" data-decision="category">允许本会话同类</button><button type="button" class="small-link" data-approve="${esc(request.id)}" data-decision="deny">拒绝</button></div></section>`;}).join('');
  box.querySelectorAll('[data-approve]').forEach(button=>button.onclick=async()=>{
    const id=value.id;box.querySelectorAll('button').forEach(b=>b.disabled=true);
    try{await api(`/api/runs/${id}/approve`,{request_id:button.dataset.approve,decision:button.dataset.decision});await refresh(true);}
    catch(error){toast(error.message);box.querySelectorAll('button').forEach(b=>b.disabled=false);}
  });
}

const childCache=new Map(),childOpen=new Map();
function childTraceHtml(value,key){
  const opened=childOpen.get(key)||new Set();
  return (value.trace||[]).map(node=>`<details class="child-call" data-child-call="${esc(node.id)}" ${opened.has(node.id)?'open':''}><summary><span>${node.kind==='model_call'?'模型':'工具'}</span><strong>${esc(node.model||TOOL_LABELS[node.tool]||node.tool||node.kind)}</strong><span>${esc(statuses[node.status]||node.status)}${node.route?.tier?' · '+esc(tierName(node.route.tier)):''}</span></summary><div class="chain-io"><section><h3>输入</h3><pre>${esc(JSON.stringify(node.input||{},null,2))}</pre></section><section><h3>输出</h3><pre>${esc(JSON.stringify(node.output||{},null,2))}</pre></section></div></details>`).join('')||'<p class="muted">子任务尚未产生调用。</p>';
}
function renderSubagentNode(value,node,index,opened){
  const live=(value.subagent_live||[]).find(item=>item.subagent_id===node.subagent_id)||{},status=live.status||node.status;
  const key=value.id+':'+node.subagent_id,cached=childCache.get(key),stepKey=`${node.turn||'legacy'}:${node.id}`;
  return `<details class="chain-step chain-subagent ${esc(status)}" data-step-key="${esc(stepKey)}" data-subagent="${esc(node.subagent_id)}" ${opened.has(stepKey)?'open':''}><summary class="chain-summary"><span class="chain-index">${index+1}</span><span class="chain-content"><span class="chain-heading"><strong>子 Agent · ${esc(node.subagent_id)}</strong><span class="invocation-kind">${esc(tierName(node.model_tier))}</span><span class="chain-status">${esc(statuses[status]||status)}</span><span class="chain-agent">${esc(roleLabel(node.role))}</span></span><span class="chain-reason">${esc(node.reason)}</span></span><span class="chain-duration">${number(live.budget?.total_tokens||node.usage?.total_tokens||0)} token</span><span class="chain-chevron">⌄</span></summary><div class="chain-body"><div class="subagent-calls">${cached?childTraceHtml(cached.value,key):'<p class="muted">展开后读取子 Agent 调用记录。</p>'}</div></div></details>`;
}
function bindAgentChain(value,records){
  $('#chain').querySelectorAll('[data-subagent]').forEach(panel=>{
    const load=async()=>{
      if(!panel.open)return;
      const id=panel.dataset.subagent,key=value.id+':'+id,stamp=(value.subagent_live||[]).find(i=>i.subagent_id===id)?.updated_at;
      let cached=childCache.get(key);
      try{
        if(!cached||cached.stamp!==stamp){const child=await api(`/api/runs/${value.id}/subagents/${id}`);cached={stamp,value:child};childCache.set(key,cached);}
        if(!panel.isConnected||run?.id!==value.id)return;
        panel.querySelector('.subagent-calls').innerHTML=childTraceHtml(cached.value,key);
        panel.querySelectorAll('[data-child-call]').forEach(details=>details.addEventListener('toggle',()=>{const set=childOpen.get(key)||new Set();details.open?set.add(details.dataset.childCall):set.delete(details.dataset.childCall);childOpen.set(key,set);}));
      }catch(error){if(panel.isConnected)panel.querySelector('.subagent-calls').textContent=error.message;}
    };
    panel.addEventListener('toggle',load);if(panel.open)load();
  });
  $('#chain').querySelectorAll('[data-fork-node]').forEach(fork=>{
    const retry=document.createElement('button');retry.type='button';retry.className='secondary-button';retry.textContent='从此处重试';
    retry.onclick=()=>openRecovery('retry',fork.dataset.forkNode);fork.before(retry);
  });
}

const recoverButton=document.createElement('button');recoverButton.id='session-recover';recoverButton.type='button';recoverButton.className='secondary-button hidden';recoverButton.textContent='补充预算继续';
$('#session-resume').after(recoverButton);recoverButton.onclick=()=>openRecovery('continue');
const dialog=document.createElement('dialog');dialog.id='recovery-dialog';
dialog.innerHTML=`<form id="recovery-form"><div class="dialog-heading"><h2 id="recovery-title">继续研究</h2><button type="button" class="icon-button" aria-label="关闭">×</button></div><p id="recovery-description" class="form-note"></p><label>后续要求（可选）<textarea id="recovery-direction" rows="2" maxlength="2000" placeholder="例如：更换来源，重点补充反对证据"></textarea></label><label class="check-row"><input type="checkbox" id="recovery-unlimited">不限预算</label><div class="form-grid three"><label>追加 token / k<input id="recovery-tokens" type="number" min="1" value="30"></label><label>追加时间 / 秒<input id="recovery-seconds" type="number" min="1" placeholder="不限"></label><label>追加步骤<input id="recovery-steps" type="number" min="1" value="12"></label></div><div class="dialog-actions"><button type="submit" class="primary-button">继续研究</button></div></form>`;
document.body.append(dialog);dialog.querySelector('.icon-button').onclick=()=>dialog.close();
let recoverySelection=null;
function openRecovery(action,nodeId=''){
  if(!run)return;recoverySelection={action,nodeId,runId:run.id};
  $('#recovery-title').textContent=action==='retry'?'从调用前重试':'补充预算继续';
  $('#recovery-description').textContent=action==='retry'?'从前一检查点创建分支，保留原调用记录；不会撤销命令已经产生的外部副作用。分支重新请求文件与命令批准。':'沿用已有证据继续研究，历史消耗保留。暂停时扩充本轮额度；已结束的研究开启下一轮，以下为新增额度。';
  $('#recovery-direction').value='';$('#recovery-unlimited').checked=false;$('#recovery-tokens').value=30;$('#recovery-seconds').value='';$('#recovery-steps').value=12;
  dialog.querySelectorAll('input[type=number]').forEach(i=>i.disabled=false);dialog.showModal();
}
$('#recovery-unlimited').onchange=()=>dialog.querySelectorAll('input[type=number]').forEach(i=>i.disabled=$('#recovery-unlimited').checked);
$('#recovery-form').onsubmit=async event=>{
  event.preventDefault();if(!recoverySelection)return;
  const selection=recoverySelection,button=event.submitter,unlimited=$('#recovery-unlimited').checked;
  const body={node_id:selection.nodeId,question:$('#recovery-direction').value.trim(),budget_enabled:!unlimited,
    token_limit:unlimited?null:Math.round(Number($('#recovery-tokens').value)*1000),
    seconds_limit:unlimited||!$('#recovery-seconds').value?null:Number($('#recovery-seconds').value),
    max_steps:unlimited||!$('#recovery-steps').value?null:Number($('#recovery-steps').value)};
  if(!unlimited&&(!body.token_limit||body.token_limit<1)){toast('请填写新增 token 额度或选择不限预算');return;}
  button.disabled=true;try{const created=await api(`/api/runs/${selection.runId}/${selection.action==='retry'?'retry':'recover'}`,body);dialog.close();await selectRun(created.id);toast(selection.action==='retry'?'已从调用前创建重试分支。':'已继续研究，历史消耗保留。');}catch(error){toast(error.message);}finally{button.disabled=false;}
};
const baseAgentRun=renderRun;
renderRun=function(){baseAgentRun();renderApprovals(run);recoverButton.classList.toggle('hidden',!run||run.domain!=='research'||!['paused','incomplete','inconclusive','budget_stopped','failed','interrupted','cancelled'].includes(run.status));};
