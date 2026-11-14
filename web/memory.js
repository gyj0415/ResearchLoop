/* Facts, preferences and versioned method lessons share one memory page. */
let memoryFilter='all', memorySearch='';
function renderMemory(){
  const value=state?.memory_system||{facts:[],methods:{active:0,versions:[]}},methods=value.methods;
  $('#memory-count').textContent=(value.facts||[]).filter(row=>row.status==='active').length+methods.versions.length;
  $('#memory-summary').textContent='新研究会自动召回相关记忆。你可以添加偏好、停用过时内容，或回退研究方法版本。';
  const stamp=JSON.stringify([value,memoryFilter,memorySearch]);const box=$('#memory-cards');if(box._stamp===stamp)return;box._stamp=stamp;
  const matches=item=>!memorySearch||JSON.stringify(item).toLowerCase().includes(memorySearch.toLowerCase());
  const sourceLink=(id)=>id&&state.runs.some(r=>r.id===id)?`<button class="small-link" data-memory-source="${esc(id)}">查看来源会话</button>`:id?`<span>来源 ${esc(id)} · 会话已移除</span>`:'<span>用户手动添加</span>';
  const labels={active:'自动召回',disabled:'已停用',stale:'已过期',contradicted:'已有冲突记录'};
  const cards=[];
  if(memoryFilter!=='methods')for(const row of value.facts||[]){
    if(!matches(row))continue;
    const preference=row.predicate==='user_preference',manageable=['active','disabled'].includes(row.status);
    cards.push(`<article class="memory-entry ${row.status==='active'?'':'is-disabled'}"><div class="memory-entry-head"><span class="memory-kind">${preference?'使用偏好':'事实记忆'}</span><span class="pill subtle">${esc(labels[row.status]||row.status)}</span></div><p class="memory-content">${esc(row.summary)}</p>${row.tags?.length?`<div class="memory-tags">${row.tags.map(t=>`<span>${esc(t)}</span>`).join('')}</div>`:''}<details><summary>来源与记录</summary><p>${esc(row.source_message||'未附来源摘要')}</p><small>最后确认 ${esc(new Date(row.last_confirmed_at||row.created_at).toLocaleString('zh-CN'))}</small></details><div class="memory-entry-footer">${sourceLink(row.session_id)}${manageable?`<button class="secondary-button" data-memory-toggle="${row.id}" data-enable="${row.status!=='active'}">${row.status==='active'?'停用':'启用'}</button>`:''}</div></article>`);
  }
  if(memoryFilter!=='facts')for(const item of [...methods.versions].reverse()){
    if(!matches(item))continue;
    cards.push(`<article class="memory-entry"><div class="memory-entry-head"><span class="memory-kind">研究方法 · v${item.version}</span><span class="pill subtle">${item.version===methods.active?'当前启用':'历史版本'}</span></div><p class="memory-content">${esc(item.lesson)}</p><p class="memory-scope">适用问题：${esc(item.question||'未注明')}<br>${esc(item.limitations||'仍需在新任务验证')}</p><div class="memory-entry-footer">${sourceLink(item.source_run)}${item.version!==methods.active?`<button class="secondary-button" data-method-version="${item.version}">启用此版本</button>`:`<button class="secondary-button" data-method-version="0">暂停方法记忆</button>`}</div></article>`);
  }
  box.innerHTML=cards.join('')||empty(memorySearch?'没有匹配的记忆':'还没有记忆','可添加希望 Agent 记住的偏好；研究中记录的事实和有依据的方法也会出现在这里。');
  $('#memory-method-state').textContent=methods.active?`方法记忆 v${methods.active} 已启用；仍按当前问题检查相关性。`:'方法记忆尚未启用，事实与偏好仍会自动召回。';
  $$('[data-memory-toggle]').forEach(button=>button.onclick=async()=>{try{await api('/api/memory/toggle',{id:Number(button.dataset.memoryToggle),enabled:button.dataset.enable==='true'});await refresh(true);}catch(error){toast(error.message);}});
  $$('[data-method-version]').forEach(button=>button.onclick=async()=>{try{await api('/api/research/memory/activate',{version:Number(button.dataset.methodVersion)});await refresh(true);}catch(error){toast(error.message);}});
  $$('[data-memory-source]').forEach(button=>button.onclick=()=>selectRun(button.dataset.memorySource));
}
$$('[data-memory-filter]').forEach(button=>button.onclick=()=>{memoryFilter=button.dataset.memoryFilter;$$('[data-memory-filter]').forEach(item=>{item.classList.toggle('active',item===button);item.setAttribute('aria-pressed',String(item===button));});renderMemory();});
$('#memory-search').addEventListener('input',event=>{memorySearch=event.target.value.trim();renderMemory();});
$('#memory-add').addEventListener('click',()=>$('#memory-dialog').showModal());
$('#memory-form').addEventListener('submit',async event=>{event.preventDefault();const button=event.submitter;button.disabled=true;try{const form=new FormData(event.target);await api('/api/memory/add',Object.fromEntries(form));$('#memory-dialog').close();event.target.reset();await refresh(true);toast('已保存，后续研究会自动召回相关内容。');}catch(error){toast(error.message);}finally{button.disabled=false;}});
