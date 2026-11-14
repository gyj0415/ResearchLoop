"""Parallel isolated sub-agents. The parent joins before its next model call.

Disjoint token allocations bound aggregate spend; actual calls are charged once
to the parent at each join. Each worker owns its state, trace and workspace.
"""
import copy
import shutil
import threading
from researchloop import storage
from researchloop.ollama import BudgetError
from . import registry, routing, history

TASK_SCHEMA = {'type':'object','properties':{
    'goal':{'type':'string'},'role':{'type':'string','enum':['planner','analyst','generalist','experimenter','verifier']},
    'model_tier':{'type':'string','enum':['auto','light','strong']},'max_steps':{'type':'integer'}},
    'required':['goal','role','model_tier'],'additionalProperties':False}
ARGUMENTS = {
    'delegate_research': ({'tasks':{'type':'array','minItems':1,'maxItems':3,'items':TASK_SCHEMA},
                           'batch_key':{'type':'string'}}, ['tasks']),
    'read_subagents': ({'subagent_id':{'type':'string'},'path':{'type':'string'},'max_chars':{'type':'integer'}}, []),
    'select_model': ({'model_tier':{'type':'string','enum':['auto','light','strong']},'reason':{'type':'string'}}, ['model_tier']),
}


def select_model(engine, args):
    tier = args['model_tier']
    conf=routing.profile(engine, tier)  # Validate availability before changing state.
    if engine.run.get('active_agent'):
        engine.run['active_agent']['model_tier'] = tier
        if conf:
            engine.run['active_agent'].update(model=conf['model'],model_source=conf.get('model_source','local'))
        else:
            model,source=engine.resolve_model(engine.run['active_agent']['role'])
            engine.run['active_agent'].update(model=model,model_source=source)
    else:
        engine.run['model_tier'] = tier
    return {'model_tier':tier,'reason':args.get('reason') or 'Agent 显式选择模型档位','applies_to':'后续模型请求；本操作不调用模型'}


def _folder(engine, identifier):
    if not identifier.startswith('sa') or not identifier[2:].isdigit():
        raise ValueError('子 Agent ID 无效')
    return engine.folder/'subagents'/identifier


def read_subagents(engine, args):
    if args.get('path'):
        if not args.get('subagent_id') or not any(i['id']==args['subagent_id'] for i in engine.run.get('subagents',[])):
            raise ValueError('读取子任务文件时须指定本会话已有 subagent_id')
        from types import SimpleNamespace
        from .engineering import read_file
        context=SimpleNamespace(folder=_folder(engine,args['subagent_id']))
        return {'subagent_id':args['subagent_id'],**read_file(context,args)}
    rows=[]
    for item in engine.run.get('subagents',[]):
        if args.get('subagent_id') and item['id']!=args['subagent_id']:
            continue
        saved = storage.read_json(_folder(engine,item['id'])/'run.json') or {}
        rows.append({'id':item['id'],'goal':item['goal'],'model_tier':item['model_tier'],'turn':item.get('turn'),
                     'status':saved.get('status',item.get('status')),'report':saved.get('report'),
                     'evidence_map':item.get('evidence_map'), 'budget':saved.get('budget'),
                     'recent_steps':[{key:step.get(key) for key in ('tool','status','observation')}
                                     for step in saved.get('steps',[])[-3:]]})
    return {'subagents':rows,'note':'子 Agent 的结论仍需核查其原始来源和实际实验；不是独立真实证据的替代品。'}


def _make_child(parent, item, task, cap, seconds):
    from .core import ResearchEngine
    folder = _folder(parent,item['id'])
    folder.mkdir(parents=True,exist_ok=True)
    for base in ('inputs','source','experiments','documents','evidence'):
        if (parent.folder/base).exists():
            shutil.copytree(parent.folder/base,folder/base,dirs_exist_ok=True)
    # Workspace edits stay isolated; no shared mutable project files.
    if (parent.folder/'workspace').exists():
        shutil.copytree(parent.folder/'workspace',folder/'workspace',
                        ignore=shutil.ignore_patterns('.git','.venv','node_modules','.cache'),dirs_exist_ok=True)
    run = copy.deepcopy(parent.run)
    run.update(id=parent.run['id']+'-'+item['id'],session_id=parent.run['id'],subagent_id=item['id'],
               is_subagent=True,parent_run=parent.run['id'],question=task['goal'],current_question=task['goal'],
               branch_note='主研究问题：'+parent.question_text(),workflow_mode='react',workflow=None,
               token_limit=cap,seconds_limit=seconds,max_steps=task.get('max_steps',3),
               budget_enabled=cap is not None or seconds is not None,memory_mode='disabled',
               model_tier=task['model_tier'],agents=[],steps=[],calls=[],trace=[],events=[],reflections=[],
               subagents=[],delegation_batches=[],approval_requests=[],permission_grants=[],
               status='queued',created_at=storage.now(),updated_at=storage.now(),finished_at=None,
               budget={},turns=[{'id':'t1','question':task['goal'],'status':'queued','at':storage.now()}])
    for key in ('report','pending_response','pending_decision','pending_tool_retry','restore_budget','budget_call_start',
                'branch_step_start','gap_fill','active_agent','recovery_feedback','decision_feedback','wrap_up',
                'draft_report','stop_reason','error','memory_saved_version','lineage'):
        run.pop(key,None)
    role = task['role']
    model, source = parent.resolve_model(role)
    run['active_agent']={'id':item['id'],'role':role,'goal':task['goal'],'model':model,'model_source':source,
                         'model_tier':task['model_tier']}
    item['inherited']={name:len(run.get(name,[])) for name in ('sources','hypotheses','experiments','documents')}
    child = ResearchEngine(run,folder)
    child.save()
    return child


def _merge(parent, item, child):
    mapping = item.setdefault('evidence_map',{})
    inherited = item['inherited']
    for group,prefix in [('sources','p'),('hypotheses','h'),('experiments','x')]:
        for index,row in enumerate(child.run.get(group,[])):
            if index < inherited[group]:
                # Independent verification of inherited frozen code can be imported.
                if group=='experiments' and row.get('verification'):
                    existing = next((r for r in parent.run[group] if r['id']==row['id']),None)
                    if existing and existing.get('code')==row.get('code') and not existing.get('verification'):
                        existing['verification']=copy.deepcopy(row['verification'])
                continue
            source_key=group+':'+row['id']
            if source_key in mapping:
                continue
            new = copy.deepcopy(row)
            new['id'] = prefix+str(len(parent.run[group])+1)
            new['subagent_origin']={'id':item['id'],'original_id':row['id']}
            if group=='experiments':
                new['hypothesis_id']=mapping.get('hypotheses:'+row.get('hypothesis_id',''),row.get('hypothesis_id'))
                source=child.folder/'experiments'/row['id']
                if source.exists():
                    shutil.copytree(source,parent.folder/'experiments'/new['id'],dirs_exist_ok=False)
            mapping[source_key]=new['id']
            parent.run[group].append(new)
    if child.run.get('sources') or child.run.get('experiments'):
        parent.run['evidence_collected']=True
    start=item.get('accounted_calls',0)
    for original in child.run.get('calls',[])[start:]:
        call=copy.deepcopy(original)
        call.update(subagent_id=item['id'],turn=parent.turn_id)
        parent.run['calls'].append(call)
        parent.budget.record(call)
    item['accounted_calls']=len(child.run.get('calls',[]))


def delegate_research(engine, args):
    from .core import resume_paused
    from .protocol import validate
    if engine.run.get('is_subagent'):
        raise ValueError('子 Agent 不能再次委派，避免无界递归与预算失控')
    tasks=args.get('tasks')
    validate(tasks, ARGUMENTS['delegate_research'][0]['tasks'],'tasks')
    for task in tasks:
        if not 8<=len(task['goal'])<=1200 or not 1<=task.get('max_steps',3)<=12:
            raise ValueError('子任务目标应为 8–1200 字符，每项 1–12 步')
        routing.profile(engine,task['model_tier'])
    key=args.get('batch_key') or 'batch-'+str(len(engine.run.get('delegation_batches',[]))+1)
    batch=next((b for b in engine.run.get('delegation_batches',[]) if b['key']==key and b['turn']==engine.turn_id),None)
    if batch and batch['status']=='done':
        return read_subagents(engine,{})
    items=[]
    if not batch:
        room=None if engine.budget.token_limit is None else engine.budget.token_limit-engine.budget.used-8000
        cap=None if room is None else room//len(tasks)
        if cap is not None and cap<7644:
            raise BudgetError('剩余预算不足以并行委派并保留主 Agent 收尾额度；请减少任务或增加预算')
        seconds=engine.budget.seconds_left()
        if seconds is not None and seconds<=5:
            raise BudgetError('剩余时间不足以启动子任务')
        batch={'key':key,'turn':engine.turn_id,'status':'running','ids':[],'tasks':copy.deepcopy(tasks)}
        engine.run.setdefault('delegation_batches',[]).append(batch)
        for task in tasks:
            item={'id':'sa'+str(len(engine.run.setdefault('subagents',[]))+1),'goal':task['goal'],
                  'role':task['role'],'model_tier':task['model_tier'],'status':'queued','turn':engine.turn_id}
            engine.run['subagents'].append(item)
            batch['ids'].append(item['id'])
            child=_make_child(engine,item,task,cap,seconds)
            items.append((item,child))
    else:
        if tasks!=batch['tasks']:
            raise ValueError('不能用相同 batch_key 改写已经启动的子任务')
        for identifier in batch['ids']:
            item=next(i for i in engine.run['subagents'] if i['id']==identifier)
            saved=storage.read_json(_folder(engine,identifier)/'run.json')
            if saved.get('status')=='paused':
                items.append((item,resume_paused(saved,_folder(engine,identifier))))
    engine.run['delegation_reserved']=sum((child.budget.token_limit or 0)-child.budget.used for _,child in items)
    threads=[]
    nodes=[]
    for item,child in items:
        child.budget.suspend()
        child.dependency_paths=[engine.folder/'workspace/.packages']
        child.permission_gate=engine.permission_gate
        child.pause_requested=engine.pause_requested
        child.cancelled=engine.cancelled
        factory=getattr(engine,'child_decision_factory',None)
        if factory:
            child.decide=factory(child,item)
        node=history.begin(engine,'subagent',subagent_id=item['id'],model_tier=item['model_tier'],
                           role=item['role'],agent=item['id'],reason=item['goal'],source='controller',
                           input={'goal':item['goal'],'model_tier':item['model_tier'],'token_limit':child.budget.token_limit})
        nodes.append((item,child,node,child.budget.snapshot()))
        def work(current=child):
            try:
                current.budget.resume_clock()
                current.execute()
            except Exception as error:
                current.run.update(status='failed',error=str(error),finished_at=storage.now())
                current.save()
            finally:
                current.budget.suspend()
        thread=threading.Thread(target=work,daemon=True,name=engine.run['id']+'-'+item['id'])
        threads.append(thread)
    # Start only after all snapshots and allocations are prepared.
    started=[]
    for index,thread in enumerate(threads):
        try:
            thread.start()
            started.append(thread)
        except Exception as error:
            engine.cancelled.set()
            engine.run['stop_reason']='子 Agent 线程无法启动，保留已执行调用的实际用量'
            for _,current,_,_ in nodes[index:]:
                current.run.update(status='failed',error=str(error),finished_at=storage.now())
                current.save()
            break
    for thread in started:
        thread.join()
    for item,child,node,before in nodes:
        _merge(engine,item,child)
        item.update(status=child.run['status'],updated_at=child.run.get('updated_at'),budget=child.budget.snapshot())
        after=child.budget.snapshot()
        delta={key:after.get(key,0)-before.get(key,0) for key in ('input_tokens','output_tokens','unknown_reserved','cached_tokens','total_tokens','model_calls','cloud_calls')}
        history.settle(engine,node,status='paused' if child.run['status']=='paused' else 'failed' if child.run['status']=='failed' else 'completed',
                       output={'status':child.run['status'],'report':child.run.get('report'),'evidence_map':item['evidence_map']},
                       usage=delta,elapsed_seconds=max(0,child.budget.elapsed-before.get('elapsed_seconds',0)))
    engine.run['delegation_reserved']=0
    batch['status']='paused' if any(i['status']=='paused' for i,_ in items) else 'done'
    if batch['status']=='paused':
        engine.run['pending_decision']={'tool':'delegate_research','reason':'继续被暂停的子任务',
                                        'args':{'tasks':batch['tasks'],'batch_key':key}}
    return read_subagents(engine,{})


registry.register_tool('delegate_research',delegate_research,
    description='并行委派 1–3 个独立子 Agent，共享主研究总预算的分配额度。每项可选择设置中的轻量 light 或高能力 strong 档。等待本批结果后继续主决策；子任务不能递归委派。',
    arguments={'tasks':[{'goal':'独立检索支持与反对证据','role':'analyst','model_tier':'light','max_steps':3}]},required=['tasks'])
registry.register_tool('read_subagents',read_subagents,description='读取已委派子 Agent 的状态、证据映射、报告和最近调用；可同时指定 subagent_id、path 读取该子任务工作目录内的文件。',arguments={'subagent_id':'sa1','path':'experiment.py'},required=[])
registry.register_tool('select_model',select_model,description='为当前阶段后续模型调用选择设置中的 light/strong，或 auto 规则路由；本工具本身不调用模型。',
                       arguments={'model_tier':'light','reason':'常规取证使用轻量模型'},required=['model_tier'])
