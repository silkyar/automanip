"""Fixed admission for one adapted SAC seed-1 performance screen."""
SOURCE_SHA='1191b6bafb1c5a34cb2f661a23da732cf2021ec680f3971477d4c02fa8f614ca'
EXPERIMENT='stackcube-sac-screen-seed1-20261027'
SELECTED=(51_200,102_400,256_000)
CONFIG={'seed':1,'num_envs':32,'total_transitions':500_000,'buffer_size':500_000,
        'batch_size':1024,'learning_starts':4000,'training_freq':64,'utd':.5,
        'gamma':.9,'max_seconds':1320,'checkpoint_every':51_200,'device':'cuda'}


def command(python,source,output):
    args=[str(python),str(source)]
    for key,value in CONFIG.items():args.extend(['--'+key.replace('_','-'),str(value)])
    return args+['--output',str(output)]


def select_exports(report):
    actual=report['transitions'];exports=report['exports']
    if type(actual) is not int or actual<=0 or actual>500_000 or actual%32:raise ValueError('invalid SAC transition count')
    by_label={entry['path'].rstrip('/').split('/')[-1]:entry for entry in exports}
    if len(by_label)!=len(exports) or any(e['transitions']>actual for e in exports):raise ValueError('inconsistent exports')
    final=by_label.get('actor-final')
    if final is None or final['transitions']!=actual:raise ValueError('final export missing')
    selected=[]
    for steps in SELECTED:
        if steps>actual:continue
        entry=final if steps==actual else by_label.get('actor-step-'+str(steps))
        if entry is None or entry['transitions']!=steps:raise ValueError('prespecified export missing')
        selected.append({'steps':steps,'label':entry['path'].rstrip('/').split('/')[-1],
                         'weights_sha256':entry['weights_sha256'],'point':'prespecified-final-coincidence' if steps==actual else 'prespecified'})
    if not any(e['steps']==actual for e in selected):
        selected.append({'steps':actual,'label':'actor-final','weights_sha256':final['weights_sha256'],'point':'final'})
    return selected


def cost_bound():
    gpu_seconds=1380+30+4*(120+30)
    gpu_cpu_ram=gpu_seconds*(.000542+4*.0000131+16*.00000222)
    cpu=5400*(.0000131+2*.00000222)
    return {'gpu_container_seconds_bound':gpu_seconds,'gpu_cpu_ram_usd':gpu_cpu_ram,
            'coordinator_cpu_ram_usd':cpu,'subtotal_usd':gpu_cpu_ram+cpu,
            'cap_usd':1.5,'reserve_usd':1.5-gpu_cpu_ram-cpu}
