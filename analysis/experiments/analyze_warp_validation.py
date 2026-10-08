"""Terminal-only paired analysis; apply the archived rule without reselection."""
import sys,json,hashlib,datetime
from pathlib import Path
import numpy as np,pandas as pd
repo=Path(__file__).resolve().parents[2]
local=Path(r'G:\Articoli\Articoli da Completare\LoSTer 2026')
out=local/'warp-validation';execution=out/'execution'
sys.dont_write_bytecode=True

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    ledger=json.loads((execution/'ledger.json').read_text())
    assert len(ledger)==40,'Do not analyze partial quality scores'
    if not all(r['status']=='COMPLETE' for r in ledger):raise RuntimeError('Failed validation fits remain explicit; no adoption')
    preserve=json.loads((execution/'pilot-preservation.json').read_text());assert preserve['all_bytes_and_mtimes_unchanged']
    rule=json.loads((execution/'warp-validation-protocol.json').read_text())
    frozen=json.loads((execution/'frozen.json').read_text())
    refs=json.loads((out/'data-only/w0-references.json').read_text());records=[]
    for ref in refs:
        name=ref['dataset'];seed=ref['seed'];key=Path(name)/'LoSTer-Legacy-Clean'/('seed-%d'%seed)
        oldmp=local/'pilot/artifacts'/key/'manifest.json';old=json.loads(oldmp.read_text())
        assert old['git']['sha']=='a247dffdea50e236b171392b8c9336ecd4cc5727' and not old['git']['dirty']
        assert digest(oldmp)==ref['manifest_sha256']
        assert digest(oldmp.parent/'complete.pt')==ref['checkpoint_sha256']
        assert digest(local/'pilot/results'/key/'metrics.json')==ref['result_sha256']
        newmp=out/'artifacts'/key/'manifest.json';new=json.loads(newmp.read_text())
        oldr=json.loads((local/'pilot/results'/key/'metrics.json').read_text());newr=json.loads((out/'results'/key/'metrics.json').read_text())
        validation=json.loads((out/'runs'/key/'validation.json').read_text())
        assert validation['original_preparation_exact'] and validation['rng_boundary_exact'] and validation['data_only_snapshots_exact'] and validation['checkpoint_round_trip_exact']
        assert digest(newmp.parent/'complete.pt')==validation['checkpoint_sha256']
        assert new['environment']==old['environment']==frozen['environment'] and new['dataset']==old['dataset']
        expected=dict(old['config']);expected.update(warp_mode='MonotoneWarp',augmentation_order=['sign','equal-segment-permutation','monotone-time-warp'])
        assert expected==new['config'],'Unexpected scientific config difference'
        oldepochs=[json.loads(l) for l in (local/'pilot/logs'/key/'epochs.jsonl').read_text().splitlines()]
        newepochs=[json.loads(l) for l in (out/'logs'/key/'epochs.jsonl').read_text().splitlines()]
        r=dict(dataset=name,seed=seed,W0_epoch=oldr['final_epoch'],W1_epoch=newr['final_epoch'],W0_cost=oldr['fit_seconds'],W1_cost=newr['fit_seconds'],W0_joint_seconds=old['timing']['joint_operational_seconds'],W1_joint_seconds=new['timing']['joint_operational_seconds'],W0_stop=old['stop_reason'],W1_stop=new['stop_reason'])
        for metric in ['ARI','NMI_arithmetic','RI','ACC']:
            r['W0_'+metric]=oldr['metrics'][metric];r['W1_'+metric]=newr['metrics'][metric];r['delta_'+metric]=r['W1_'+metric]-r['W0_'+metric]
        for arm,result,epochs in [('W0',oldr,oldepochs),('W1',newr,newepochs)]:
            r[arm+'_collapse']=bool(result['collapse_original'] or result['collapse_augmented'])
            r[arm+'_near_collapse']=bool(epochs[-1]['original']['near_collapse'] or epochs[-1]['augmented_A_train']['near_collapse'])
            r[arm+'_any_epoch_collapse']=any(e['original']['complete_collapse'] or e['augmented_A_train']['complete_collapse'] for e in epochs)
            r[arm+'_occupied_original']=epochs[-1]['original']['occupied_clusters'] if 'occupied_clusters' in epochs[-1]['original'] else sum(n>0 for n in epochs[-1]['original']['counts'])
            r[arm+'_occupied_augmented']=sum(n>0 for n in epochs[-1]['augmented_A_train']['counts'])
            r[arm+'_final_assignment_change_original']=epochs[-1]['original']['assignment_change_fraction']
            r[arm+'_final_assignment_change_augmented']=epochs[-1]['augmented_A_train']['assignment_change_fraction']
            r[arm+'_final_hard_entropy_original']=epochs[-1]['original']['hard_entropy']
            r[arm+'_final_hard_entropy_augmented']=epochs[-1]['augmented_A_train']['hard_entropy']
            for component,value in epochs[-1]['losses'].items():r[arm+'_final_loss_'+component]=value
            r[arm+'_final_reconstruction']=epochs[-1]['losses']['reconstruction'] if 'reconstruction' in epochs[-1]['losses'] else None
        r['new_collapse']=r['W1_collapse'] and not r['W0_collapse'];r['new_near_collapse']=r['W1_near_collapse'] and not r['W0_near_collapse']
        records.append(r)
    d=pd.DataFrame(records);summaries=[]
    for name,g in d.groupby('dataset',sort=False):
        r=dict(dataset=name,seeds=len(g),W0_ARI_mean=g.W0_ARI.mean(),W1_ARI_mean=g.W1_ARI.mean(),paired_ARI_mean=g.delta_ARI.mean(),paired_ARI_sd=g.delta_ARI.std(ddof=1),W0_ARI_sd=g.W0_ARI.std(ddof=1),W1_ARI_sd=g.W1_ARI.std(ddof=1),W0_NMI_mean=g.W0_NMI_arithmetic.mean(),W1_NMI_mean=g.W1_NMI_arithmetic.mean(),paired_NMI_mean=g.delta_NMI_arithmetic.mean(),negative_ARI_seeds=int((g.delta_ARI<0).sum()),W0_epochs_mean=g.W0_epoch.mean(),W1_epochs_mean=g.W1_epoch.mean(),W0_cost_mean=g.W0_cost.mean(),W1_cost_mean=g.W1_cost.mean(),W0_occupied_original_mean=g.W0_occupied_original.mean(),W1_occupied_original_mean=g.W1_occupied_original.mean(),W0_occupied_augmented_mean=g.W0_occupied_augmented.mean(),W1_occupied_augmented_mean=g.W1_occupied_augmented.mean())
        r['ARI_sd_increase']=r['W1_ARI_sd']-r['W0_ARI_sd'];summaries.append(r)
    s=pd.DataFrame(summaries);a=rule['adoption']
    gates=dict(complete_40=len(d)==40,valid_all_paths=json.loads((out/'data-only-summary.json').read_text())['monotone_invalid']==0,new_collapse=int(d.new_collapse.sum())==0,new_near_collapse=int(d.new_near_collapse.sum())==0,dataset_ARI_band=bool(s.paired_ARI_mean.min()>=a['minimum_dataset_mean_delta_ARI']),macro_ARI_band=bool(s.paired_ARI_mean.mean()>=a['minimum_macro_delta_ARI']),macro_NMI_band=bool(s.paired_NMI_mean.mean()>=a['minimum_macro_delta_NMI_arithmetic']),stability_band=bool(s.ARI_sd_increase.mean()<=a['maximum_mean_dataset_ARI_SD_increase']))
    catastrophic_datasets=s[(s.paired_ARI_mean<rule['catastrophic']['dataset']['mean_delta_ARI_below'])&(s.negative_ARI_seeds>=rule['catastrophic']['dataset']['minimum_negative_seeds'])].dataset.tolist()
    systematic=bool(s.paired_ARI_mean.mean()<-.01 and (s.paired_ARI_mean<0).sum()>=6 and s.paired_NMI_mean.mean()<-.01)
    decision='MONOTONEWARP ADOPTED' if all(gates.values()) else rule['failure_decision']
    result=dict(decision=decision,gates=gates,catastrophic_datasets=catastrophic_datasets,systematic_catastrophic=systematic,macro_delta_ARI=float(s.paired_ARI_mean.mean()),macro_delta_NMI_arithmetic=float(s.paired_NMI_mean.mean()),mean_ARI_SD_increase=float(s.ARI_sd_increase.mean()),positive_datasets=int((s.paired_ARI_mean>0).sum()),negative_datasets=int((s.paired_ARI_mean<0).sum()),W1_collapse=int(d.W1_collapse.sum()),W1_near_collapse=int(d.W1_near_collapse.sum()),W1_any_epoch_collapse=int(d.W1_any_epoch_collapse.sum()),pilot_preservation=preserve)
    dest=repo/'analysis/experiments/warp';dest.mkdir(exist_ok=True)
    d.to_csv(dest/'paired-runs.csv',index=False);s.to_csv(dest/'paired-dataset-effects.csv',index=False)
    (dest/'decision.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    text=(repo/'analysis/experiments/07-warp-validity-resolution.md').read_text()
    text=text.replace('Data-only stages complete; W1 training results pending.','Data-only and all 40 W1 fits complete; terminal paired analysis applied.')
    text=text.replace('Adoption is pending the complete 40 W1 fits and the rule below.','Final outcome: **'+decision+'**. The fixed rule below determines adoption; no alternate method search was opened.')
    text=text.replace('Pending. Analyze only after all 40 terminal outcomes.','**40/40 complete, zero failures.** All forty runs passed exact checkpoint reload, bitwise original-view preparation, RNG boundary pairing and exact data-only snapshot agreement. Forty W0 runs were reused; zero W0 retrains.\n\n| Dataset | W0 ARI | W1 ARI | Paired delta ARI | Paired delta NMI | ARI SD increase | Negative seeds | W0/W1 epochs |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'+''.join('| %s | %.4f | %.4f | %+.4f | %+.4f | %+.4f | %d/5 | %.1f / %.1f |\n'%(r['dataset'],r['W0_ARI_mean'],r['W1_ARI_mean'],r['paired_ARI_mean'],r['paired_NMI_mean'],r['ARI_sd_increase'],r['negative_ARI_seeds'],r['W0_epochs_mean'],r['W1_epochs_mean']) for r in summaries)+'\nMacro paired delta ARI **%+.6f**; arithmetic NMI **%+.6f**. Positive/negative dataset means: %d/%d. Mean ARI SD increase **%+.6f**. W1 final complete collapse %d/40, near-collapse %d/40; any-epoch complete collapse %d/40. Estimated standalone full-fit cost means W0 %.1fs, W1 %.1fs; median costs %.1fs/%.1fs. These are instrumented stage-summed estimates, not benchmark-quality repeated timing. See lightweight [paired runs](warp/paired-runs.csv) and [dataset effects](warp/paired-dataset-effects.csv).\n\n'%(result['macro_delta_ARI'],result['macro_delta_NMI_arithmetic'],result['positive_datasets'],result['negative_datasets'],result['mean_ARI_SD_increase'],result['W1_collapse'],result['W1_near_collapse'],result['W1_any_epoch_collapse'],d.W0_cost.mean(),d.W1_cost.mean(),d.W0_cost.median(),d.W1_cost.median())+'Analyze only after all 40 terminal outcomes.')
    text=text.replace('Pending complete validation; no method freeze yet.','**'+decision+'**.\n\n'+''.join('- '+k+': '+('PASS' if v else 'FAIL')+'\n' for k,v in gates.items())+'\nCatastrophic datasets: '+(', '.join(catastrophic_datasets) or 'none')+'. Systematic catastrophic flag: '+str(systematic)+'. No threshold, seed, arm or hyperparameter was changed after looking at W1 scores.\n')
    if decision=='MONOTONEWARP ADOPTED':
        spec=dict(schema=1,final_method_name='LoSTer',historical_reference='LoSTer-Legacy-Clean',final_2026_definition='LoSTer-Legacy-Clean with MonotoneWarp',warp_mode='MonotoneWarp',warp_definition='positive-cubic-log-speed-trapezoid-v1',sigma=.2,knot=4,augmentation_snapshots=2,augmentation_order=['sign','equal-segment-permutation','monotone-time-warp'],base_config=json.loads((out/'artifacts/SyntheticControl/LoSTer-Legacy-Clean/seed-0/manifest.json').read_text())['config'],parent_git_sha=frozen['git_sha'],source_sha256=frozen['source_sha256'],source_archive=str(execution/'source'),protocol_sha256=digest(execution/'warp-validation-protocol.json'),validation_decision_sha256=digest(dest/'decision.json'),NoResoftmax='DO NOT ADOPT',AlignedInit='DO NOT ADOPT',CORE44_launch_authorized=False,remaining_prerequisites=['Tier-1 baseline source/protocol gates','explicit CORE-44 campaign authorization'])
        for field in ('dataset','seed','device'):spec['base_config'].pop(field)
        spec['per_run_fields']=['dataset','seed','device']
        spec['final_benchmark_seeds']=[100,101,102,103,104]
        spec['source_committed']=False
        (repo/'analysis/experiments/frozen-method-2026.json').write_text(json.dumps(spec,indent=2)+'\n')
        text=text.replace('Freeze a lightweight machine-readable specification only after the rule passes.','The rule passed; [frozen method specification](frozen-method-2026.json) defines the final method, exact source and protocol. The warp-method scientific gate is resolved; separate baseline prerequisites and launch authorization remain.')
    else:text=text.replace('Freeze a lightweight machine-readable specification only after the rule passes.','No final 2026 method is frozen: the rule failed. Do not revert automatically to invalid HistoricalWarp or launch a third arm. A separately predeclared time-warp removal/reconsideration decision is required before CORE-44.')
    text+='\nTerminal preservation: **%d original pilot files unchanged in SHA256, size and mtime**, with identical file membership. New W1 source hash `%s`; exact dirty sources archived with parent commit `%s`. Training source did not change during the campaign.\n'%(preserve['files'],frozen['source_sha256'],frozen['git_sha'])
    (repo/'analysis/experiments/07-warp-validity-resolution.md').write_text(text)
    print(s[['dataset','paired_ARI_mean','paired_NMI_mean','ARI_sd_increase']].to_string(index=False));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
