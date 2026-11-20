"""Measured single-pixel waveform classification; deterministic group-disjoint MLP."""
from pathlib import Path
import os,re,json,time,random,hashlib,shutil,importlib.util
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,classification_report

BASE=Path(r'D:\石阳\博士工作\科研\周报')
DATA=BASE/'week update-65-shiyang/gamry/20250313/night-1st test/全部条件切分周期-auto-two stages'
ORIGINAL=BASE/'week update-87-shiyang/神经网络'
OUT=BASE/'week update-140-shiyang/单像素三任务_轻量MLP_光强方向频率'
SEED=42;EPOCHS=100;BATCH=256
TASKS=['intensity','direction','frequency']

class LightMLP(nn.Module):
    def __init__(self,classes):
        super().__init__()
        self.body=nn.Sequential(nn.Linear(102,64),nn.ReLU(),nn.Dropout(.1),nn.Linear(64,32),nn.ReLU())
        self.heads=nn.ModuleList([nn.Linear(32,n) for n in classes])
    def forward(self,x):
        h=self.body(x);return [head(h) for head in self.heads]

def dump(p,o):p.write_text(json.dumps(o,ensure_ascii=False,indent=2),encoding='utf-8')
def load_preprocessor():
    p=ORIGINAL/'predict_utils.py';spec=importlib.util.spec_from_file_location('original_preprocess',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def prepare():
    target=OUT/'01_数据与划分';target.mkdir(parents=True,exist_ok=True)
    if (target/'inputs.npz').exists():
        z=np.load(target/'inputs.npz');return z['features'],z['y'],pd.read_csv(target/'split_manifest.csv'),json.loads((target/'labels.json').read_text(encoding='utf-8'))
    prep=load_preprocessor();jobs=[]
    for folder in sorted(DATA.iterdir()):
        m=re.fullmatch(r'peaks_auto_(\d+)%_(\d+)_(\d+)Hz',folder.name)
        if m:
            values=tuple(map(int,m.groups()))
            for p in sorted(folder.glob('*.csv')):jobs.append((p,values))
    print('PREPROCESS',len(jobs),'files',flush=True)
    def one(job):
        p,values=job;rawbytes=p.read_bytes();df=pd.read_csv(p)
        a=df.Current_A.to_numpy(dtype=np.float64)
        if len(a)<20 or not np.isfinite(a).all():raise ValueError(f'Invalid input {p}')
        feat=prep.extract_features(prep.wavelet_denoise(a),100)
        assert feat.shape==(102,) and np.isfinite(feat).all()
        source=str(df.File.iloc[0]) if 'File' in df else p.parent.name
        num=int(re.search(r'(\d+)$',p.stem).group(1))
        return feat,dict(path=str(p),condition=p.parent.name,intensity=values[0],direction=values[1],frequency=values[2],source_file=source,cycle_index=num,pair_index=num//2,sample_count=len(a),file_sha256=hashlib.sha256(rawbytes).hexdigest(),waveform_sha256=hashlib.sha256(a.tobytes()).hexdigest())
    rows=[];features=[];start=time.perf_counter()
    with ThreadPoolExecutor(max_workers=8) as pool:
        for k,(feat,row) in enumerate(pool.map(one,jobs),1):
            features.append(feat);rows.append(row)
            if k%3000==0:print('PREPROCESS',k,'/',len(jobs),'seconds',round(time.perf_counter()-start),flush=True)
    meta=pd.DataFrame(rows);x=np.stack(features)
    # Union adjacent edge pairs and exact repeated raw waveforms before assigning splits.
    parent=np.arange(len(meta))
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(a,b):parent[root(b)]=root(a)
    seen_pair={};seen_wave={}
    for i,r in enumerate(rows):
        key=(r['condition'],r['source_file'],r['pair_index'])
        if key in seen_pair:union(i,seen_pair[key])
        else:seen_pair[key]=i
        h=r['waveform_sha256']
        if h in seen_wave:union(i,seen_wave[h])
        else:seen_wave[h]=i
    meta['group']=[root(i) for i in range(len(meta))]
    # Each connected group must belong to one condition for per-condition stratification.
    assert meta.groupby('group').condition.nunique().max()==1,'Identical waveforms with conflicting conditions; inspect before training'
    rng=np.random.default_rng(SEED);meta['split']=''
    for condition,part in meta.groupby('condition',sort=True):
        groups=np.sort(part.group.unique());rng.shuffle(groups);n=len(groups)
        assert n>=5,(condition,n)
        nt=max(1,int(n*.2));nv=max(1,int(n*.1))
        for split,ids in [('test',groups[:nt]),('val',groups[nt:nt+nv]),('train',groups[nt+nv:])]:meta.loc[part.index[part.group.isin(ids)],'split']=split
    assert meta.groupby('group').split.nunique().max()==1
    assert meta.groupby('waveform_sha256').split.nunique().max()==1
    assert meta.split.ne('').all()
    labels={t:sorted(map(int,meta[t].unique())) for t in TASKS}
    y=np.column_stack([[labels[t].index(int(v)) for v in meta[t]] for t in TASKS]).astype(np.int64)
    np.savez_compressed(target/'inputs.npz',features=x,y=y)
    meta.to_csv(target/'split_manifest.csv',index=False);dump(target/'labels.json',labels)
    meta.groupby(['split','intensity','direction','frequency']).size().rename('samples').to_csv(target/'class_counts.csv')
    dump(target/'split_audit.json',{'samples':len(meta),'groups':int(meta.group.nunique()),'conditions':int(meta.condition.nunique()),'split_counts':meta.split.value_counts().to_dict(),'same_group_across_splits':False,'identical_waveform_across_splits':False,'policy':'condition-stratified connected groups of same source adjacent cycle indices floor(index/2), merged with exact repeated raw waveforms; approximately70/10/20','generalization_limit':'same acquisition sessions; not independent-session generalization'})
    return x,y,meta,labels

@torch.inference_mode()
def evaluate(model,x,y,ids,weights):
    model.eval();loss=0;preds=[]
    for ix in ids.split(1024):
        out=model(x[ix]);truth=y[ix];l=sum(F.cross_entropy(o,truth[:,j],weight=weights[j]) for j,o in enumerate(out))
        loss+=float(l)*len(ix);preds.append(torch.stack([o.argmax(1) for o in out],1))
    pred=torch.cat(preds);truth=y[ids]
    return {'loss':loss/len(ids),**{f'{t}_accuracy':float((pred[:,j]==truth[:,j]).float().mean()) for j,t in enumerate(TASKS)},'joint_accuracy':float((pred==truth).all(1).float().mean())},pred.cpu().numpy()

def main():
    import sys;sys.stdout.reconfigure(encoding='utf-8')
    if (OUT/'完成检查.json').exists():raise RuntimeError('Completed run exists')
    for folder in ['01_数据与划分','02_训练过程','03_独立留出测试','code']:(OUT/folder).mkdir(parents=True,exist_ok=True)
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.set_num_threads(4)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    raw,y,meta,labels=prepare();train=np.flatnonzero(meta.split.eq('train'));val=np.flatnonzero(meta.split.eq('val'));test=np.flatnonzero(meta.split.eq('test'))
    # Preserve original waveform features; train-only feature standardization exposes amplitude statistics to MLP.
    mean=raw[train].mean(0,dtype=np.float64).astype(np.float32);std=raw[train].std(0,dtype=np.float64).astype(np.float32);std=np.maximum(std,1e-12)
    scaled=(raw-mean)/std;assert np.isfinite(scaled).all()
    np.savez(OUT/'01_数据与划分/train_only_standardization.npz',mean=mean,std=std)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');x=torch.tensor(scaled,device=device);target=torch.tensor(y,device=device)
    ids={s:torch.tensor(a,device=device) for s,a in [('train',train),('val',val),('test',test)]}
    classes=[len(labels[t]) for t in TASKS];model=LightMLP(classes).to(device);opt=torch.optim.Adam(model.parameters(),lr=.001)
    # Equal task weights; class balancing derived exclusively from training labels.
    weights=[]
    for j,n in enumerate(classes):
        counts=np.bincount(y[train,j],minlength=n);w=1/counts;w=w/w.mean();weights.append(torch.tensor(w,dtype=torch.float32,device=device))
    print('TRAIN',len(train),'VAL',len(val),'TEST',len(test),'LABELS',labels,'PARAMS',sum(p.numel() for p in model.parameters()),'DEVICE',device,flush=True)
    hist=[];best=-1e30;bestepoch=0;start=time.perf_counter()
    for epoch in range(1,EPOCHS+1):
        model.train();loss_sum=0;correct=np.zeros(3,dtype=np.int64)
        order=ids['train'][torch.randperm(len(train),device=device)]
        for ix in order.split(BATCH):
            out=model(x[ix]);truth=target[ix];loss=sum(F.cross_entropy(o,truth[:,j],weight=weights[j]) for j,o in enumerate(out))
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
            loss_sum+=float(loss.detach())*len(ix)
            correct+=np.array([int((o.argmax(1)==truth[:,j]).sum()) for j,o in enumerate(out)])
        vm,_=evaluate(model,x,target,ids['val'],weights)
        row={'epoch':epoch,'train_loss':loss_sum/len(train),**{f'train_{t}_accuracy':float(correct[j]/len(train)) for j,t in enumerate(TASKS)},**{'val_'+k:v for k,v in vm.items()}}
        hist.append(row)
        if -vm['loss']>best:
            best=-vm['loss'];bestepoch=epoch
            torch.save({'state_dict':model.state_dict(),'labels':labels,'feature_mean':mean,'feature_std':std,'epoch':epoch},OUT/'02_训练过程/best_model.pth')
        pd.DataFrame(hist).to_csv(OUT/'02_训练过程/training_history.csv',index=False)
        if epoch%10==0 or epoch==1:print('EPOCH',epoch,'loss',round(row['train_loss'],4),'VAL',vm,'seconds',round(time.perf_counter()-start),flush=True)
    torch.save(model.state_dict(),OUT/'02_训练过程/last_model.pth')
    checkpoint=torch.load(OUT/'02_训练过程/best_model.pth',map_location=device,weights_only=False);model.load_state_dict(checkpoint['state_dict'])
    tm,pred=evaluate(model,x,target,ids['test'],weights)
    result=meta.iloc[test].copy();truth=y[test]
    for j,t in enumerate(TASKS):
        result['pred_'+t]=np.array(labels[t])[pred[:,j]];result[t+'_correct']=pred[:,j]==truth[:,j]
        cm=confusion_matrix(truth[:,j],pred[:,j],labels=np.arange(classes[j]));norm=cm/np.maximum(cm.sum(1,keepdims=True),1)
        pd.DataFrame(cm,index=labels[t],columns=labels[t]).to_csv(OUT/f'03_独立留出测试/confusion_{t}_counts.csv')
        pd.DataFrame(norm,index=labels[t],columns=labels[t]).to_csv(OUT/f'03_独立留出测试/confusion_{t}_normalized.csv')
        dump(OUT/f'03_独立留出测试/classification_{t}.json',classification_report(truth[:,j],pred[:,j],labels=np.arange(classes[j]),target_names=list(map(str,labels[t])),output_dict=True,zero_division=0))
        fig,ax=plt.subplots(figsize=(5,4));im=ax.imshow(norm,cmap='Blues',vmin=0,vmax=1);ax.set(xticks=np.arange(classes[j]),yticks=np.arange(classes[j]),xticklabels=labels[t],yticklabels=labels[t],xlabel='Predicted class',ylabel='True class',title=t)
        for r in range(classes[j]):
            for c in range(classes[j]):ax.text(c,r,f'{norm[r,c]:.2f}',ha='center',va='center',fontsize=8,color='white' if norm[r,c]>.5 else 'black')
        fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(OUT/f'03_独立留出测试/confusion_{t}.png',dpi=600);plt.close(fig)
    result.to_csv(OUT/'03_独立留出测试/predictions.csv',index=False)
    dump(OUT/'03_独立留出测试/test_summary.json',dict(samples=len(test),selected_epoch=bestepoch,**tm))
    result.groupby(['intensity','direction','frequency'])[[t+'_correct' for t in TASKS]].mean().to_csv(OUT/'03_独立留出测试/accuracy_by_condition.csv')
    h=pd.DataFrame(hist)
    # Requested four-curve overview: stochastic training loss and three training accuracies.
    fig,ax=plt.subplots(figsize=(7,5));right=ax.twinx();lines=ax.plot(h.epoch,h.train_loss,color='#df3027',label='Loss')
    for t,color in zip(TASKS,['#159d76','#d85f00','#7d70b2']):lines+=right.plot(h.epoch,h['train_'+t+'_accuracy'],color=color,label=f'Accuracy ({t})')
    ax.set(xlabel='Epoch',ylabel='Training loss');right.set(ylabel='Training accuracy',ylim=(0,1.04));ax.legend(lines,[l.get_label() for l in lines],loc='center right',fontsize=9);fig.tight_layout();fig.savefig(OUT/'02_训练过程/training_curves.png',dpi=600);fig.savefig(OUT/'02_训练过程/training_curves.pdf');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,7));axes[0,0].plot(h.epoch,h.train_loss,label='train');axes[0,0].plot(h.epoch,h.val_loss,label='validation');axes[0,0].set_title('Loss')
    for ax,t in zip(axes.ravel()[1:],TASKS):
        ax.plot(h.epoch,h['train_'+t+'_accuracy'],label='train');ax.plot(h.epoch,h['val_'+t+'_accuracy'],label='validation');ax.set_title(t);ax.set_ylim(0,1.02)
    for ax in axes.ravel():ax.set_xlabel('Epoch');ax.legend()
    fig.tight_layout();fig.savefig(OUT/'02_训练过程/train_validation_curves.png',dpi=600);plt.close(fig)
    config={'model':'MLP102-64-32 with 3/7/5 classification heads','class_labels':labels,'parameters':sum(p.numel() for p in model.parameters()),'linear_MACs_per_sample':102*64+64*32+32*sum(classes),'linear_FLOPs_per_sample_2_per_MAC':2*(102*64+64*32+32*sum(classes)),'input_features':'original db5 denoising,100 resampled normalized signed waveform points plus pre-normalization mean/std; then train-only feature standardization','dropout':.1,'activation':'ReLU','epochs':EPOCHS,'batch_size':BATCH,'optimizer':'Adam','lr':.001,'loss':'sum of three training-count inverse-frequency weighted cross-entropies, no sampler or manual class boosts','selection':'minimum validation loss, test evaluated once after selection','best_epoch':bestepoch,'seed':SEED,'device':str(device),'gpu':torch.cuda.get_device_name(0) if device.type=='cuda' else None,'training_seconds':time.perf_counter()-start,'frequency_is_speed_proxy':True,'frequency_to_speed_conversion_supplied':False,'array_simulation_used':False,'original_split_reproduced':False,'split_reason':'original train/test independently shuffled; prevent overlap with source-cycle-pair grouped 70/10/20 split','same_session_limit':True}
    dump(OUT/'实验设置.json',config)
    shutil.copy2(__file__,OUT/'code'/Path(__file__).name)
    shutil.copy2(ORIGINAL/'predict_utils.py',OUT/'code/original_predict_utils_reference.py')
    shutil.copy2(ORIGINAL/'training_code_v4.py',OUT/'code/original_training_reference_v4.py')
    (OUT/'说明.md').write_text('轻量MLP重新完成原单像素光强、方向、频率三任务。输入来自65周相同two stages实测数据，小波去噪与100点波形+均值标准差特征沿用87周实现；随后按训练集统计标准化。102→64→32共享主体，ReLU，第一层后Dropout0.1，三个分类头。不是140周仅200Hz的两任务阵列MLP。\n\n原训练与测试分别洗牌会有重叠风险，本次按条件分层，将同来源相邻cycle索引floor(index/2)成组，并合并完全相同的原始波形后，固定约70%/10%/20%训练/验证/测试。同组与重复波形不跨集合。仍属于同采集批次内留出，不证明跨批次泛化。和旧曲线的数值不构成严格等划分对照。\n\n100轮，Adam lr0.001，batch256，三任务交叉熵相加，权重只依据训练集类别数，无手工频率加权和重采样。按验证损失选模型，只在选定后评估测试集。训练图是训练损失和训练准确率，测试结果另存；频率是原任务中的速度代理标签，未擅自换算m/s。计算量仅全连接乘加，未包含预处理。\n\n01保存预处理特征缓存、标签、源文件哈希和每个样本的固定划分；02保存权重、100轮CSV和曲线；03保存测试预测、混淆矩阵及分类报告；code保存实际代码及原实现参考。复现训练需原始源目录，已缓存features可直接用于同划分训练。\n',encoding='utf-8')
    dump(OUT/'完成检查.json',{'complete':True,'epochs':len(hist),'test_samples':len(test),'no_group_or_duplicate_waveform_overlap':True,'test_not_used_for_model_selection':True,'test_metrics':tm})
    print('COMPLETE',OUT,'TEST',json.dumps(tm),'BEST_EPOCH',bestepoch,flush=True)
if __name__=='__main__':main()
