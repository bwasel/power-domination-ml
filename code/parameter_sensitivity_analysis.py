import math, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score

SEED_GEN=1
SEED_MODEL=2026
N=100
NUM_GRAPHS=100
FRACTIONS=[0.05,0.10,0.15]
STRATEGIES=['random','high-degree','distributed']
NUM_RANDOM=5
OUT=Path(__file__).resolve().parent / 'reviewer2_sensitivity_outputs'
OUT.mkdir(exist_ok=True)
rng=np.random.RandomState(SEED_GEN)

# ---------------- generation matching MATLAB logic ----------------
def generate_er(n, avg_degree):
    p=avg_degree/(n-1)
    for _ in range(1000):
        R=rng.rand(n,n)
        A=np.triu(R < p,1)
        A=A+A.T
        G=nx.from_numpy_array(A.astype(np.int8))
        if nx.is_connected(G): return G
    raise RuntimeError('ER connectivity failure')

def generate_ba(n, m):
    m0=m+1
    A=np.zeros((n,n),dtype=np.int8)
    for i in range(m0):
        for j in range(i+1,m0): A[i,j]=A[j,i]=1
    for new in range(m0,n):
        deg=A[:new,:new].sum(axis=1).astype(float)
        available=np.ones(new,dtype=bool)
        selected=[]
        for _ in range(m):
            w=deg.copy(); w[~available]=0
            probs=w/w.sum(); c=np.cumsum(probs); r=rng.rand(); chosen=int(np.searchsorted(c,r,side='left'))
            selected.append(chosen); available[chosen]=False
        for t in selected: A[new,t]=A[t,new]=1
    return nx.from_numpy_array(A)

def generate_ws(n,K,beta):
    half=K//2
    A=np.zeros((n,n),dtype=np.int8)
    for i in range(n):
        for off in range(1,half+1):
            j=(i+off)%n; A[i,j]=A[j,i]=1
    for i in range(n):
        for off in range(1,half+1):
            j=(i+off)%n
            if rng.rand() < beta:
                cand=np.where(A[i,:]==0)[0]
                cand=cand[cand!=i]
                if len(cand):
                    new=int(cand[rng.randint(len(cand))])
                    A[i,j]=A[j,i]=0
                    A[i,new]=A[new,i]=1
    G=nx.from_numpy_array(A)
    if not nx.is_connected(G):
        # Original code does not reject disconnected WS. With tested parameters this is extremely unlikely;
        # preserve diagnostics rather than silently modifying the graph.
        raise RuntimeError(f'Disconnected WS generated K={K}, beta={beta}')
    return G

def graph_distance_matrix(G):
    n=G.number_of_nodes(); D=np.full((n,n),np.inf)
    for s, lengths in nx.all_pairs_shortest_path_length(G):
        for t,d in lengths.items(): D[s,t]=d
    return D

def graph_features(G,D):
    n=G.number_of_nodes(); m=G.number_of_edges(); deg=np.array([G.degree(i) for i in range(n)],dtype=float)
    upper=D[np.triu_indices(n,1)]
    return dict(NumEdges=m,AvgDegree=deg.mean(),Density=2*m/(n*(n-1)),AvgClustering=nx.average_clustering(G),Diameter=float(D.max()),AvgPathLength=float(upper.mean()),deg=deg)

def candidate_set(G,D,deg,fraction,strategy):
    n=G.number_of_nodes(); s=max(1, round(fraction*n))
    if strategy=='random':
        S=rng.choice(n,size=s,replace=False)
    elif strategy=='high-degree':
        S=np.array(sorted(range(n),key=lambda v:(-deg[v],v))[:s],dtype=int)
    elif strategy=='distributed':
        maxd=deg.max(); first=int(np.where(deg==maxd)[0][0]); selected=[first]
        while len(selected)<s:
            cand=[v for v in range(n) if v not in selected]
            mind=np.min(D[np.ix_(cand,selected)],axis=1)
            mx=mind.max(); far=[cand[i] for i,val in enumerate(mind) if val==mx]
            md=max(deg[v] for v in far); best=[v for v in far if deg[v]==md]; selected.append(min(best))
        S=np.array(selected,dtype=int)
    else: raise ValueError(strategy)
    return np.sort(S)

def candidate_features(D,deg,S,n):
    ss=len(S)
    if ss>=2:
        ds=D[np.ix_(S,S)][np.triu_indices(ss,1)]
        meanpair=float(ds.mean()); minpair=float(ds.min())
    else: meanpair=np.nan; minpair=np.nan
    return dict(SetSize=ss,ActualFraction=ss/n,MeanDegreeS=float(deg[S].mean()),MeanPairDistanceS=meanpair,MinPairDistanceS=minpair)

def power_domination(G,S):
    n=G.number_of_nodes(); monitored=np.zeros(n,dtype=bool); monitored[S]=True
    for v in S:
        monitored[list(G.neighbors(int(v)))]=True
    c0=float(monitored.mean()); ppt=1; history=[int(monitored.sum())]
    while not monitored.all():
        newly=np.zeros(n,dtype=bool)
        for v in np.where(monitored)[0]:
            un=[u for u in G.neighbors(int(v)) if not monitored[u]]
            if len(un)==1: newly[un[0]]=True
        if not newly.any(): return False, np.inf, c0, history
        monitored |= newly; ppt+=1; history.append(int(monitored.sum()))
    return True, float(ppt), c0, history

def add_graph_rows(rows,G,topology,graph_id,target_k,beta,condition_group,condition_label):
    n=G.number_of_nodes(); D=graph_distance_matrix(G); gf=graph_features(G,D); deg=gf.pop('deg')
    for f in FRACTIONS:
        for cid in range(1,NUM_RANDOM+1):
            S=candidate_set(G,D,deg,f,'random'); cf=candidate_features(D,deg,S,n); ok,ppt,c0,h=power_domination(G,S)
            rows.append(dict(GraphID=graph_id,Topology=topology,N=n,**gf,Strategy='random',ConfigID=cid,RequestedFraction=f,**cf,C0=c0,IsPDS=int(ok),PPT=ppt if ok else np.nan,FinalCoverage=h[-1]/n,TargetAvgDegree=target_k,WSBeta=beta,ConditionGroup=condition_group,ConditionLabel=condition_label))
    for strategy in ['high-degree','distributed']:
        for f in FRACTIONS:
            S=candidate_set(G,D,deg,f,strategy); cf=candidate_features(D,deg,S,n); ok,ppt,c0,h=power_domination(G,S)
            rows.append(dict(GraphID=graph_id,Topology=topology,N=n,**gf,Strategy=strategy,ConfigID=1,RequestedFraction=f,**cf,C0=c0,IsPDS=int(ok),PPT=ppt if ok else np.nan,FinalCoverage=h[-1]/n,TargetAvgDegree=target_k,WSBeta=beta,ConditionGroup=condition_group,ConditionLabel=condition_label))

def build_dataset():
    rows=[]; t0=time.time()
    # connectivity regimes, WS beta fixed 0.1
    for k in [6,8,10]:
        for topo in ['ER','BA','WS']:
            for gid in range(1,NUM_GRAPHS+1):
                if topo=='ER': G=generate_er(N,k); beta=np.nan
                elif topo=='BA': G=generate_ba(N,k//2); beta=np.nan
                else: G=generate_ws(N,k,0.1); beta=0.1
                add_graph_rows(rows,G,topo,gid,k,beta,'connectivity',f'{topo}_k{k}')
            print('done',topo,k,'elapsed',round(time.time()-t0,1),flush=True)
    # extra WS rewiring conditions, K=8 (beta=.1 already above)
    for beta in [0.01,0.5]:
        for gid in range(1,NUM_GRAPHS+1):
            G=generate_ws(N,8,beta)
            add_graph_rows(rows,G,'WS',gid,8,beta,'ws_beta',f'WS_K8_beta{beta}')
        print('done WS beta',beta,'elapsed',round(time.time()-t0,1),flush=True)
    df=pd.DataFrame(rows)
    df.to_csv(OUT/'reviewer2_parameter_sensitivity_dataset.csv',index=False)
    return df

def graph_uid(df):
    b=df['WSBeta'].fillna(-1).astype(str)
    return df['ConditionLabel'].astype(str)+'_G'+df['GraphID'].astype(str)

def split_graphs(df,seed=SEED_MODEL):
    d=df.copy(); d['GraphUID']=graph_uid(d); ug=d['GraphUID'].unique().copy(); rr=np.random.RandomState(seed); rr.shuffle(ug); ntr=round(.8*len(ug)); tr=set(ug[:ntr]); te=set(ug[ntr:]); return d[d.GraphUID.isin(tr)].copy(), d[d.GraphUID.isin(te)].copy()

def eval_cls(y,p):
    tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel()
    return dict(Accuracy=accuracy_score(y,p),Precision=precision_score(y,p,zero_division=0),Recall=recall_score(y,p,zero_division=0),Specificity=tn/(tn+fp) if tn+fp else np.nan,F1=f1_score(y,p,zero_division=0),BalancedAccuracy=balanced_accuracy_score(y,p))

def fit_models(df):
    train,test=split_graphs(df)
    base=['N','ActualFraction','C0']
    struct=['N','AvgDegree','Density','AvgClustering','Diameter','AvgPathLength','Topology','Strategy','ActualFraction','MeanDegreeS','MeanPairDistanceS','MinPairDistanceS','C0']
    # A
    ytr=train.IsPDS.astype(int); yte=test.IsPDS.astype(int)
    mb=RandomForestClassifier(n_estimators=200,min_samples_leaf=5,random_state=SEED_MODEL,n_jobs=-1).fit(train[base],ytr)
    Xtr=pd.get_dummies(train[struct],columns=['Topology','Strategy'],dtype=int); Xte=pd.get_dummies(test[struct],columns=['Topology','Strategy'],dtype=int).reindex(columns=Xtr.columns,fill_value=0)
    ms=RandomForestClassifier(n_estimators=200,min_samples_leaf=5,random_state=SEED_MODEL,n_jobs=-1).fit(Xtr,ytr)
    A={'Baseline':eval_cls(yte,mb.predict(test[base])),'Structural':eval_cls(yte,ms.predict(Xte))}
    # B successful only with independent graph-level split as original
    s=df[df.IsPDS==1].copy(); tr,te=split_graphs(s); ytr=tr.PPT.astype(float); yte=te.PPT.astype(float)
    rb=RandomForestRegressor(n_estimators=200,min_samples_leaf=5,random_state=SEED_MODEL,n_jobs=-1).fit(tr[base],ytr)
    Xtrb=pd.get_dummies(tr[struct],columns=['Topology','Strategy'],dtype=int); Xteb=pd.get_dummies(te[struct],columns=['Topology','Strategy'],dtype=int).reindex(columns=Xtrb.columns,fill_value=0)
    rs=RandomForestRegressor(n_estimators=200,min_samples_leaf=5,random_state=SEED_MODEL,n_jobs=-1).fit(Xtrb,ytr)
    def reg(y,p): return dict(MAE=mean_absolute_error(y,p),RMSE=math.sqrt(mean_squared_error(y,p)),R2=r2_score(y,p))
    B={'Baseline':reg(yte,rb.predict(te[base])),'Structural':reg(yte,rs.predict(Xteb))}
    return A,B

def descriptive(df):
    g=df.groupby(['ConditionGroup','ConditionLabel','Topology','TargetAvgDegree','WSBeta'],dropna=False).agg(Rows=('IsPDS','size'),Graphs=('GraphID','nunique'),SuccessRate=('IsPDS','mean'),MeanC0=('C0','mean')).reset_index()
    ppt=df[df.IsPDS==1].groupby(['ConditionGroup','ConditionLabel','Topology','TargetAvgDegree','WSBeta'],dropna=False).PPT.agg(['mean','median','min','max','count']).reset_index().rename(columns={'mean':'MeanPPT','median':'MedianPPT','min':'MinPPT','max':'MaxPPT','count':'SuccessfulRows'})
    return g.merge(ppt,on=['ConditionGroup','ConditionLabel','Topology','TargetAvgDegree','WSBeta'],how='left')

def per_regime_models(df):
    rows=[]
    conn=df[df.ConditionGroup=='connectivity']
    for k in [6,8,10]:
        A,B=fit_models(conn[conn.TargetAvgDegree==k])
        for model,v in A.items(): rows.append({'Analysis':'Connectivity','Regime':f'k={k}','Task':'A','Model':model,**v})
        for model,v in B.items(): rows.append({'Analysis':'Connectivity','Regime':f'k={k}','Task':'B','Model':model,**v})
    # pooled connectivity across k
    A,B=fit_models(conn)
    for model,v in A.items(): rows.append({'Analysis':'Connectivity','Regime':'pooled k=6,8,10','Task':'A','Model':model,**v})
    for model,v in B.items(): rows.append({'Analysis':'Connectivity','Regime':'pooled k=6,8,10','Task':'B','Model':model,**v})
    # beta sensitivity dataset includes beta=.1 from connectivity K8 and extras
    beta=pd.concat([conn[(conn.Topology=='WS')&(conn.TargetAvgDegree==8)],df[df.ConditionGroup=='ws_beta']],ignore_index=True)
    A,B=fit_models(beta)
    for model,v in A.items(): rows.append({'Analysis':'WS beta','Regime':'pooled beta=.01,.1,.5','Task':'A','Model':model,**v})
    for model,v in B.items(): rows.append({'Analysis':'WS beta','Regime':'pooled beta=.01,.1,.5','Task':'B','Model':model,**v})
    # each beta separately
    for b in [0.01,0.1,0.5]:
        sub=beta[np.isclose(beta.WSBeta,b)]
        A,B=fit_models(sub)
        for model,v in A.items(): rows.append({'Analysis':'WS beta','Regime':f'beta={b}','Task':'A','Model':model,**v})
        for model,v in B.items(): rows.append({'Analysis':'WS beta','Regime':f'beta={b}','Task':'B','Model':model,**v})
    return pd.DataFrame(rows), beta

def apl_correlations(df):
    s=df[df.IsPDS==1].copy(); out=[]
    # pooled fixed C0 bins
    s['C0Bin']=pd.cut(s.C0,bins=np.arange(0,1.0001,0.1),include_lowest=True,right=False)
    for name,sub in [('all sensitivity',s),('connectivity pooled',s[s.ConditionGroup=='connectivity'])]:
        for binv,g in sub.groupby('C0Bin',observed=True):
            if len(g)>=100 and g.AvgPathLength.nunique()>1 and g.PPT.nunique()>1:
                out.append({'Scope':name,'Regime':str(binv),'N':len(g),'Correlation_APL_PPT':g[['AvgPathLength','PPT']].corr().iloc[0,1]})
    conn=s[s.ConditionGroup=='connectivity']
    for k in [6,8,10]:
        g=conn[conn.TargetAvgDegree==k]
        out.append({'Scope':'connectivity by k','Regime':f'k={k}','N':len(g),'Correlation_APL_PPT':g[['AvgPathLength','PPT']].corr().iloc[0,1]})
    beta=pd.concat([conn[(conn.Topology=='WS')&(conn.TargetAvgDegree==8)],s[s.ConditionGroup=='ws_beta']],ignore_index=True)
    for b in [0.01,0.1,0.5]:
        g=beta[np.isclose(beta.WSBeta,b)]
        out.append({'Scope':'WS beta','Regime':f'beta={b}','N':len(g),'Correlation_APL_PPT':g[['AvgPathLength','PPT']].corr().iloc[0,1]})
    return pd.DataFrame(out)

def validate_against_original(df):
    try: orig=pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "pds_propagation_final_results.csv")
    except FileNotFoundError: return pd.DataFrame()
    orig=orig[orig.N==100].copy()
    rec=df[(df.ConditionGroup=='connectivity')&(df.TargetAvgDegree==8)].copy()
    rows=[]
    for source,d in [('Original',orig),('Recreated',rec)]:
        for topo in ['ER','BA','WS']:
            x=d[d.Topology==topo]; ss=x[x.IsPDS==1]
            rows.append({'Source':source,'Topology':topo,'Rows':len(x),'SuccessRate':x.IsPDS.mean(),'MeanPPT_success':ss.PPT.mean(),'MeanC0':x.C0.mean(),'MeanAvgDegree':x.AvgDegree.mean(),'MeanClustering':x.AvgClustering.mean(),'MeanAPL':x.AvgPathLength.mean()})
    return pd.DataFrame(rows)

if __name__=='__main__':
    df=build_dataset()
    desc=descriptive(df); desc.to_csv(OUT/'descriptive_by_condition.csv',index=False)
    perf,beta=per_regime_models(df); perf.to_csv(OUT/'model_performance_by_regime.csv',index=False)
    corr=apl_correlations(df); corr.to_csv(OUT/'apl_ppt_correlations.csv',index=False)
    val=validate_against_original(df); val.to_csv(OUT/'validation_vs_original_N100_k8.csv',index=False)
    summary={
      'rows':len(df),'graphs':df.assign(GraphUID=graph_uid(df)).GraphUID.nunique(),
      'success_rate':float(df.IsPDS.mean()),
      'outputs':[p.name for p in OUT.iterdir()]
    }
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    print('\nSUMMARY',json.dumps(summary,indent=2))
    print('\nDESCRIPTIVE\n',desc.to_string(index=False))
    print('\nPERFORMANCE\n',perf.to_string(index=False))
    print('\nCORRELATIONS\n',corr.to_string(index=False))
    print('\nVALIDATION\n',val.to_string(index=False))
