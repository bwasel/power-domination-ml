import math, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
SEED_GEN=1; SEED_MODEL=2026; N=500; NUM_GRAPHS=50; FRACTIONS=[0.05,0.10,0.15]; NUM_RANDOM=5
OUT=Path(__file__).resolve().parent / 'network_generalizability_outputs'; OUT.mkdir(exist_ok=True)
rng=np.random.RandomState(SEED_GEN)

def generate_er(n,avg_degree=8):
 p=avg_degree/(n-1)
 for _ in range(1000):
  R=rng.rand(n,n); A=np.triu(R<p,1); A=A+A.T; G=nx.from_numpy_array(A.astype(np.int8))
  if nx.is_connected(G): return G
 raise RuntimeError('ER connectivity failure')
def generate_ba(n,m=4):
 m0=m+1; A=np.zeros((n,n),dtype=np.int8)
 for i in range(m0):
  for j in range(i+1,m0): A[i,j]=A[j,i]=1
 for new in range(m0,n):
  deg=A[:new,:new].sum(axis=1).astype(float); available=np.ones(new,dtype=bool); selected=[]
  for _ in range(m):
   w=deg.copy(); w[~available]=0; probs=w/w.sum(); c=np.cumsum(probs); chosen=int(np.searchsorted(c,rng.rand(),side='left')); selected.append(chosen); available[chosen]=False
  for t in selected: A[new,t]=A[t,new]=1
 return nx.from_numpy_array(A)
def generate_ws(n,K=8,beta=0.1):
 half=K//2; A=np.zeros((n,n),dtype=np.int8)
 for i in range(n):
  for off in range(1,half+1):
   j=(i+off)%n; A[i,j]=A[j,i]=1
 for i in range(n):
  for off in range(1,half+1):
   j=(i+off)%n
   if rng.rand()<beta:
    cand=np.where(A[i,:]==0)[0]; cand=cand[cand!=i]
    if len(cand):
     new=int(cand[rng.randint(len(cand))]); A[i,j]=A[j,i]=0; A[i,new]=A[new,i]=1
 G=nx.from_numpy_array(A)
 if not nx.is_connected(G): raise RuntimeError('Disconnected WS')
 return G
def generate_sbm(n=500,blocks=4,p_in=0.05,p_out=0.005):
 bs=n//blocks; membership=np.repeat(np.arange(blocks),bs)
 for _ in range(1000):
  R=rng.rand(n,n); same=membership[:,None]==membership[None,:]; P=np.where(same,p_in,p_out); A=np.triu(R<P,1); A=A+A.T; G=nx.from_numpy_array(A.astype(np.int8))
  if nx.is_connected(G):
   perm=rng.permutation(n); A2=A[np.ix_(perm,perm)]; return nx.from_numpy_array(A2.astype(np.int8))
 raise RuntimeError('SBM connectivity failure')
def distance_matrix(G):
 A=nx.to_scipy_sparse_array(G,format='csr',dtype=np.float64); return shortest_path(csr_matrix(A),directed=False,unweighted=True)
def graph_features(G,D):
 n=G.number_of_nodes(); m=G.number_of_edges(); deg=np.array([G.degree(i) for i in range(n)],dtype=float); upper=D[np.triu_indices(n,1)]
 return dict(NumEdges=m,AvgDegree=float(deg.mean()),Density=float(2*m/(n*(n-1))),AvgClustering=float(nx.average_clustering(G)),Diameter=float(D.max()),AvgPathLength=float(upper.mean()),deg=deg)
def candidate_set(D,deg,fraction,strategy):
 n=len(deg); s=max(1,int(round(fraction*n)))
 if strategy=='random': S=rng.choice(n,size=s,replace=False)
 elif strategy=='high-degree': S=np.array(sorted(range(n),key=lambda v:(-deg[v],v))[:s],dtype=int)
 elif strategy=='distributed':
  first=int(np.where(deg==deg.max())[0][0]); selected=[first]; mask=np.zeros(n,dtype=bool); mask[first]=True; md=D[:,first].copy()
  while len(selected)<s:
   tmp=md.copy(); tmp[mask]=-1; mx=tmp.max(); far=np.where(tmp==mx)[0]; dm=deg[far].max(); best=far[deg[far]==dm]; v=int(best.min()); selected.append(v); mask[v]=True; md=np.minimum(md,D[:,v])
  S=np.array(selected,dtype=int)
 else: raise ValueError(strategy)
 return np.sort(S)
def candidate_features(D,deg,S,n):
 ss=len(S)
 if ss>=2:
  ds=D[np.ix_(S,S)][np.triu_indices(ss,1)]; meanpair=float(ds.mean()); minpair=float(ds.min())
 else: meanpair=np.nan; minpair=np.nan
 return dict(SetSize=ss,ActualFraction=ss/n,MeanDegreeS=float(deg[S].mean()),MeanPairDistanceS=meanpair,MinPairDistanceS=minpair)
def power_domination(G,S):
 n=G.number_of_nodes(); mon=np.zeros(n,dtype=bool); mon[S]=True; neigh=[np.fromiter(G.neighbors(i),dtype=int) for i in range(n)]
 for v in S: mon[neigh[int(v)]]=True
 c0=float(mon.mean()); ppt=1; hist=[int(mon.sum())]
 while not mon.all():
  newly=np.zeros(n,dtype=bool)
  for v in np.where(mon)[0]:
   un=neigh[int(v)][~mon[neigh[int(v)]]]
   if len(un)==1: newly[un[0]]=True
  if not newly.any(): return False,np.inf,c0,hist
  mon|=newly; ppt+=1; hist.append(int(mon.sum()))
 return True,float(ppt),c0,hist
def add_graph_rows(rows,G,topo,gid):
 n=G.number_of_nodes(); D=distance_matrix(G); gf=graph_features(G,D); deg=gf.pop('deg')
 for f in FRACTIONS:
  for cid in range(1,NUM_RANDOM+1):
   S=candidate_set(D,deg,f,'random'); cf=candidate_features(D,deg,S,n); ok,ppt,c0,h=power_domination(G,S); rows.append(dict(GraphID=gid,Topology=topo,N=n,**gf,Strategy='random',ConfigID=cid,RequestedFraction=f,**cf,C0=c0,IsPDS=int(ok),PPT=ppt,FinalCoverage=h[-1]/n,TargetAvgDegree=8))
 for strategy in ['high-degree','distributed']:
  for f in FRACTIONS:
   S=candidate_set(D,deg,f,strategy); cf=candidate_features(D,deg,S,n); ok,ppt,c0,h=power_domination(G,S); rows.append(dict(GraphID=gid,Topology=topo,N=n,**gf,Strategy=strategy,ConfigID=1,RequestedFraction=f,**cf,C0=c0,IsPDS=int(ok),PPT=ppt,FinalCoverage=h[-1]/n,TargetAvgDegree=8))
def build_dataset():
 rows=[]; t0=time.time(); gens={'ER':lambda:generate_er(N,8),'BA':lambda:generate_ba(N,4),'WS':lambda:generate_ws(N,8,0.1),'SBM':lambda:generate_sbm(N,4,0.05,0.005)}
 for topo,gen in gens.items():
  for gid in range(1,NUM_GRAPHS+1):
   add_graph_rows(rows,gen(),topo,gid)
   if gid%10==0: print(f'done {topo} {gid}/{NUM_GRAPHS} elapsed={time.time()-t0:.1f}s',flush=True)
 df=pd.DataFrame(rows); df.to_csv(OUT/'extended_network_dataset.csv',index=False); return df
BASE=['N','ActualFraction','C0']
STRUCT_FULL=['N','AvgDegree','Density','AvgClustering','Diameter','AvgPathLength','Topology','Strategy','ActualFraction','MeanDegreeS','MeanPairDistanceS','MinPairDistanceS','C0']
STRUCT_AG=[x for x in STRUCT_FULL if x!='Topology']
def prep(train,test,features):
 cats=[c for c in ['Topology','Strategy'] if c in features]
 if cats:
  Xtr=pd.get_dummies(train[features],columns=cats,dtype=int); Xte=pd.get_dummies(test[features],columns=cats,dtype=int).reindex(columns=Xtr.columns,fill_value=0)
 else: Xtr=train[features].copy(); Xte=test[features].copy()
 return Xtr,Xte
def cls_metrics(y,p):
 tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel(); return dict(Accuracy=accuracy_score(y,p),F1=f1_score(y,p,zero_division=0),BalancedAccuracy=balanced_accuracy_score(y,p),Precision=precision_score(y,p,zero_division=0),Recall=recall_score(y,p,zero_division=0),Specificity=tn/(tn+fp) if tn+fp else np.nan)
def reg_metrics(y,p): return dict(MAE=mean_absolute_error(y,p),RMSE=math.sqrt(mean_squared_error(y,p)),R2=r2_score(y,p))
def external_eval(orig,ext):
 rows=[]; ytr=orig.IsPDS.astype(int); cb=RandomForestClassifier(n_estimators=200,min_samples_leaf=5,random_state=SEED_MODEL,n_jobs=-1).fit(orig[BASE],ytr)
 Xf,_=prep(orig,orig.iloc[:1],STRUCT_FULL); cf=RandomForestClassifier(n_estimators=200,min_samples_leaf=5,random_state=SEED_MODEL,n_jobs=-1).fit(Xf,ytr)
 Xa,_=prep(orig,orig.iloc[:1],STRUCT_AG); ca=RandomForestClassifier(n_estimators=200,min_samples_leaf=5,random_state=SEED_MODEL,n_jobs=-1).fit(Xa,ytr)
 for topo in ['ER','BA','WS','SBM']:
  te=ext[ext.Topology==topo]; y=te.IsPDS.astype(int); rows.append({'Evaluation':'External transfer','Task':'A','Condition':topo,'Model':'Baseline','FeatureMode':'Baseline',**cls_metrics(y,cb.predict(te[BASE]))}); feat=STRUCT_FULL if topo!='SBM' else STRUCT_AG; mod=cf if topo!='SBM' else ca; _,Xt=prep(orig,te,feat); rows.append({'Evaluation':'External transfer','Task':'A','Condition':topo,'Model':'Structural','FeatureMode':'Full' if topo!='SBM' else 'Topology-agnostic',**cls_metrics(y,mod.predict(Xt))})
 tr=orig[orig.IsPDS==1]; ytr=tr.PPT.astype(float); rb=RandomForestRegressor(n_estimators=200,min_samples_leaf=5,random_state=SEED_MODEL,n_jobs=-1).fit(tr[BASE],ytr); Xf,_=prep(tr,tr.iloc[:1],STRUCT_FULL); rf=RandomForestRegressor(n_estimators=200,min_samples_leaf=5,random_state=SEED_MODEL,n_jobs=-1).fit(Xf,ytr); Xa,_=prep(tr,tr.iloc[:1],STRUCT_AG); ra=RandomForestRegressor(n_estimators=200,min_samples_leaf=5,random_state=SEED_MODEL,n_jobs=-1).fit(Xa,ytr)
 for topo in ['ER','BA','WS','SBM']:
  te=ext[(ext.Topology==topo)&(ext.IsPDS==1)]; y=te.PPT.astype(float); rows.append({'Evaluation':'External transfer','Task':'B','Condition':topo,'Model':'Baseline','FeatureMode':'Baseline',**reg_metrics(y,rb.predict(te[BASE]))}); feat=STRUCT_FULL if topo!='SBM' else STRUCT_AG; mod=rf if topo!='SBM' else ra; _,Xt=prep(tr,te,feat); rows.append({'Evaluation':'External transfer','Task':'B','Condition':topo,'Model':'Structural','FeatureMode':'Full' if topo!='SBM' else 'Topology-agnostic',**reg_metrics(y,mod.predict(Xt))})
 return pd.DataFrame(rows)
def split_extended(ext):
 parts=[]; rr=np.random.RandomState(SEED_MODEL)
 for topo,g in ext.groupby('Topology',sort=False):
  ids=np.array(sorted(g.GraphID.unique())); rr.shuffle(ids); tr=set(ids[:round(.8*len(ids))]); z=g.copy(); z['Split']=np.where(z.GraphID.isin(tr),'train','test'); parts.append(z)
 return pd.concat(parts,ignore_index=True)
def holdout_eval(ext):
 d=split_extended(ext); train=d[d.Split=='train']; test=d[d.Split=='test']; rows=[]; ytr=train.IsPDS.astype(int)
 for name,feat in [('Baseline',BASE),('Structural',STRUCT_FULL)]:
  Xtr,Xte=prep(train,test,feat); m=RandomForestClassifier(n_estimators=200,min_samples_leaf=5,random_state=SEED_MODEL,n_jobs=-1).fit(Xtr,ytr); pred=m.predict(Xte)
  for cond,mask in [('Pooled',np.ones(len(test),dtype=bool))]+[(t,(test.Topology==t).to_numpy()) for t in ['ER','BA','WS','SBM']]: rows.append({'Evaluation':'Extended holdout','Task':'A','Condition':cond,'Model':name,**cls_metrics(test.IsPDS.astype(int).to_numpy()[mask],pred[mask])})
 tr=train[train.IsPDS==1]; te=test[test.IsPDS==1]; ytr=tr.PPT.astype(float)
 for name,feat in [('Baseline',BASE),('Structural',STRUCT_FULL)]:
  Xtr,Xte=prep(tr,te,feat); m=RandomForestRegressor(n_estimators=200,min_samples_leaf=5,random_state=SEED_MODEL,n_jobs=-1).fit(Xtr,ytr); pred=m.predict(Xte)
  for cond,mask in [('Pooled',np.ones(len(te),dtype=bool))]+[(t,(te.Topology==t).to_numpy()) for t in ['ER','BA','WS','SBM']]: rows.append({'Evaluation':'Extended holdout','Task':'B','Condition':cond,'Model':name,**reg_metrics(te.PPT.astype(float).to_numpy()[mask],pred[mask])})
 return pd.DataFrame(rows)
def descriptive(ext):
 out=[]
 for topo,g in ext.groupby('Topology'):
  s=g[g.IsPDS==1]; out.append(dict(Topology=topo,N=int(g.N.iloc[0]),Graphs=g.GraphID.nunique(),Rows=len(g),SuccessRate=g.IsPDS.mean(),MeanPPT_success=s.PPT.mean(),MedianPPT_success=s.PPT.median(),MeanC0=g.C0.mean(),MeanAvgDegree=g.AvgDegree.mean(),MeanClustering=g.AvgClustering.mean(),MeanAPL=g.AvgPathLength.mean(),MeanDiameter=g.Diameter.mean()))
 return pd.DataFrame(out)
def main():
 t0=time.time(); ext=build_dataset(); orig=pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "pds_propagation_final_results.csv"); ex=external_eval(orig,ext); ho=holdout_eval(ext); desc=descriptive(ext); ex.to_csv(OUT/'external_transfer_metrics.csv',index=False); ho.to_csv(OUT/'extended_holdout_metrics.csv',index=False); desc.to_csv(OUT/'extended_network_descriptive.csv',index=False); summary={'dataset_rows':len(ext),'graphs':int(ext.groupby(['Topology','GraphID']).ngroups),'conditions':['ER_N500','BA_N500','WS_N500','SBM_N500'],'graphs_per_condition':50,'configurations_per_graph':21,'sbm':{'blocks':4,'block_size':125,'p_in':0.05,'p_out':0.005,'expected_average_degree':8.075},'elapsed_seconds':time.time()-t0}; (OUT/'analysis_summary.json').write_text(json.dumps(summary,indent=2)); print('\nDESCRIPTIVE\n'+desc.to_string(index=False)); print('\nEXTERNAL\n'+ex.to_string(index=False)); print('\nHOLDOUT\n'+ho.to_string(index=False)); print('\nSUMMARY\n'+json.dumps(summary,indent=2))
if __name__=='__main__': main()
