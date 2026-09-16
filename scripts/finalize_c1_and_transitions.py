from pathlib import Path
import json
import pandas as pd

root=Path('/home/user/gxq/MedCBR/workspaces/chest_imagenome_audit_full')
out=root/'c1'; out.mkdir(exist_ok=True)
mapdf=pd.read_csv(root/'texture_finding_mapping.csv')
phrase=pd.read_parquet(out/'c1_texture_phrase_annotations.parquet')
keys=['dicom_id','region','phrase_id']
# Keep only findings that can affect a texture cue, then join by the exact phrase.
a=pd.read_parquet(root/'atomic_annotations.parquet',columns=keys+['category','raw_label','relation'])
f=a[(a.category=='anatomicalfinding') & a.raw_label.isin(mapdf.finding)].drop_duplicates(keys+['raw_label','relation'])
matched=f.merge(mapdf,left_on='raw_label',right_on='finding',how='inner')
pol=matched.groupby(keys+['texture'],as_index=False).agg(
    has_yes=('relation',lambda s: bool((s=='yes').any())),
    has_no=('relation',lambda s: bool((s=='no').any())))
pol['state']=pol.apply(lambda r:'yes' if r.has_yes and not r.has_no else ('no' if r.has_no and not r.has_yes else 'unknown'),axis=1)
phrase=phrase.drop(columns=['state','matched_findings'],errors='ignore').merge(pol,left_on=keys+['texture_raw'],right_on=keys+['texture'],how='left')
phrase['state']=phrase.state.fillna('unknown')
phrase=phrase.drop(columns=['texture','has_yes','has_no'],errors='ignore')
phrase=phrase.sort_values(['dicom_id','region','canonical','phrase_id'])
gkey=['dicom_id','region','canonical']
last=phrase.drop_duplicates(gkey,keep='last')
explicit=phrase[phrase.state.isin(['yes','no'])].drop_duplicates(gkey,keep='last')[gkey+['state']].rename(columns={'state':'explicit_state'})
region=last.merge(explicit,on=gkey,how='left')
region['state']=region.explicit_state.fillna(region.state)
region=region.drop(columns=['explicit_state'])
phrase.to_parquet(out/'c1_texture_phrase_annotations.parquet',index=False)
region.to_parquet(out/'c1_texture_region_labels.parquet',index=False)
st=region.groupby('canonical').agg(positive_count=('state',lambda s:int((s=='yes').sum())),negative_count=('state',lambda s:int((s=='no').sum())),unknown_count=('state',lambda s:int((s=='unknown').sum())),image_count=('dicom_id','nunique'),region_count=('region','nunique')).reset_index()
st.to_csv(out/'c1_texture_statistics.csv',index=False)
json.dump({'phrase_rows':len(phrase),'region_labels':len(region),'concepts':int(region.canonical.nunique()),'images':int(region.dicom_id.nunique())},open(out/'c1_build_summary.json','w'),indent=2)
print(st.to_string(index=False))
print(json.dumps({'phrase_rows':len(phrase),'region_labels':len(region),'concepts':int(region.canonical.nunique()),'images':int(region.dicom_id.nunique())}))
