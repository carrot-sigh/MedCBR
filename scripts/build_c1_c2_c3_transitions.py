from pathlib import Path
import pandas as pd

root=Path('/home/user/gxq/MedCBR/workspaces/chest_imagenome_audit_full')
out=root/'transitions'; out.mkdir(exist_ok=True)
a=pd.read_parquet(root/'atomic_annotations.parquet',columns=['dicom_id','study_id','region','category','raw_label','relation'])
# Explicit assertion states only; unknown is retained in C1 separately but not counted as evidence for transitions.
c2=a[a.category.eq('anatomicalfinding')].assign(c2_concept=lambda x:x.raw_label.str.lower(), state=lambda x:x.relation.str.lower())
c2=c2[['dicom_id','study_id','region','c2_concept','state']].drop_duplicates()
c2_yes=c2[c2.state.eq('yes')][['dicom_id','study_id','region','c2_concept']].drop_duplicates()
c3=a[a.category.eq('disease')].assign(c3_concept=lambda x:x.raw_label.str.lower(), state=lambda x:x.relation.str.lower())
c3=c3[c3.state.eq('yes')][['dicom_id','study_id','c3_concept']].drop_duplicates()
c1=pd.read_parquet(root/'c1/c1_texture_region_labels.parquet',columns=['dicom_id','study_id','region','canonical','state'])
c1=c1[c1.state.eq('yes')].rename(columns={'canonical':'c1_concept'})[['dicom_id','study_id','region','c1_concept']].drop_duplicates()
# C1 -> C2: same image and canonical region.
c12=c1.merge(c2_yes,on=['dicom_id','study_id','region'],how='inner')
c12s=c12.groupby(['c1_concept','c2_concept'],as_index=False).agg(image_count=('dicom_id','nunique'),region_instance_count=('region','size'),study_count=('study_id','nunique')).sort_values('image_count',ascending=False)
c12s.to_csv(out/'c1_c2_same_region.csv',index=False)
# C2 -> C3: same image/study, region is intentionally ignored.
c23=c2_yes[['dicom_id','study_id','c2_concept']].drop_duplicates().merge(c3,on=['dicom_id','study_id'],how='inner')
c23s=c23.groupby(['c2_concept','c3_concept'],as_index=False).agg(image_count=('dicom_id','nunique'),study_count=('study_id','nunique')).sort_values('image_count',ascending=False)
c23s.to_csv(out/'c2_c3_same_study.csv',index=False)
# C1 -> C2 -> C3: C1/C2 same region, then C3 same image/study.
tri=c12.merge(c3,on=['dicom_id','study_id'],how='inner')
tris=tri.groupby(['c1_concept','c2_concept','c3_concept'],as_index=False).agg(image_count=('dicom_id','nunique'),region_instance_count=('region','size'),study_count=('study_id','nunique')).sort_values('image_count',ascending=False)
tris.to_csv(out/'c1_c2_c3_same_study.csv',index=False)
# Marginal coverage and metadata.
summary=pd.DataFrame([
 {'table':'c1_c2_same_region','rows':len(c12s),'evidence_instances':len(c12),'images':c12.dicom_id.nunique()},
 {'table':'c2_c3_same_study','rows':len(c23s),'evidence_instances':len(c23),'images':c23.dicom_id.nunique()},
 {'table':'c1_c2_c3_same_study','rows':len(tris),'evidence_instances':len(tri),'images':tri.dicom_id.nunique()},
])
summary.to_csv(out/'transition_summary.csv',index=False)
print(summary.to_string(index=False)); print('\nTop C1-C2'); print(c12s.head(20).to_string(index=False)); print('\nTop C2-C3'); print(c23s.head(20).to_string(index=False)); print('\nTop triples'); print(tris.head(20).to_string(index=False))
