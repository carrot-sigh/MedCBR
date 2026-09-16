from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path('gxq/MedCBR/workspaces/chest_imagenome_audit_full')
OUT=ROOT/'hierarchy_audit'; OUT.mkdir(parents=True,exist_ok=True)
N_IMAGES=243076  # images represented in atomic_annotations.parquet
# Positive C1 region labels are the final, polarity-resolved labels.
c1=pd.read_parquet(ROOT/'c1/c1_texture_region_labels.parquet',columns=['dicom_id','study_id','region','c1_concept','canonical','state'] if False else ['dicom_id','study_id','region','canonical','state'])
c1=c1[c1.state.eq('yes')].rename(columns={'canonical':'c1_concept'})[['dicom_id','study_id','region','c1_concept']].drop_duplicates()
# Positive C2/C3 assertions are explicit yes assertions.
a=pd.read_parquet(ROOT/'atomic_annotations.parquet',columns=['dicom_id','study_id','region','category','raw_label','relation'])
c2=a[(a.category=='anatomicalfinding')&(a.relation=='yes')].rename(columns={'raw_label':'c2_concept'})[['dicom_id','study_id','region','c2_concept']].drop_duplicates()
c3=a[(a.category=='disease')&(a.relation=='yes')].rename(columns={'raw_label':'c3_concept'})[['dicom_id','study_id','c3_concept']].drop_duplicates()
# Vocabulary audit, including regions outside the intended 29-object design.
regions=pd.DataFrame({'region':sorted(set(c1.region)|set(c2.region))})
regions['c1_images']=regions.region.map(c1.groupby('region').dicom_id.nunique()).fillna(0).astype(int)
regions['c2_images']=regions.region.map(c2.groupby('region').dicom_id.nunique()).fillna(0).astype(int)
regions.to_csv(OUT/'region_vocabulary.csv',index=False)
# Marginals.
c1_img=c1.groupby('c1_concept').dicom_id.nunique().rename('c1_images')
c2_img=c2.groupby('c2_concept').dicom_id.nunique().rename('c2_images')
c3_img=c3.groupby('c3_concept').dicom_id.nunique().rename('c3_images')
c1_regions=c1.groupby('c1_concept').region.nunique().rename('c1_regions')
# Same-region C1-C2 and same-study C2-C3 supports.
c12=c1.merge(c2,on=['dicom_id','study_id','region'],how='inner').drop_duplicates()
c23=c2[['dicom_id','study_id','c2_concept']].drop_duplicates().merge(c3,on=['dicom_id','study_id'],how='inner').drop_duplicates()
tri=c12.merge(c3,on=['dicom_id','study_id'],how='inner').drop_duplicates()
# Conditional paths, with image-level supports. C1-C2 is region grounded; C2-C3 is study/image grounded.
path12=c12.groupby(['c1_concept','c2_concept']).agg(support_images=('dicom_id','nunique'),support_regions=('region','size')).reset_index()
path12['c1_images']=path12.c1_concept.map(c1_img)
path12['c2_images']=path12.c2_concept.map(c2_img)
path12['p_c2_given_c1']=path12.support_images/path12.c1_images
path12['baseline_p_c2']=path12.c2_images/N_IMAGES
path12['lift_c1_c2']=path12.p_c2_given_c1/path12.baseline_p_c2
path12['pmi_c1_c2']=np.log2(path12.lift_c1_c2)
path12=path12.sort_values(['lift_c1_c2','support_images'],ascending=False)
path12.to_csv(OUT/'c1_c2_strength.csv',index=False)
path23=c23.groupby(['c2_concept','c3_concept']).agg(support_images=('dicom_id','nunique')).reset_index()
path23['c2_images']=path23.c2_concept.map(c2_img)
path23['c3_images']=path23.c3_concept.map(c3_img)
path23['p_c3_given_c2']=path23.support_images/path23.c2_images
path23['baseline_p_c3']=path23.c3_images/N_IMAGES
path23['lift_c2_c3']=path23.p_c3_given_c2/path23.baseline_p_c3
path23['pmi_c2_c3']=np.log2(path23.lift_c2_c3)
path23=path23.sort_values(['lift_c2_c3','support_images'],ascending=False)
path23.to_csv(OUT/'c2_c3_strength.csv',index=False)
# Triple paths. P(C3|C1,C2) uses the C1-C2 same-region support as denominator.
path123=tri.groupby(['c1_concept','c2_concept','c3_concept']).agg(support_images=('dicom_id','nunique'),support_regions=('region','size')).reset_index()
den=path12[['c1_concept','c2_concept','support_images']].rename(columns={'support_images':'c1_c2_support_images'})
path123=path123.merge(den,on=['c1_concept','c2_concept'],how='left')
path123['c3_images']=path123.c3_concept.map(c3_img)
path123['p_c3_given_c1_c2']=path123.support_images/path123.c1_c2_support_images
path123['baseline_p_c3']=path123.c3_images/N_IMAGES
path123['lift_c1_c2_c3']=path123.p_c3_given_c1_c2/path123.baseline_p_c3
path123['pmi_c1_c2_c3']=np.log2(path123.lift_c1_c2_c3)
path123=path123.sort_values(['lift_c1_c2_c3','support_images'],ascending=False)
path123.to_csv(OUT/'c1_c2_c3_strength.csv',index=False)
# C1 quality/imbalance table.
q=pd.DataFrame({'concept':sorted(set(c1.c1_concept)|set(pd.read_csv(ROOT/'c1/c1_texture_statistics.csv').canonical))})
old=pd.read_csv(ROOT/'c1/c1_texture_statistics.csv').rename(columns={'canonical':'concept'})
q=q.merge(old,on='concept',how='left').merge(c1_img.rename_axis('concept').reset_index(),on='concept',how='left').merge(c1_regions.rename_axis('concept').reset_index(),on='concept',how='left')
q['image_coverage']=q.c1_images.fillna(0)/N_IMAGES
q['positive_rate']=q.positive_count/(q.positive_count+q.negative_count+q.unknown_count)
q['negative_rate']=q.negative_count/(q.positive_count+q.negative_count+q.unknown_count)
q['unknown_rate']=q.unknown_count/(q.positive_count+q.negative_count+q.unknown_count)
q['mapping_source']='texture_cues -> compatible anatomicalfinding'
q['mapping_confidence']=q.apply(lambda r:'high' if r.unknown_rate < .2 else ('medium' if r.unknown_rate < .7 else 'low'),axis=1)
q.to_csv(OUT/'c1_concept_quality.csv',index=False)
summary={
 'image_denominator':N_IMAGES,'c1_positive_images':int(c1.dicom_id.nunique()),'c2_positive_images':int(c2.dicom_id.nunique()),'c3_positive_images':int(c3.dicom_id.nunique()),
 'region_vocabulary_size':len(regions),'c1_region_vocabulary_size':int(c1.region.nunique()),'c2_region_vocabulary_size':int(c2.region.nunique()),
 'c1_concepts':int(c1.c1_concept.nunique()),'c2_concepts':int(c2.c2_concept.nunique()),'c3_concepts':int(c3.c3_concept.nunique()),
 'c1_c2_pairs':len(path12),'c2_c3_pairs':len(path23),'c1_c2_c3_triples':len(path123),
 'definitions':{'c1_c2':'same dicom_id + canonical region','c2_c3':'same dicom_id/study; region ignored','triple':'C1/C2 same region, C3 same dicom_id/study','probability_denominator':'unique positive image supports','unknown_policy':'unknown C1 is excluded from positive transition evidence'},
 'interpretation_limits': {
  'c1_c2_circularity': 'C1 polarity is weakly supervised from compatible C2 finding assertions, so C1-C2 strength measures extraction/ontology consistency rather than independent causal hierarchy evidence.',
  'c2_c3': 'C2-C3 statistics are natural same-study associations, not causal rules.',
  'triples': 'C1-C2-C3 statistics describe supported paths and coverage; their strength partly inherits C1-C2 construction dependence.',
 },
}
json.dump(summary,open(OUT/'audit_summary.json','w'),indent=2)
print(json.dumps(summary,indent=2))
print('\nRegions:',regions.to_string(index=False))
print('\nTop C1-C2 strength:',path12.head(15).to_string(index=False))
print('\nTop C2-C3 strength:',path23.head(15).to_string(index=False))
print('\nTop triples:',path123.head(15).to_string(index=False))
print('\nC1 quality:',q.to_string(index=False))
