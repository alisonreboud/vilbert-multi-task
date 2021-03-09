import pandas as pd
"""vilbert=pd.read_csv('st_surrey_vilbert_big_segments.csv',names=['index','scores'])





text=pd.read_csv('new_surrey_results_ismail.csv')



picsom=pd.read_csv('picsom_big_segments.csv',names=['index','scores'])
print(picsom['scores'])



ensemble_score=0.5*picsom['scores']+ 0.2* vilbert['scores']+0.3*text['results_st']
"""
df=pd.read_csv('Surrey_bigsegments_notsorted _SU.csv')
#df['ensemble_score']=ensemble_score
#df['vilbert_score']=vilbert['scores']
#df['text_score']=text['results_st']
#df['picsom_score']=picsom['scores']
#df.to_csv('captions_clean.csv')
df=df.sort_values(by='ensemble_score', ascending=False)
df.to_csv('Surrey_bigsegments_sorted_SU.csv')
print(df)
