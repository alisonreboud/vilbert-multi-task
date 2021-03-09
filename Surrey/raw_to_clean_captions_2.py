import pandas as pd
import re

def read_alto_captions(path):
    df = pd.read_csv(path,sep=':t',header=None,names=['mess','caption'])
    #print(df.caption.dtype)
    df[['trash1','caption']] = df.caption.str.split(n=1, expand=True)
    df[['video_id','caption']] = df.caption.str.split(n=1, expand=True)
    df[['start','end']]=df.mess.str.split(n=1,expand=True)
    #df['video']=df.video_name.astype(str).str.slice(stop=3)
    #df[['trash1','name']] = df.mess.str.split(":t",expand=True,)
    #df[['trash2','trash3']] =df.name.str.split("a ",expand=True,)
    #df['video_id']=df.trash3.astype(str).str.slice(start=2)
    #df['start_frame'] = df.video_name.astype(str).str.slice(start=3,stop=8)
    #df['end_frame'] = df.video_name.astype(str).str.slice(start=8, stop=-1)
    df[['video','frame']]=df.video_id.str.split(":",expand=True,)
    df.video_id=df.video_id.str.replace(':','_')
    print(len(df.video.value_counts()))
    #print(df.trash3)

    return df[['caption','video_id','video','frame','start','end']]

df =read_alto_captions('captions_raw.txt')
print(df)
df.to_csv('Surrey_captions_clean.csv')

#print(df['frame'][df['video']=='000200'].tolist())


