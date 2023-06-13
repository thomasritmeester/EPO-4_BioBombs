#CrossFold Test
import pyarrow.parquet as pq
from CrossFold import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
all_data_pq = pq.read_table('all_data_df.parquet')
all_data_df= all_data_pq.to_pandas()

# print(all_data_df.loc[4,:])

subjects=["P1","P2","P5","P6","P9","P10","P11","P12"]
lda=LDA(n_components=1)

accuracy_score, f1_score, mcc_score=CrossFold(subject=subjects, model=lda, all_data_df=all_data_df, data = 'False')
print("accuracy_score=", accuracy_score, '\n')
print("f1_score=", accuracy_score, '\n')
print("mcc_score", accuracy_score, '\n')