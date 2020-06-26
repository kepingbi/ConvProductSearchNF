# ConvProductSearchNF
This is code for paper "[Conversational Product Search Based on Negative Feedback](https://arxiv.org/abs/1909.02071)" [1]
- We adopt the same dataset and partition as [HEM](https://github.com/QingyaoAi/Amazon-Product-Search-Datasets)[2]
- Aspects and values are extracted with toolkit by [3]. 
- Extracted aspect-value lexicons are included in directory "extracted_av_lexicon". Movies&TV is too large for the tool to process, so we divide it into several parts and extract the lexicon for each part individually. 
The 2nd column in the lexicon file is "aspect|value". The rest columns are not used in our experiments.
## Data
Please find the data in the paper [here](https://drive.google.com/drive/folders/1ARBX_bIad-eZkwWm3EZ3D13oYCyEhMJz?usp=sharing)
Download the data to your machine and replace `PATH/TO/DATA` in the parameters input to the script with the path where you store the downloaded data. 
Note that there are only three extra files compared with [HEM](https://github.com/QingyaoAi/Amazon-Product-Search-Datasets), which are `lexicon.strict.stemmed.id.sorted.txt`, `av.train.strict.txt.gz`, and `av.test.strict.txt.gz`. 
```
Extra Files: 
1. lexicon.strict.stemmed.id.sorted.txt
    Lexicon extracted from the review data using the tool provided by Zhang et al. [3]. The words in aspects and values are mapped to ids which share the same vocabulary with the words in the product reviews. 
    File Format:
      Aspect|ValueCount|Frequency:Value|ValueFrequency, Value|ValueFrequency, ...
      Aspect: indices of the words in the aspect (can be more than one)
      ValueCount: number of unique values associated with the aspect
      Frequency: the number of times the asepct appear in the data (user reviews)
      Value: indices of the words in the value (can be more than one, but only one will be kept)
      ValueFrequency: the frequency of this value occurring with the aspect. 
    
2. av.train.strict.txt.gz/av.test.strict.txt.gz
    Reviews that contain the extracted aspect values. They are obtained by applying the lexicon to the user-product reviews to get the aspect-values corresponding to each review. Not all the reviews appear in the files since they may not contain any aspect-values. 
    File Format:
      UserId@ProductId, Aspect|Value|Sentiment, Aspect|Value|Sentiment, ...
      UserId: user index
      ProductId: product index
      Aspect: indices of the words in the aspect
      Value: indices of the words in the value
      Sentiment: the sentiment (positive or negative) associated with the aspect-value pair. Not used in the model.     
```

## Train a AVHEM model (AVLEM in the paper[1])
```
python main.py --cuda --data_dir PATH/TO/DATA \
                --input_train_dir PATH/TO/DATA/query_split \
                --save_dir PATH/TO/WHERE/YOU/WANT/TO/SAVE/YOUR/MODEL \
                --qnet_struct fs --similarity_func product \
                --model_net_struct AVHEM \
                --batch_size 64 --subsampling_rate 1e-5 \
                --sparse_emb \ # only update the parameters that change in the batch during back propagation; this does not always result in a higher speed
                --init_learning_rate 0.5 \
                --value_loss_func sep_emb \
```
Note that "--sparse_emb" is useful for Movies&TV but not useful for CellPhones&Accessories and Health&PersonalCare

## Test a AVHEM model (AVLEM in the paper[1])
Use the same setting for training and append "--decode --test_mode product_scores|iter_result"
```
python main.py --cuda --data_dir PATH/TO/DATA \
                --input_train_dir PATH/TO/DATA/query_split \
                --save_dir PATH/TO/WHERE/YOU/SAVE/YOUR/MODEL \
                --qnet_struct fs --similarity_func product \
                --model_net_struct AVHEM \
                --batch_size 64 --subsampling_rate 1e-5 \
                --sparse_emb \ # only update the parameters that change in the batch during back propagation
                --init_learning_rate 0.5 \
                --value_loss_func sep_emb \
                --decode \ # test the model
                --test_mode product_scores  #"product_scores" corresponds to AVLEM_{init} where no feedback is used #--test_mode iter_result # "iter_result" correponds to iterative ranking with fine-grained feedback
```
## Train a HEM [2] model
```
python main.py --cuda --data_dir PATH/TO/DATA \
                --input_train_dir PATH/TO/DATA/query_split \
                --save_dir PATH/TO/WHERE/YOU/WANT/TO/SAVE/YOUR/MODEL \
                --qnet_struct fs --similarity_func product \
                --model_net_struct HEM \
                --batch_size 64 --subsampling_rate 1e-5 \
                --init_learning_rate 0.5 \
```
## Test a HEM [2] model
Use the same setting for training and append "--decode --test_mode product_scores", e.g.,
```
python main.py --cuda --data_dir PATH/TO/DATA \
                --input_train_dir PATH/TO/DATA/query_split \
                --save_dir PATH/TO/WHERE/YOU/WANT/TO/SAVE/YOUR/MODEL \
                --qnet_struct fs --similarity_func product \
                --model_net_struct HEM \
                --batch_size 64 --subsampling_rate 1e-5 \
                --init_learning_rate 0.5 \
                --decode \ # test the model
                --test_mode product_scores
```
## References
* [1] Keping Bi, Qingyao Ai, Yongfeng Zhang and W. Bruce Croft. Conversational Product Search Based on Negative Feedback. Accepted in Proceedings of CIKM'19
* [2] Qingyao Ai, Yongfeng Zhang, Keping Bi, Xu Chen, W. Bruce Croft. 2017. Learning a Hierarchical Embedding Model for Personalized ProductSearch. In Proceedings of SIGIR ’17
* [3] Yongfeng Zhang, Guokun Lai, Min Zhang, Yi Zhang, Yiqun Liu, and Shaoping Ma. 2014. Explicit factor models for explainable recommendation based on phrase-level sentiment analysis. In Proceedings of SIGIR’14.
