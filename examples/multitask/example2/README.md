# kaggle-ieee solution
https://github.com/white-bird/kaggle-ieee

if you wanna only run the best model, you should run:  
f3deepwalk.py, fdkey.py,  
feV307.py, fiyu.py,  
model26.ipynb,  
model32.ipynb,  
https://www.kaggle.com/whitebird/ieee-internal-blend  
  
LB 9500-9520 : I spent most of my time at here while I try to dig the count/mean/std features which didn't work.  
  
LB 9520-9580 : I realized the bad guys stole the cards and make transactions for money, but cards always have some protects, like the biggest amount for one transaction. So they need to have many similar transactions on one card in a lone period or many cards in a short time. That's the keypoint of this competition ----- the series samples make it fraud, not single sample.  
We need to find some "keys" to group the data:  
  
1) V307. There are too many V features. Some are int and some are floats. It's not hard to find out that int means the times this card have transactions with same website/seller, and float means the accumulated amount. Obviously, int + cardid may casue misjudge easily. If you have some baseline models, I recommend you the lib eli5 to find which feature is most important, which leads me to the V307. You can find these eda at model14.ipynb. I use the fe_V307.py to process the feature.  
  
2).deviceinfo & id. Different cards have same amt in same addr with same device. is it strange? So I use fd_key.py to extract them.  
  
3). cardid + D. My teammates found this. All people knows the D features minus days mean a lot. We find the D2 and D15 run through the time best by max all the data, while D2 and D15 has the biggest value. fi_yu.py  
  
4). amt + days + addr1. It is simple but easy to misjudge.  
  
LB 9590-9600: So we all know the fraud sample is fraud because its similar samples is fraud. Why not let the infect of fraud more crazy? Making a two-stage models improve 0.001:model26.ipynb + model32.ipynb    
  
LB 9600-9630: This is caused by a bug. I grouped the keys above and get big improvments offline. However, there is only one key, cardid + D, behave badly online. I used 2~3 days to find out that I grouped them with train and test separately. It make improvments online when I grouped the key with all data. It means the key is not working as other keys to make group features but as a embedding key. Then I wrote some rules to process results with kernels. It's easy to understand but make huge boost:https://www.kaggle.com/whitebird/ieee-internal-blend?scriptVersionId=21198581  
  
And there are other small improvments I don't mention. Post here if you have any question while reading/running my code.  
