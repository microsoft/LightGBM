library(stringi)

lightgbm.train <- function(
  y_train=NA,
  x_train=NA,
  y_val=NA,
  x_val=NA,
  application = 'regression', #regression, binary, lambdarank
  validation=TRUE,
  num_iterations = 10,
  learning_rate = 0.1,
  num_leaves = 127,
  tree_learner = 'serial', #serial, feature, data
  num_threads = 5,
  min_data_in_leaf = 100,
  min_sum_hessian_in_leaf = 10,
  feature_fraction = 1,
  feature_fraction_seed = 2,
  bagging_fraction = 1,
  bagging_freq = 0,
  bagging_seed = 3,
  max_bin = 255,
  data_random_seed = 1,
  data_has_label = 'true',
  output_model = 'LightGBM_model.txt',
  input_model = 'LightGBM_model.txt',
  output_result = 'LightGBM_predict_result.txt',
  is_sigmoid = 'true',
  init_score = '',
  is_pre_partition = 'false',
  is_sparse = 'true',
  two_round = 'false',
  save_binary = 'false',
  sigmoid = 1,
  is_unbalance = 'false',
  max_position = 20,
  label_gain = '0,1,3,7,15,31,63',
  metric = 'l2', #l1,l2,ndcg,auc,binary_logloss,binary_error
  metric_freq = 1,
  is_training_metric = 'false',
  ndcg_at = '1,2,3,4,5',
  num_machines = 1,
  local_listen_port = 12400,
  time_out = 120,
  machine_list_file = '',
  gbmpath = '/home/dba/KAGGLE/LightGBM',
  workingdir = '',
  files_exist = FALSE
) {
  
  if(!file.exists(paste0(gbmpath,'/lightgbm'))){
    return('lightgbm.exe file does not exist... exiting!')
  }
  print(paste('lightGBM path:','/home/dba/KAGGLE/LightGBM'))
  if (workingdir == '') {
    workingdir = stri_rand_strings(1,20)
  }
  dir.create(file.path(gbmpath, workingdir), showWarnings = FALSE)
  print(paste('working directory:',file.path(gbmpath, workingdir)))
  file.copy(paste0(gbmpath,'/lightgbm'), file.path(gbmpath, workingdir))
  fileConn<-file(file.path(gbmpath, workingdir,"train.conf"),"w")
  write(paste0('task=train'), fileConn, append=T)
  write(paste0('application=',application), fileConn, append=T)
  write(paste0('data="',file.path(gbmpath, workingdir,'train.csv"')), fileConn, append=T)
  if (validation) write(paste0('valid="',file.path(gbmpath, workingdir,'val.csv"')), fileConn, append=T)
  write(paste0('num_iterations=',num_iterations), fileConn, append=T)
  write(paste0('learning_rate=',learning_rate), fileConn, append=T)
  write(paste0('num_leaves=',num_leaves), fileConn, append=T)
  write(paste0('tree_learner=',tree_learner), fileConn, append=T)  
  write(paste0('num_threads=',num_threads), fileConn, append=T)  
  write(paste0('min_data_in_leaf=',min_data_in_leaf), fileConn, append=T)  
  write(paste0('min_sum_hessian_in_leaf=',min_sum_hessian_in_leaf), fileConn, append=T)  
  write(paste0('feature_fraction=',feature_fraction), fileConn, append=T)  
  write(paste0('feature_fraction_seed=',feature_fraction_seed), fileConn, append=T)  
  write(paste0('bagging_fraction=',bagging_fraction), fileConn, append=T)  
  write(paste0('bagging_freq=',bagging_freq), fileConn, append=T)  
  write(paste0('bagging_seed=',bagging_seed), fileConn, append=T)  
  write(paste0('max_bin=',max_bin), fileConn, append=T)  
  write(paste0('data_random_seed=',data_random_seed), fileConn, append=T)  
  write(paste0('data_has_label=',data_has_label), fileConn, append=T)  
  if (output_model!='') write(paste0('output_model="',file.path(gbmpath, workingdir,output_model),'"'), fileConn, append=T)  
  write(paste0('is_sigmoid=',is_sigmoid), fileConn, append=T)    
  if (init_score!='') write(paste0('init_score="',file.path(gbmpath, workingdir,init_score),'"'), fileConn, append=T)   
  write(paste0('is_pre_partition=',is_pre_partition), fileConn, append=T)   
  write(paste0('is_sparse=',is_sparse), fileConn, append=T)   
  write(paste0('two_round=',two_round), fileConn, append=T)   
  write(paste0('save_binary=',save_binary), fileConn, append=T)   
  write(paste0('sigmoid=',sigmoid), fileConn, append=T)   
  write(paste0('is_unbalance=',is_unbalance), fileConn, append=T)   
  write(paste0('max_position=',max_position), fileConn, append=T)   
  write(paste0('label_gain=',label_gain), fileConn, append=T)   
  write(paste0('metric=',metric), fileConn, append=T)   
  write(paste0('metric_freq=',metric_freq), fileConn, append=T)    
  write(paste0('is_training_metric=',is_training_metric), fileConn, append=T)   
  write(paste0('ndcg_at=',ndcg_at), fileConn, append=T)   
  write(paste0('num_machines=',num_machines), fileConn, append=T)   
  write(paste0('local_listen_port=',local_listen_port), fileConn, append=T)   
  write(paste0('time_out=',time_out), fileConn, append=T)   
  if (machine_list_file!='') write(paste0('machine_list_file="',file.path(gbmpath, workingdir,machine_list_file),'"'), fileConn, append=T)     
  close(fileConn)
  print(paste('config file saved to:',file.path(gbmpath, workingdir,"train.conf")))  
  if (!files_exist) {
    print(paste('saving train data file to:',file.path(gbmpath, workingdir,"train.csv")))  
    write.table(cbind(y_train,x_train),file.path(gbmpath, workingdir,"train.csv"),row.names=F,col.names=F,sep=',',na = "nan")
  }
  if (validation) {
    print(paste('saving validation data file to:',file.path(gbmpath, workingdir,'val.csv')))    
    if (!files_exist) {
      write.table(cbind(y_val,x_val),file.path(gbmpath, workingdir,"val.csv"),row.names=F,col.names=F,sep=',',na = "nan")
    }
  }
  system(paste0(file.path(gbmpath, workingdir),'/lightgbm config=',file.path(gbmpath, workingdir),'/train.conf'))   
  print(paste('model complete, results saved in ',file.path(gbmpath, workingdir)))    
  return(workingdir)    
  
}

lightgbm.predict <- function(
  model,
  y_val=NA,
  x_val=NA,
  data_has_label = 'true',
  input_model = 'LightGBM_model.txt',
  output_result = 'LightGBM_predict_result.txt',
  gbmpath = '/home/dba/KAGGLE/LightGBM',
  newx = TRUE
) {
  
  if(!file.exists(paste0(gbmpath,'/lightgbm'))){
    return('lightgbm.exe file does not exist... exiting!')
  }
  if (newx){
    print(paste('saving validation data file to:',file.path(gbmpath, model,'test.csv')))    
    write.table(cbind(y_val,x_val),file.path(gbmpath, model,"test.csv"),row.names=F,col.names=F,sep=',')
    fileConn<-file(file.path(gbmpath, model,"pred.conf"),"w")
    write(paste0('task=prediction'), fileConn, append=T)  
    write(paste0('data="',file.path(gbmpath, model,'test.csv"')), fileConn, append=T)    
    if (input_model!='') write(paste0('input_model="',file.path(gbmpath, model,input_model),'"'), fileConn, append=T)  
    if (output_result!='') write(paste0('output_result="',file.path(gbmpath, model,output_result),'"'), fileConn, append=T)      
    write(paste0('data_has_label=',data_has_label), fileConn, append=T)     
    close(fileConn) 
    system(paste0(file.path(gbmpath, model),'/lightgbm config=',file.path(gbmpath, model),'/pred.conf'))
    return(read.csv(file.path(gbmpath, model,output_result),header=F))
  }
  if (!newx){
    fileConn<-file(file.path(gbmpath, model,"pred.conf"),"w")
    write(paste0('task=prediction'), fileConn, append=T)  
    write(paste0('data="',file.path(gbmpath, model,'val.csv"')), fileConn, append=T)    
    if (input_model!='') write(paste0('input_model="',file.path(gbmpath, model,input_model),'"'), fileConn, append=T)  
    if (output_result!='') write(paste0('output_result="',file.path(gbmpath, model,output_result),'"'), fileConn, append=T)      
    write(paste0('data_has_label=',data_has_label), fileConn, append=T)     
    close(fileConn) 
    system(paste0(file.path(gbmpath, model),'/lightgbm config=',file.path(gbmpath, model),'/pred.conf'))
    return(read.csv(file.path(gbmpath, model,output_result),header=F))
  }  
}


lightgbm.cv <- function(
  y_train,
  x_train,
  idx,
  application = 'regression', #regression, binary, lambdarank
  validation=TRUE,
  num_iterations = 10,
  learning_rate = 0.1,
  num_leaves = 127,
  tree_learner = 'serial', #serial, feature, data
  num_threads = 5,
  min_data_in_leaf = 100,
  min_sum_hessian_in_leaf = 10,
  feature_fraction = 1,
  feature_fraction_seed = 2,
  bagging_fraction = 1,
  bagging_freq = 0,
  bagging_seed = 3,
  max_bin = 255,
  data_random_seed = 1,
  data_has_label = 'true',
  output_model = 'LightGBM_model.txt',
  input_model = 'LightGBM_model.txt',
  output_result = 'LightGBM_predict_result.txt',
  is_sigmoid = 'true',
  init_score = '',
  is_pre_partition = 'false',
  is_sparse = 'true',
  two_round = 'false',
  save_binary = 'false',
  sigmoid = 1,
  is_unbalance = 'false',
  max_position = 20,
  label_gain = '0,1,3,7,15,31,63',
  metric = 'l2', #l1,l2,ndcg,auc,binary_logloss,binary_error
  metric_freq = 1,
  is_training_metric = 'false',
  ndcg_at = '1,2,3,4,5',
  num_machines = 1,
  local_listen_port = 12400,
  time_out = 120,
  machine_list_file = '',
  gbmpath = '/home/dba/KAGGLE/LightGBM',
  workingdir = '',
  prediction=TRUE
) {
  models=list()
  idx_list=unique(idx)
  for (i in 1:length(idx_list)) {
    print('************')
    print(paste('fold no:',i))
    print('************')    
    models[[i]]=
      lightgbm.train(
        x_train=x_train[idx!=i,],
        y_train=y_train[idx!=i],
        x_val=x_train[idx==i,],
        y_val=y_train[idx==i],
        application,validation,num_iterations,learning_rate,num_leaves,tree_learner,
        num_threads,min_data_in_leaf,min_sum_hessian_in_leaf,feature_fraction,
        feature_fraction_seed,bagging_fraction,bagging_freq,bagging_seed,max_bin,
        data_random_seed,data_has_label,output_model,input_model,output_result,
        is_sigmoid,init_score,is_pre_partition,is_sparse,two_round,save_binary,
        sigmoid,is_unbalance,max_position,label_gain,metric,metric_freq,
        is_training_metric,ndcg_at,num_machines,local_listen_port,time_out,
        machine_list_file,gbmpath,workingdir
      )
  }
  if (!prediction) { return(models) }
  if(prediction) {
    return(lightgbm.cv.predict(models))
  }
}



lightgbm.cv.predict <- function(
  models,
  data_has_label = 'true',
  input_model = 'LightGBM_model.txt',
  output_result = 'LightGBM_predict_result.txt',
  gbmpath = '/home/dba/KAGGLE/LightGBM'
) {
  preds=list()
  for (i in 1:length(models)) {
    dat = read.csv(file.path(gbmpath, models[[i]],'val.csv'))
    preds[[i]]=
      lightgbm.predict(
        models[[i]],
        y_val=dat[,1],
        x_val=dat[,-1],
        data_has_label = 'true',
        input_model = 'LightGBM_model.txt',
        output_result = 'LightGBM_predict_result.txt',
        gbmpath = '/home/dba/KAGGLE/LightGBM',
        newx = FALSE
      )    
    
  }
  return(preds)
}




