import os
import pandas as pd
import pickle
import numpy as np
from utils import preprocess,metrics,models
import keras


def GoF(data:list,hparams:dict,model_path:str):
        """
        Trains a classifier to discriminate synthetic from real samples. 
        
        data: List of data, ordered from real to synthetic and train to test.
            Each data object within the list is a list of a 2D and (padded) 3D numpy array.
        hparams: Dictionary of hyperparameters (epochs, batch size, dropout rate, hidden units per layer,
            activation function in hidden layers).
        model_path: Path to which we save trained model for checkpointing / early stopping.
        returns: Classification accuracy, p-value of KS test on real/synthetic classification scores, 
            kernel density estimation plot of classifier scores.

        """
        X_real_tr,X_real_te,X_syn_tr,X_syn_te = data
        #concatenate real and synthetic data to pooled dataframes/arrays
        X_train = [np.concatenate((X_real_tr[0],X_syn_tr[0]),axis=0),\
                np.concatenate((X_real_tr[1],X_syn_tr[1]),axis=0)]
        X_test = [np.concatenate((X_real_te[0],X_syn_te[0]),axis=0),\
                np.concatenate((X_real_te[1],X_syn_te[1]),axis=0)]
        #create binary labels
        y_train = np.concatenate((np.zeros(X_real_tr[0].shape[0]),np.ones(X_syn_tr[0].shape[0])),axis=0)
        y_test = np.concatenate((np.zeros(X_real_te[0].shape[0]),np.ones(X_syn_te[0].shape[0])),axis=0)

        #set up model checkpointing for early stopping
        checkpoint_filepath = model_path + '/GoF.keras'
        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)
        #instantiate and fit model
        config = {'input_shape_attr':(X_train[0].shape[1],),
                  'input_shape_feat':(X_train[1].shape[1],X_train[1].shape[2],),
                  'hidden_units':hparams['HIDDEN_UNITS'],
                  'dropout_rate':hparams['DROPOUT_RATE'],
                  'activation':hparams['ACTIVATION']
                  }
        model = models.test_gof(config=config)
        model.compile(optimizer='Adam',loss='binary_crossentropy',metrics='accuracy')
        model.fit(X_train,y_train,batch_size=hparams['BATCH_SIZE'],epochs=hparams['EPOCHS'],
                  validation_split=.2,callbacks=[model_checkpoint_callback])
        #load best model and make predictions
        model.load_weights(checkpoint_filepath)
        pred = model.predict(X_test)
        #perform KS test on classifier scores
        test_stat,pval = metrics.ks_test(real_pred=pred[y_test==0],syn_pred=pred[y_test==1])
        accuracy = metrics.accuracy(y_test,np.round(pred))
        #kde plot of classifier scores
        GoF_kdeplot = metrics.GoF_kdeplot(pred=pred,y_test=y_test)
        GoF_kdeplot.title('Distribution of classification scores')

        return accuracy,pval,GoF_kdeplot

def mortality_prediction(data:list,attr:list,hparams:dict,model_path:str,pred_model='RNN'):
    """
    Trains a machine learning model to predict patient mortalitya according to TSTR approach.

    data: List of data, ordered from real to synthetic and train to test.
            Each data object within the list is a list of a 2D and (padded) 3D numpy array.
    attr: List of static data column names, to access index of target feature.
    hparams: Dictionary of hyperparameters. depends on prediction model used.
    model_path: Path to which we save trained model for checkpointing / early stopping.
    pred_model: The prediction model used. Options are RNN, Random Forest, Logistic Regression.
    returns: Accuracy and AUC of real and synthetically trained mortality prediction model.
    """

    X_real_tr,X_real_te,X_syn_tr,X_syn_te = data
    assert pred_model  in ['RNN','LR','RF']
    #get input/output data
    target = 'deceased'
    target = attr.get_loc(target)
    x = []
    y = []
    for data in [X_real_tr,X_real_te,X_syn_tr,X_syn_te]:
        y.append(data[0][:,target])
        x0 = np.delete(data[0],target,axis=1)   
        x.append([x0,data[1]])
    x_real_tr,x_real_te,x_syn_tr,x_syn_te = x
    y_real_tr,y_real_te,y_syn_tr,y_syn_te = y
    
    if pred_model=='RNN':
        model_list = []
        for data,name in zip([[x_real_tr,y_real_tr],[x_syn_tr,y_syn_tr]],['real','syn']):
            X_tr,y_tr = data 
            #setup model checkpointing
            checkpoint_filepath=model_path + f'/mortality_{pred_model}_{name}.keras'
            model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_auc',
            mode='max',
            save_best_only=True)
            config = {'input_shape_attr':(X_tr[0].shape[1],),
                  'input_shape_feat':(X_tr[1].shape[1],X_tr[1].shape[2],),
                  'hidden_units':hparams['HIDDEN_UNITS'],
                  'dropout_rate':hparams['DROPOUT_RATE'],
                  'activation':hparams['ACTIVATION']
                  }
            model = models.test_mortality(config=config)
            #model = models.mortality_RNN_simple(config=config)
            auc = keras.metrics.AUC(name='auc')
            model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=auc)
            model.fit(X_tr,y_tr,batch_size=hparams['BATCH_SIZE'],epochs=hparams['EPOCHS'],
                      validation_split=.2,callbacks=[model_checkpoint_callback])
            #choose the best version and load its weights for real and synthetic model
            model.load_weights(checkpoint_filepath)
            model_list.append(model)
        real_model,syn_model = model_list
    else:
        x = []
        for data in [x_real_tr,x_real_te,x_syn_tr,x_syn_te]:
             #create count features for dynamic data in static models RF and LR
             x0,x1 = data 
             x1 = np.sum(x1,axis=1)
             x.append(np.concatenate((x0,x1),axis=1))
        x_real_tr,x_real_te,x_syn_tr,x_syn_te = x
        model_list = []
        for data,name in zip([[x_real_tr,y_real_tr],[x_syn_tr,y_syn_tr]],['real','syn']):
            X_tr,y_tr = data
            checkpoint_filepath=model_path + f'/mortality_{pred_model}_{name}.pkl'
            if pred_model=='LR':
                model = models.mortality_LR(penalty='elasticnet',l1_ratio=hparams['L1'])
            else:
                model = models.mortality_RF(n_estimators=hparams['N_TREES'],max_depth=hparams['MAX_DEPTH'])
            model.fit(X_tr,y_tr.flatten())
            with open(checkpoint_filepath,'wb') as f:
                pickle.dump(model,f)
            model_list.append(model)
        real_model,syn_model = model_list
    #make predictions
    real_preds = real_model.predict(x_real_te)
    syn_preds = syn_model.predict(x_real_te)
    real_acc = metrics.accuracy(y_real_te,np.round(real_preds))
    syn_acc = metrics.accuracy(y_real_te,np.round(syn_preds))
    real_auc = metrics.auc(y_real_te,real_preds)
    syn_auc = metrics.auc(y_real_te,syn_preds)
    return real_acc,real_auc,syn_acc,syn_auc


def trajectory_prediction(data:list,hparams:dict,model_path:str):
    """
    Trains an RNN to predict next-step diagnoses according to TSTR approach.

    data: List of data, ordered from real to synthetic and train to test.
            Each data object within the list is a list of a 2D and (padded) 3D numpy array.
    hparams: Dictionary of hyperparameters (epochs, batch size, dropout rate, hidden units per layer,
        activation function in hidden layers).
    model_path: Path to which we save trained model for checkpointing / early stopping.
    returns: Accuracy of real and synthetically trained model.

    """

    X_real_tr,X_real_te,X_syn_tr,X_syn_te = data
    #get input/target pairs for all datasets
    x = []
    y = []
    for data in [X_real_tr,X_real_te,X_syn_tr,X_syn_te]:
        x_,y_ = preprocess.trajectory_input_output(data,max_t=X_real_tr[1].shape[1])
        x.append(x_)
        y.append(y_)
    x_real_tr,x_real_te,x_syn_tr,x_syn_te = x
    y_real_tr,y_real_te,y_syn_tr,y_syn_te = y
    #fit the synthetic and real model
    model_list = []
    for data,name in zip([[x_real_tr,y_real_tr],[x_syn_tr,y_syn_tr]],['real','syn']):
        X_tr,y_tr = data
        #setup model checkpointing
        checkpoint_filepath=model_path + f'/trajectory_{name}.keras'
        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)
        config = {'input_shape_attr':(X_tr[0].shape[1],),
                'input_shape_feat':(X_tr[1].shape[1],X_tr[1].shape[2],),
                'hidden_units':hparams['HIDDEN_UNITS'],
                'dropout_rate':hparams['DROPOUT_RATE'],
                'activation':hparams['ACTIVATION'],
                'output_units':y_tr.shape[1]
                }
        model = models.trajectory_RNN(config=config)
        model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics='accuracy')
        model.fit(X_tr,y_tr,batch_size=hparams['BATCH_SIZE'],epochs=hparams['EPOCHS'],
                    validation_split=.2,callbacks=[model_checkpoint_callback])
        model.load_weights(checkpoint_filepath)
        model_list.append(model)
    real_model,syn_model = model_list
    #make predictions
    real_preds = real_model.predict(x_real_te)
    syn_preds = syn_model.predict(x_real_te)
    #evaluate results
    labels = np.argmax(y_real_te,axis=1)
    real_preds_labels = np.argmax(real_preds,axis=1)
    syn_preds_labels = np.argmax(syn_preds,axis=1)
    real_acc = metrics.accuracy(labels,real_preds_labels)
    syn_acc = metrics.accuracy(labels,syn_preds_labels)
    return real_acc,syn_acc


def privacy_AIA(data:list,attr:list,scaler:preprocess.Scaler,hparams:dict,model_path:str,sensitive_attr=list):
    """
    Trains an RNN on synthetic data to infer real sensitive attributes. Loops over all provided target sets.

    data: List of data, ordered from real to synthetic and train to test.
            Each data object within the list is a list of a 2D and (padded) 3D numpy array.
    attr: List of static data column names, to access index of target feature.
    scaler: Fitted numerical feature scaler to reverse transform predictions, for interpretable AIA 
        accuracy metrics (i.e. MAE).
    hparams: Dictionary of hyperparameters (epochs, batch size, dropout rate, hidden units per layer,
        activation function in hidden layers).
    model_path: Path to which we save trained model for checkpointing / early stopping.
    sensitive_attr: List of lists of sensitive target attribute sets
    returns: AIA accuracy metrics
    """
    X_real_tr,X_real_te,X_syn_tr,X_syn_te = data

    #instantiate lists of outputs
    mape_age = []
    mae_age = []
    acc_gender = []
    acc_race = []
    #perform AIA for every label combination
    for j,labels in enumerate(sensitive_attr):
        #ensure we are taking the one hot encoded columns
        if 'race' in labels:
            labels.remove('race')
            labels.extend([x for x in attr if 'race' in x]) 
        #find target indices to select 
        targets = attr.get_indexer(labels)
        y = []
        x = []
        for data in [X_real_tr,X_real_te,X_syn_tr,X_syn_te]:
            #find target data
            y.append(data[0][:,targets])
            #find input data
            x0 = np.delete(data[0],targets,axis=1)
            x.append([x0,data[1]]) 
        x_real_tr,x_real_te,x_syn_tr,x_syn_te = x 
        y_real_tr,y_real_te,y_syn_tr,y_syn_te = y 
        config = {'input_shape_attr':(x_real_tr[0].shape[1],),
                  'input_shape_feat':(x_real_tr[1].shape[1],x_real_tr[1].shape[2],),
                  #first layer is separate processing, afterwards joint Dense layers
                  'hidden_units':hparams['HIDDEN_UNITS'],
                  'dropout_rate':hparams['DROPOUT_RATE'],
                  'activation':hparams['ACTIVATION']
                  }
        #build the model, which takes account of input labels and builds output layer accordingly
        model = models.privacy_RNN(labels=labels,config=config)
        #specify the loss functions and metrics we require
        #note that the output layer names are output_1 , output_2, output_3 and are dynamic not fixed
        #changing this is TBD (however does not seem possible due to dynamic structure of the model)
        losses = {}
        metric = {}
        var_count = 1
        if 'age' in labels:
            key = 'output_'+str(var_count)
            losses[key] = 'mse'
            metric[key] = keras.metrics.MeanSquaredError()
            var_count+=1
        if 'gender' in labels:
            key = 'output_'+str(var_count)
            losses[key] = 'binary_crossentropy'
            metric[key] = keras.metrics.Accuracy()
            var_count+=1
        if sum(l.count('race') for l in labels)>0:
            key = 'output_'+str(var_count)
            losses[key] = 'categorical_crossentropy'
            metric[key] = keras.metrics.CategoricalAccuracy()

        model.compile(optimizer='Adam',loss=losses,metrics=metric)

        #keras model expects a list of the different outputs instead of a concatenated array
        y_list = []
        for d in [y_real_tr,y_real_te,y_syn_tr,y_syn_te]:
            output_list = []
            for label in [x for x in labels if 'race' not in x]:
                t = labels.index(label)
                output_list.append(d[:,t])
            if sum(x.count('race') for x in labels)>0:
                t=[]
                for l in [x for x in labels if 'race' in x]:
                    t.append([i for i,x in enumerate(labels) if x==l])
                output_list.append(d[:,t])
            y_list.append(output_list)
        y_real_tr,y_real_te,y_syn_tr,y_syn_te = y_list
        

        #we train the model on synthetic data and assess on real test set
        checkpoint_filepath = model_path+f'/privacy_AIA_label{j}.keras'
        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True)
        model.fit(x_syn_tr,y_syn_tr,epochs=hparams['EPOCHS'],batch_size=hparams['BATCH_SIZE'],
                  validation_split=.2,callbacks=[model_checkpoint_callback])
        model.load_weights(checkpoint_filepath)
        preds = model.predict(x_real_te)

        #evaluate predictions
        var_count = 0
        if not isinstance(preds,list):
            preds = [preds]
        if 'age' in labels:
            #rescale age to original range
            y_real_te[var_count] = scaler.reverse_transform(y_real_te[var_count])
            preds[var_count] = scaler.reverse_transform(preds[var_count])

            #compute metrics
            mape_age.append(metrics.mape(y_real_te[var_count],preds[var_count]))
            mae_age.append(metrics.mae(y_real_te[var_count],preds[var_count]))
            var_count+=1
        if 'gender' in labels:
            acc_gender.append(metrics.accuracy(y_real_te[var_count],np.round(preds[var_count])))
            var_count+=1
        if sum(x.count('race') for x in labels)>0:
            acc_race.append(metrics.accuracy(np.squeeze(np.concatenate(y_real_te[var_count:],axis=1),-1),
                                    np.concatenate(np.round(preds[var_count:]),axis=1)))
            
    return mape_age,mae_age,acc_gender,acc_race
            
    
if __name__=='__main__':  
    #set up paths for data loading, result saving, and model checkpointing
    path = 'C:/Users/jlachterberg/Documents/data'
    syn_model = 'dgan'
    result_path = os.path.join('results',syn_model)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    model_path = os.path.join('model',syn_model)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    #------------------------------------------------------------------------------------------
    #load data
    cols = ['subject_id','seq_num','icd_code','gender','age','deceased','race']
    real_df = pd.read_csv(path+'/real.csv.gz',sep=',',compression='gzip',usecols=cols)
    syn_df = pd.read_csv(path+'/cpar.csv.gz',sep=',',compression='gzip',usecols=cols)

    #select only part of data for quick code testing
    n = 100
    real_df = real_df[real_df.subject_id.isin(np.random.choice(real_df.subject_id.unique(),size=n))]
    syn_df = syn_df[syn_df.subject_id.isin(np.random.choice(syn_df.subject_id.unique(),size=n))]

    #------------------------------------------------------------------------------------------------------
    #preprocess data to list of static (pandas dataframe) and sequential (padded 3D numpy array) data
    real,syn = preprocess.preprocess_(real_df,syn_df)
    attr = real[0].columns #necessary to find target column later

    #make k test splits stratified on mortality
    test_split_real,test_split_syn = preprocess.k_test_splits(real,syn,k=10)    

    #setup report files for later appending of results
    for file in ['gof_test_report.txt','mortality_pred_accuracy.txt','trajectory_pred_accuracy.txt','privacy_AIA.txt']:
        with open(os.path.join(result_path,file),'w') as f:
            pass

    #select full data indices to select training indices from test indices in each test split
    idx_real = np.arange(0,real[0].shape[0]) 
    idx_syn = np.arange(0,syn[0].shape[0])

    #set which numerical features we need to scale
    num_features = ['age']

    #loop over k test splits and execute the analyses for each split
    for s,(te_real,te_syn) in enumerate(zip(test_split_real,test_split_syn)):
        #find training indices for this test split
        tr_real = np.setdiff1d(idx_real,te_real).tolist()
        tr_syn = np.setdiff1d(idx_syn,te_syn).tolist()
        
        #save a fitted numerical scaler instance to reverse transform predictions later
        scaler = preprocess.Scaler()
        scaler.transform(real[0].loc[te_real][num_features]) #save for later reverse transformation on test data

        #select relevant train and test set from indices and scale numerical features independently
        data_ = preprocess.scale(real,syn,tr_real,te_real,tr_syn,te_syn,feature=num_features)
        
        #------------------------------------------------------------------------------------------
        #GoF
        # nn_params = {'EPOCHS':10,
        #        'BATCH_SIZE':16,
        #        'HIDDEN_UNITS':[10,10],
        #        'ACTIVATION':'relu',
        #        'DROPOUT_RATE':.2
        #        }
        # acc,pval,plot = GoF(data=data_,hparams=nn_params,model_path=model_path)
        # filename = 'gof_test_report.txt'
        # with open(os.path.join(result_path,filename),'a') as f:
        #     f.write(f'accuracy at fold {s}: {str(acc)}'+'\n')
        #     f.write(f'pval at fold {s}: {str(pval)}'+'\n')
        # #save plot only at even numbers
        # if (s%2)==0:
        #     filename = f'gof_plot_{s}.png'
        #     plot.savefig(os.path.join(result_path,filename))
        #     plot.clf()
        
        # #------------------------------------------------------------------------------------------
        #mortality 
        # nn_params = {'EPOCHS':10,
        #        'BATCH_SIZE':16,
        #        'HIDDEN_UNITS':[10,10],
        #        'ACTIVATION':'relu',
        #        'DROPOUT_RATE':.2
        #        }
        # rf_params = {'N_TREES':150,
        #              'MAX_DEPTH':None}
        # lr_params = {'L1':.5} 
        # for pred_model,params in zip(['RNN','RF','LR'],[nn_params,rf_params,lr_params]):
        #     real_acc,real_auc,syn_acc,syn_auc = mortality_prediction(data=data_,attr=attr,hparams=params,model_path=model_path,pred_model=pred_model)
        #     filename = 'mortality_pred_accuracy.txt'
        #     with open(os.path.join(result_path,'mortality_pred_accuracy.txt'),'a') as f:
        #         f.write(f'Fold: {s} real AUC ({pred_model}): {real_auc}' + '\n')
        #         f.write(f'Fold: {s} synthetic AUC (RNN): {syn_auc}' + '\n')

        # #------------------------------------------------------------------------------------------
        # #trajectory prediction
        # nn_params = {'EPOCHS':10,
        #     'BATCH_SIZE':16,
        #     'HIDDEN_UNITS':[10,10],
        #     'ACTIVATION':'relu',
        #     'DROPOUT_RATE':.2
        #     }
        # real_acc,syn_acc = trajectory_prediction(data=data_,hparams=nn_params,model_path=model_path)
        # filename = 'trajectory_pred_accuracy.txt'
        # with open(os.path.join(result_path,filename),'a') as f:
        #     f.write(f'Real accuracy at fold {s}: {real_acc}'+'\n')
        #     f.write(f'Synthetic accuracy at fold {s}: {syn_acc}'+'\n')

        # #------------------------------------------------------------------------------------------
        # #privacy AIA
        # nn_params = {'EPOCHS':10,
        #     'BATCH_SIZE':16,
        #     'HIDDEN_UNITS':[10,10],
        #     'ACTIVATION':'relu',
        #     'DROPOUT_RATE':0.2
        #     }
        # sensitive_attr = [['age'],['gender'],['race'],['age','gender'],['age','race'],['gender','race'],['age','gender','race']]
        # mape_age,mae_age,acc_gender,acc_race = privacy_AIA(data=data_,attr=attr,scaler=scaler,model_path=model_path,hparams=nn_params,sensitive_attr=sensitive_attr)    
        # filename = 'privacy_AIA.txt'
        # with open(os.path.join(result_path,filename),'a') as f:
        #     age_j = 0
        #     gender_j = 0
        #     race_j = 0
        #     for labels in sensitive_attr:
        #         f.write(f'We are at fold {s}'+'\n')
        #         f.write(f'Labels are {labels}'+'\n')
        #         if 'age' in labels:
        #             f.write(f'Age MAPE: {mape_age[age_j]}'+'\n')
        #             f.write(f'Age MAE: {mae_age[age_j]}'+'\n')
        #             age_j+=1
        #         if 'gender' in labels:
        #             f.write(f'Gender accuracy: {acc_gender[gender_j]}'+'\n')
        #             gender_j+=1
        #         if sum(x.count('race') for x in labels)>0:
        #             f.write(f'Race accuracy: {acc_race[race_j]}'+'\n')
        #             race_j+=1