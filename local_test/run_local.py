import os, shutil
import sys
import pandas as pd, numpy as np
import json
import pprint
from skopt.space import Real, Categorical, Integer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

sys.path.insert(0, './../app')
import algorithm.utils as utils 
import algorithm.model_trainer as model_trainer 
import algorithm.model_server as model_server
import algorithm.model_tuner as model_tuner
import algorithm.preprocessing.pipeline as pipeline
import algorithm.model.logistic_regression as logreg 


inputs_path = "./ml_vol/inputs/"

data_schema_path = os.path.join(inputs_path, "data_config")

data_path = os.path.join(inputs_path, "data")
train_data_path = os.path.join(data_path, "training", "binaryClassificationBaseMainInput")
test_data_path = os.path.join(data_path, "testing", "binaryClassificationBaseMainInput")

model_path = "./ml_vol/model/"
hyper_param_path = os.path.join(model_path, "model_config")
model_artifacts_path = os.path.join(model_path, "artifacts")

output_path = "./ml_vol/outputs"
hpt_results_path = os.path.join(output_path, "hpt_outputs")
testing_outputs_path = os.path.join(output_path, "testing_outputs")
errors_path = os.path.join(output_path, "errors")

dataset_name = "credit_card"; target_class = "negative"; other_class = "positive"; target_field = "class"
# dataset_name = "spam"; target_class = 1; other_class = 0; target_field = "class"
# dataset_name = "segment"; target_class = "P"; other_class = "N"; target_field = "binaryClass"
# dataset_name = "telco_churn"; target_class = "Yes"; other_class = "No"; target_field = "Churn"
# dataset_name = "cancer"; target_class = "M"; other_class = "B"; target_field = "diagnosis"


'''
this script is useful for doing the algorithm testing locally without needing 
to build the docker image and run the container.
make sure you create your virtual environment, install the dependencies
from requirements.txt file, and then use that virtual env to do your testing. 
This isnt foolproof. You can still have host os-related issues, so beware. 
'''





def create_ml_vol():    
    dir_tree = {
        "ml_vol": {
            "inputs": {
                "data_config": None,
                "data": {
                    "training": {
                        "binaryClassificationBaseMainInput": None
                    },
                    "testing": {
                        "binaryClassificationBaseMainInput": None
                    }
                }
            },
            "model": {
                "model_config": None,
                "artifacts": None,
            }, 
            
            "outputs": {
                "hpt_outputs": None,
                "testing_outputs": None,
                "errors": None,                
            }
        }
    }    
    def create_dir(curr_path, dir_dict): 
        for k in dir_dict: 
            dir_path = os.path.join(curr_path, k)
            if os.path.exists(dir_path): shutil.rmtree(dir_path)
            os.mkdir(dir_path)
            if dir_dict[k] != None: 
                create_dir(dir_path, dir_dict[k])

    create_dir("", dir_tree)



def copy_example_files():     
    # data schema
    shutil.copyfile(f"./examples/{dataset_name}_schema.json", os.path.join(data_schema_path, f"{dataset_name}_schema.json"))
    # train data    
    shutil.copyfile(f"./examples/{dataset_name}_train.csv", os.path.join(train_data_path, f"{dataset_name}_train.csv"))    
    # test data     
    shutil.copyfile(f"./examples/{dataset_name}_test.csv", os.path.join(test_data_path, f"{dataset_name}_test.csv"))    
    # hyperparameters
    shutil.copyfile("./examples/hyperparameters.json", os.path.join(hyper_param_path, "hyperparameters.json"))



def run_HPT(): 
    # Read data
    train_data = utils.get_data(train_data_path)    
    # read data config
    data_schema = utils.get_data_schema(data_schema_path)  
    # run hyper-parameter tuning. This saves results in each trial, so nothing is returned
    num_trials = 20
    model_tuner.tune_hyperparameters(train_data, data_schema, num_trials, hyper_param_path, hpt_results_path)



def train_and_save_algo():        
    # Read hyperparameters 
    hyper_parameters = utils.get_hyperparameters(hyper_param_path)    
    # Read data
    train_data = utils.get_data(train_data_path)    
    # read data config
    data_schema = utils.get_data_schema(data_schema_path)  
    # get trained preprocessor, model, training history 
    preprocessor, model, history = model_trainer.get_trained_model(train_data, data_schema, hyper_parameters)            
    # Save the processing pipeline   
    pipeline.save_preprocessor(preprocessor, model_artifacts_path)
    # Save the model 
    logreg.save_model(model, model_artifacts_path)
    # Save training history
    logreg.save_training_history(history, model_artifacts_path)    
    print("done with training")



def load_and_test_algo(): 
    # Read data
    test_data = utils.get_data(test_data_path)   
    # read data config
    data_schema = utils.get_data_schema(data_schema_path)    
    # instantiate the trained model 
    predictor = model_server.ModelServer(model_artifacts_path)
    # make predictions
    predictions = predictor.predict_proba(test_data, data_schema)
    # save predictions
    predictions.to_csv(os.path.join(testing_outputs_path, "test_predictions.csv"), index=False)
    # score the results
    score(test_data, predictions)  
    print("done with predictions")



def score(test_data, predictions): 
    predictions["pred_class"] = predictions.apply(lambda row: 
        target_class if row[target_class] >= 0.5 else other_class, axis=1)    
    
    accu = accuracy_score(test_data[target_field], predictions['pred_class'])
    print(f"test accu: {accu}")
    
    f1 = f1_score(test_data[target_field], predictions['pred_class'], pos_label=target_class)
    print(f"f1_score: {f1}")    
    
    precision = precision_score(test_data[target_field], predictions['pred_class'], pos_label=target_class)
    print(f"precision_score: {precision}")    
    
    recall = recall_score(test_data[target_field], predictions['pred_class'], pos_label=target_class)
    print(f"recall_score: {recall}")    
    
    y_true = np.where(test_data[target_field] == target_class, 1., 0.)
    auc = roc_auc_score(y_true, predictions[target_class])
    print(f"auc_score: {auc}")




if __name__ == "__main__": 
    create_ml_vol()   # create the directory which imitates the bind mount on container
    copy_example_files()   # copy the required files for model training   
    # run_HPT()                   # run hyperparameter tuning and save best tuned hyperparameters
    train_and_save_algo()        # train the model and save
    load_and_test_algo()        # load the trained model and get predictions on test data
    
    
    