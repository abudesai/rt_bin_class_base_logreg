import numpy as np
import os

import algorithm.utils as utils
import algorithm.preprocessing.pipeline as pipeline
import algorithm.model.logistic_regression as logistic_regression


# get model configuration parameters 
model_cfg = utils.get_model_config()


class ModelServer:
    def __init__(self, model_path): 
        self.model_path = model_path
        self.preprocessor = None
        self.model = None
    
    
    def _get_preprocessor(self): 
        if self.preprocessor is None: 
            try: 
                self.preprocessor = pipeline.load_preprocessor(self.model_path)
                return self.preprocessor
            except: 
                print(f'Could not load preprocessor from {self.model_path}. Did you train the model first?')
                return None
        else: return self.preprocessor
    
    def _get_model(self): 
        if self.model is None: 
            try: 
                self.model = logistic_regression.load_model(self.model_path)
                return self.model
            except: 
                print(f'Could not load model from {self.model_path}. Did you train the model first?')
                return None
        else: return self.model
        
    
    def _get_predictions(self, data, data_schema):  
        preprocessor = self._get_preprocessor()
        model = self._get_model()
        
        if preprocessor is None:  raise Exception("No preprocessor found. Did you train first?")
        if model is None:  raise Exception("No model found. Did you train first?")
                    
        # transform data - returns a dict of X (transformed input features) and Y(targets, if any, else None)
        proc_data = preprocessor.transform(data)          
        # Grab input features for prediction
        pred_X = proc_data['X'].astype(np.float)        
        # make predictions
        preds = model.predict( pred_X )
        return preds    
    
    
    def predict_proba(self, data, data_schema):  
        
        preds = self._get_predictions(data, data_schema)
        # get class names (labels)
        class_names = pipeline.get_class_names(self.preprocessor, model_cfg)        
        # get the name for the id field
        id_field_name = data_schema["inputDatasets"]["binaryClassificationBaseMainInput"]["idField"]  
        # return te prediction df with the id and class probability fields
        preds_df = data[[id_field_name]].copy()
        preds_df[class_names[0]] = 1 - preds   
        preds_df[class_names[-1]] = preds   
        
        return preds_df
    
    
    
    def predict(self, data, data_schema):
        preds = self._get_predictions(data, data_schema)        
        
        # inverse transform the prediction probabilities to class labels
        pred_classes = pipeline.get_inverse_transform_on_preds(self.preprocessor, model_cfg, preds)    
        # return te prediction df with the id and prediction fields
        id_field_name = data_schema["inputDatasets"]["binaryClassificationBaseMainInput"]["idField"]  
        preds_df = data[[id_field_name]].copy()
        preds_df['prediction'] = pred_classes
        
        return preds_df
