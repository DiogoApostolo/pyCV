


from collections import Counter
import copy
import math
import pickle
import numpy as np
import arff
import random
import pandas as pd
import scipy
from sklearn.calibration import LabelEncoder

import matplotlib.pyplot as plt

class CrossValidation:



    def __init__(self,file_name,distance_func="default",file_type="arff",metric="N1",is_calculate_dist_matrix=True,is_numeric=True):
        '''
        Constructor method, setups up the the necessary class attributes to be
        used by the complexity measure functions.
        Starts by reading the file in arff format which contains the class samples X (self.X), class labels y (self.y) and contextual information
        about the features (self.meta).
        It also saves in an array the unique labels of all existing classes (self.classes), the number of samples in each class (self.class_count) and
        the indexes in X of every class (self.class_inxs).
        -----
        Parameters:
        file_name (string): Location of the file that contains the dataset.
        distance_func (string): The distance function to be used to calculate the distance matrix. Only available option right now is "HEOM".
        file_type (string): The type of file where the dataset is stored. Only available option right now is "arff".
        
        '''
        if(file_type=="arff"):
            [X,y,meta,class_inds]=self.__read_file(file_name,is_numeric)
        elif(file_type=="pickle"):
            [X,y,meta,class_inds]=self.__prepare_array(file_name)
        elif(file_type=="csv"):
            [X,y,meta,class_inds]=self.__read_csv(file_name)
        else:
            print("Only arff files are available for now")
            return
        self.file_type = file_type
        self.X=np.array(X)
        self.y=np.array(y)

        

        self.classes=np.unique(self.y)
        self.meta=meta
        
        self.class_inxs = class_inds

        self.is_calculate_distance_matrix = is_calculate_dist_matrix
        self.distance_func = distance_func
        
        if is_numeric:
            self.X_num=np.array(X)
            self.y_num=np.array(y)
            self.meta_num=meta
            self.class_inx_num = class_inds

       
        # self.dist_matrix_per_class,self.dist_matrix = self.__calculate_distance_matrix(distance_func=distance_func)
        
        if(is_calculate_dist_matrix):
            self.dist_matrix_per_class,self.dist_matrix = self.__calculate_distance_matrix(distance_func=distance_func)
        
        
        #self.class_count = self.__count_class_instances()

        

       


        self.class_count = self.__count_class_instances()
        self.min_class = self.determine_min_class()
        self.max_class = self.determine_max_class()

        #if(len(self.class_count)<2):
        #   print("ERROR: Less than two classes are in the dataset.")

        return 

    def determine_min_class(self):
        min_inx = 0
        min_val = self.class_count[0]
        for i in range(1,len(self.class_count)):
            num = self.class_count[i]
            if(num < min_val ):
                min_val = num
                min_inx = i 
        return min_inx
        
    def determine_max_class(self):
        max_inx = 0
        max_val = self.class_count[0]
        for i in range(1,len(self.class_count)):
            num = self.class_count[i]
            if(num > max_val ):
                max_val = num
                max_inx = i 

        return max_inx


    
    def get_class_inx(self,y,classes):
        class_inds = []
        for cls in classes:
            cls_ind=np.where(y==cls)[0]
            class_inds.append(cls_ind)
        
        return class_inds

    def __count_class_instances(self):
        '''
        Is called by the __init__ method.
        Count instances of each class in the dataset.
        --------
        Returns:
        class_count (numpy.array): An (Nx1) array with the number of intances for each of the N classes in the dataset 
        '''
        class_count = np.zeros(len(self.classes))
        for i in range(len(self.classes)):
            count=len(np.where(self.y == self.classes[i])[0])
            class_count[i]+=count
        return class_count
    

    def is_categorical(self,X):
        meta = []
        # Iterate through each column in the array
        for i in range(X.shape[1]):
            column = X[:, i]
            # Check if all elements in the column can be converted to float (numerical)
            try:
                float_column = column.astype(float)
                meta.append(0)
            except ValueError:
                # If conversion to float raises an error, the column likely contains non-numeric values (categorical)
                meta.append(1)

        # Print the detected numerical and categorical feature indices
        return meta
    
    def __prepare_array(self,dataset_name):
        
        


        #change this
        with open(dataset_name, 'rb') as f:
            epsilon_train = pickle.load(f)

        
        X = epsilon_train[:,0:len(epsilon_train[0])-1]
        y = epsilon_train[:,-1]

        meta = self.is_categorical(X)

        
        #indentify the existing classes
        classes = np.unique(y)

        class_inds = []
        for cls in classes:
            cls_ind=np.where(y==cls)[0]
            class_inds.append(cls_ind)


        #create new class labels from 0 to n, where n is the number of classes
        y = [np.where(classes == i)[0][0] for i in y]
        
        
             

        return [X,y,meta,class_inds]
    
    def __convert_columns_to_float(self,arr):
        num_cols = arr.shape[1]
        converted_arr = np.empty_like(arr, dtype=float)
        
        for i in range(num_cols):
            try:
                converted_arr[:, i] = arr[:, i].astype(float)
            except ValueError:
                converted_arr[:, i] = arr[:, i]
        
        return converted_arr


    def __fill_missing_with_column_means(self,arr):
        # Calculate the mean of each column ignoring nan values
        column_means = np.nanmean(arr, axis=0)

        # Find indices of nan values
        nan_indices = np.isnan(arr)

        # Fill nan values with column means
        arr[nan_indices] = np.take(column_means, np.where(nan_indices)[1])

        return arr

    def encode(self,X,meta):
        self.encoders = []
        for i in range(len(meta)):
            
            if meta[i]==1:
                label_encoder=LabelEncoder()
                
                # Fit the LabelEncoder to your categorical values and transform them to numerical values
                numerical_values = label_encoder.fit_transform(X[:,i])
                X[:,i] = numerical_values
                self.encoders.append(label_encoder)
        if 1 in meta:
            X = X.astype(float)

        return X
    
    def decode_var(self,X,meta):
        X = X.astype(object)
        j=0

        for i in range(len(meta)):
            
            if meta[i]==1:
                
                # Fit the LabelEncoder to your categorical values and transform them to numerical values
                numerical_values = self.encoders[j].inverse_transform(X[:,i].astype(int))
               
                X[:,i] = numerical_values
                j+=1
        

        return X


    def __read_csv(self,dataset_name):
        data = np.genfromtxt(dataset_name, delimiter=',', dtype=None, encoding=None, missing_values=" ")
        
        #skip header start at 1

        self.header = data[0:1]
     
        X = data[1:,0:len(data[0])-1]
        y = data[1:,-1]
        meta = self.is_categorical(X)

        
        
        
        #missing values
        X[X == ""] = np.nan
        X[X == " "] = np.nan
        X[X == "NA"] = np.nan
        X[X == "?"] = np.nan


        X = self.encode(X,meta)
        X = self.__convert_columns_to_float(X)
        X = self.__fill_missing_with_column_means(X)
        
        #indentify the existing classes
        classes = np.unique(y)

        class_inds = []
        for cls in classes:
            cls_ind=np.where(y==cls)[0]
            class_inds.append(cls_ind)

        #create new class labels from 0 to n, where n is the number of classes
        y = [np.where(classes == i)[0][0] for i in y]
        return [np.array(X),np.array(y),meta,class_inds]

    
    
    
    
    
    
    
    def __read_file(self,file,is_numeric):
        file_arff = arff.load(open(file, 'r'))
        data = file_arff['data']
        num_attr = len(data[0])-1
        att=file_arff['attributes']

        #print(att)
        self.att = att

        meta=[]
        for i in range(len(att)-1):
            if(att[i][1]=="NUMERIC" or att[i][1]=="REAL" or att[i][1]=="INTEGER"):
                meta.append(0)
            else:
                meta.append(1)

        
        X = np.array([i[:num_attr] for i in data],dtype=object)
        y = np.array([i[-1] for i in data])


        X = self.encode(X,meta)


        '''
        inxs = np.random.permutation(len(X))
        X = X[inxs]
        y = y[inxs]
        '''

        if is_numeric:
            for i in range(len(meta)):
                if meta[i]==1:
                    
                    b, c = np.unique(X[:,i], return_inverse=True)
                    X[:,i] = c

            if 1 in meta:
                X = X.astype(np.float64)
        

        

        classes = np.unique(y)
        
        

        class_inds = []
        for cls in classes:
            cls_ind=np.where(y==cls)[0]
            class_inds.append(cls_ind)
        


        X = np.array(X,dtype=object)
        
        y = np.array(y)
        

        return [X,y,meta,class_inds]
    


    def __distance_HEOM(self,X):
        '''
        Is called by the calculate_distance_matrix method.
        Calculates the distance matrix between all pairs of points from an input matrix, using the HEOM metric, that way categorical attributes are
        allow in the dataset.
        --------
        Parameters: 
        X (numpy.array): An (N*M) numpy matrix containing the points, where N is the number of points and M is the number of attributes per point.
        --------
        Returns:
        dist_matrix (numpy.array): A (M*M) matrix containing the distance between all pairs of points in X

        '''
        
        
        meta = self.meta
        dist_matrix=np.zeros((len(X),len(X)))
        unnorm_dist_matrix = np.zeros((len(X),len(X)))

        #calculate the ranges of all attributes
        #range_max=np.max(X,axis=0)
        #range_min=np.min(X,axis=0)


        range_max = np.array([])
        range_min = np.array([])
        for attr in range(len(X[0])):
            
            if(meta[attr]==0):
                range_max = np.append(range_max,np.max(X[:,attr]))
                range_min = np.append(range_min,np.min(X[:,attr]))
            else:
                range_max = np.append(range_max,0)
                range_min = np.append(range_min,0)

        
        #print(range_max)
        #print(range_min)
        
        for i in range(len(X)): 
            for j in range(i+1,len(X)):
                #for attribute
                dist = 0
                unnorm_dist = 0
                for k in range(len(X[0])):
                    #missing value
                    if(X[i][k] == None or X[j][k]==None):
                        dist+=1
                        unnorm_dist+=1
                    #numerical
                    if(meta[k]==0):
                        #dist+=(abs(X[i][k]-X[j][k]))**2
                        
                        #dist+=(abs(X[i][k]-X[j][k])/(range_max[k]-range_min[k]))**2
                        if(range_max[k]==range_min[k]):
                            dist+=(abs(X[i][k]-X[j][k]))**2
                            unnorm_dist+=(abs(X[i][k]-X[j][k]))**2
                        else:
                            dist+=(abs(X[i][k]-X[j][k])/(range_max[k]-range_min[k]))**2
                            unnorm_dist+= abs(X[i][k]-X[j][k])**2
                            
                            #dist+=(abs(X[i][k]-X[j][k]))**2
                    #categorical
                    if(meta[k]==1):
                        if(X[i][k]!=X[j][k]):
                            dist+=1
                            unnorm_dist+=1

                dist_matrix[i][j]=np.sqrt(dist)
                dist_matrix[j][i]=np.sqrt(dist)

                unnorm_dist_matrix[i][j]=np.sqrt(unnorm_dist)
                unnorm_dist_matrix[j][i]=np.sqrt(unnorm_dist)
        #print(dist_matrix)
        return dist_matrix,unnorm_dist_matrix
    


    def __calculate_distance_matrix(self,distance_func="HEOM"):
        '''
        Is called by the __init__ method.
        Function used to select which distance metric will be used to calculate the distance between a matrix of points.
        Only the HEOM metric is implemented for now, however if more metrics are added this function can easily be changed to
        incomporate the new metrics.
        --------
        Parameters:
        X (numpy.array): An (N*M) numpy matrix containing the points, where N is the number of points and M is the number of attributes per point.
        distance_func (string): The distance function to be used, only available option right now is "HEOM"
        --------
        Returns:
        dist_matrix (numpy.array): A (M*M) matrix containing the distance between all pairs of points in X
        dist_matrix_per_class (numpy.array): An array with distance matrixes for the samples in each class.
        --------
        '''

        dist_matrix_per_class = []
        if(distance_func=="HEOM"):
            
            dist_matrix,unnorm_dist_matrix = self.__distance_HEOM(self.X)

        elif(distance_func=="default"):

            dist_matrix,unnorm_dist_matrix = self.__distance_HEOM(self.X)
        
        #add other distance functions

        #separate per class
        for c_count in range(len(self.classes)):
            
            dist_matrix_cls = dist_matrix[self.class_inxs[c_count][:, None],self.class_inxs[c_count]]
           
            dist_matrix_per_class.append(dist_matrix_cls)
        
        
        
        return dist_matrix_per_class,dist_matrix
    
    
    def __write_arff(self,X_res,y_res,output_folder,file):
        '''
        Called by write_folds to write a partition into a file.
        -----
        Parameters:
        X_res (np.array): An array containing the samples
        y_res (np.array): An array containing the labels of the samples in X
        output_folder (str): The name to save the partition under
        file (str): the folder the files will be saved on
        '''
       

        X_res = pd.DataFrame(X_res)
       
        y_res = pd.DataFrame(y_res)
    
        y_res.rename(columns = {0 : 'class'}, inplace = True)
        df =  X_res.join(y_res)

        #print(df)
        #attributes = [(str(j), 'NUMERIC') if X_res[j].dtypes in ['int64', 'float64'] else (j, X_res[j].unique().astype(str).tolist()) for j in X_res]
        #attributes += [('label',['0.0','1.0'])]


        attributes = self.att

        arff_dic = {
                'attributes': attributes,
                'data': df.values,
                'relation': 'myRel',
                'description': ''
                }
        #print(arff_dic)
        
        #new_name = file.split(".")[0]  + ".arff"
        with open(output_folder + file + ".arff", "w", encoding="utf8") as f:
            arff.dump(arff_dic, f)
    

    def __write_csv(self,X_res,y_res,output_folder,file):
        

        # Concatenate X and y horizontally
        data = np.column_stack((X_res, y_res))
        
        header_str = ','.join(self.header[0])

        # Save to CSV
        np.savetxt(output_folder + file + ".csv", data, delimiter=',', fmt='%f', header=header_str, comments='')


    def write_folds(self,folds,folds_y,filename,folder):
        '''
        Function used to write the folds to arff files, it will create the n training and testing partitions, where n is lenght of folds array passed as parameter.
        -------
        Parameters:
        folds (numpy.array): An array of arrays contaning the n folds to be saved to the file
        folds_y (numpy.array): An array of arrays containing the labels of the samples in each fold
        filename (str): The filename the partitions will be saved under. Each partition will contain a suffix in from of the filename to indentify it. For example, assuming 5 folds, the first train partition will be
        saved under "filename-5-1tra.arff" and the first test partition under "filename-5-1tst.arff"
        folder (str): the folder the files will be saved on
        --------
        Returns:
        dist_matrix (numpy.array): A (M*M) matrix containing the distance between all pairs of points in X
        
        
        '''



        training_partitions = []
        testing_partitions = []

        #n partitions will be generated. n is equal to the number of folds. Each partition will have n-1 folds for training and the remaining fold will be used for testing.
        for i in range(len(folds)):

            train_fold_X=[]
            test_fold_X = np.array(folds[i])

            train_fold_y=[]
            test_fold_y= np.array(folds_y[i])

            for j in range(len(folds)):

                if(j!=i):
                    
                    if(len(train_fold_X)==0):
                        train_fold_X = folds[j]
                        train_fold_y = folds_y[j]
                    else:
                        train_fold_X = np.append(train_fold_X,folds[j],axis=0)
                        train_fold_y = np.append(train_fold_y,folds_y[j])


            training_partitions.append(train_fold_X)
            testing_partitions.append(test_fold_X)

            #name_tra = file.split(".")[0] + "-" + cv_algorithm +"-V" + str(v) + "-" + str(fold_num) + "-" + str(i+1) + "tra.arff" 
            #name_tst = file.split(".")[0] + "-" + cv_algorithm +"-V" + str(v) + "-" + str(fold_num) + "-" + str(i+1) + "tst.arff" 

            #make the new file name for each partition
            name_tra = filename + "-" + str(len(folds)) + "-" + str(i+1) + "tra"
            name_tst = filename + "-" + str(len(folds)) + "-" + str(i+1) + "tst"

            #write the partitions to files


            train_fold_X = self.decode_var(train_fold_X,self.meta)
            test_fold_X = self.decode_var(test_fold_X,self.meta)
            
            if(self.file_type=="arff"):
                self.__write_arff(train_fold_X,train_fold_y,folder,name_tra)
                self.__write_arff(test_fold_X,test_fold_y,folder,name_tst)
            elif(self.file_type=="pickle"):
                pass
            elif(self.file_type=="csv"):
                self.__write_csv(train_fold_X,train_fold_y,folder,name_tra)
                self.__write_csv(test_fold_X,test_fold_y,folder,name_tst)
            


    def reorganize(self, iter,folds,folds_y,folds_inx,selected_class_inx,ratio=0.1):
        
        
        for i in range(iter):
            IH_folds,IH_folds_inx  = self.reorganize_folds(folds,folds_y,folds_inx,selected_class_inx,ratio=0.1)

        return IH_folds,IH_folds_inx

    def reorganize_folds(self,folds,folds_y,folds_inx,selected_class_inx,ratio=0.1):
        
        classes = self.classes
        
        inxs = []
        
        
        fold = np.array(folds[0])
        fold_y = np.array(folds_y[0])
        


        class_inxs = self.get_class_inx(fold_y,classes)
        min_class = fold[class_inxs[selected_class_inx]]
        min_num = len(min_class)
        inx_num_min = math.ceil(min_num*ratio) 


        #interate through all folds and find how much can be sampled by finding the mininum
        for i in range(1,len(folds)):
            fold = np.array(folds[i])
            fold_y = np.array(folds_y[i])
            

            class_inxs = self.get_class_inx(fold_y,classes)
            min_class = fold[class_inxs[selected_class_inx]]
            min_num = len(min_class)
            inx_num = math.ceil(min_num*ratio) 
            if(inx_num < inx_num_min):
                inx_num_min = inx_num

        inx_num = inx_num_min


        #sampling from each fold
        for i in range(len(folds)):

            fold = np.array(folds[i])
            fold_y = np.array(folds_y[i])

            class_inxs = self.get_class_inx(fold_y,classes)
            min_class = fold[class_inxs[selected_class_inx]]
            inx_array= np.random.choice(np.arange(len(min_class), dtype=np.int32), size=(1, inx_num), replace=False)
            inxs.append(inx_array[0])

            
        
        #rotate
        for i in range(len(folds)-1):
            folds[i] = np.array(folds[i])
            folds_inx[i] = np.array(folds_inx[i])
            fold_y = np.array(folds_y[i])
            
            
            class_inxs = self.get_class_inx(fold_y,classes)


            folds[i+1] = np.append(folds[i+1],folds[i][class_inxs[selected_class_inx][inxs[i]]],axis=0)
            folds[i]=np.delete(folds[i], class_inxs[selected_class_inx][inxs[i]],axis=0)        
            
            folds_inx[i+1] = np.append(folds_inx[i+1],folds_inx[i][class_inxs[selected_class_inx][inxs[i]]])
            folds_inx[i]=np.delete(folds_inx[i], class_inxs[selected_class_inx][inxs[i]]) 

        
        folds[len(folds)-1] = np.array(folds[len(folds)-1])
        folds_inx[len(folds)-1] = np.array(folds_inx[len(folds)-1])

        fold_y = np.array(folds_y[len(folds)-1])
        class_inxs = self.get_class_inx(fold_y,classes)
        
        
        folds[0]=np.append(folds[0],folds[len(folds)-1][class_inxs[selected_class_inx][inxs[len(folds)-1]]],axis=0)
        folds[len(folds)-1]=np.delete(folds[len(folds)-1], class_inxs[selected_class_inx][inxs[len(folds)-1]],axis=0) 


        folds_inx[0]=np.append(folds_inx[0],folds_inx[len(folds_inx)-1][class_inxs[selected_class_inx][inxs[len(folds_inx)-1]]])
        folds_inx[len(folds_inx)-1]=np.delete(folds_inx[len(folds_inx)-1], class_inxs[selected_class_inx][inxs[len(folds_inx)-1]]) 
        return folds,folds_inx


    def __knn(self,inx,line,k,clear_diag=True):
        count = np.zeros(len(self.classes))
        if(clear_diag):
            line[inx]=math.inf
        for i in range(k):
            index=np.where(line == min(line))[0][0]
            line[index]=math.inf
            #if(self.y[index]!=self.y[inx]):
            cls_inx = np.where(self.classes == self.y[index])[0][0]  
            count[cls_inx]+=1

        return count


    def instance_hardness_N3(self,k=5):
        sample_hardness = []
        #for each sample
        for sample in range(len(self.X)):
            #calculate the nearest neighbour
            count       = self.__knn(sample,copy.copy(self.dist_matrix[sample]),k)
            cls_inx     = np.where(self.classes == self.y[sample])[0][0]
            class_count = count[cls_inx]
            sample_hardness.append(1-class_count/k)

        return np.array(sample_hardness)


    def instance_hardness_N1(self):
        
        #calculate the MST using the distance matrix
        #print(dist_matrix)
        minimum_spanning_tree = scipy.sparse.csgraph.minimum_spanning_tree(csgraph=np.triu(self.dist_matrix, k=1))
        

        #convert the mst to an array
        mst_array = minimum_spanning_tree.toarray().astype(float)
        
        
        for i in range(len(mst_array)):
            for j in range(i + 1, len(mst_array[0])):
                mst_array[j][i] = mst_array[i][j]


        #iterate over the MST to determine which vertixes that are connected are from different classes.
        vertix = []
        
        for i in range(len(mst_array)):
            dif_count = 0
            total_count = 0
            for j in range(len(mst_array[0])):
                if(mst_array[i][j]!=0):
                    total_count +=1
                    
                    if (self.y[i]!=self.y[j]):
                        
                        dif_count+=1
            
            
            vertix.append(dif_count/total_count)


        return np.array(vertix)

    def aux_F2(self,sample_c1,sample_c2):
        maxmin = np.max([np.min(sample_c1,axis=0),np.min(sample_c2,axis=0)],axis=0)
        minmax = np.min([np.max(sample_c1,axis=0),np.max(sample_c2,axis=0)],axis=0)
        
    
        sample_hardness = np.zeros(len(self.X))
        
        denom = ((minmax-maxmin)/2)
        denom[denom==0]=1
        
        r =  1-(abs(self.X - (minmax+maxmin)/2)/denom)
        sample_hardness= np.mean(r,axis=1)
            

        return sample_hardness

    def instance_hardness_F2(self):
        
        #one vs one method

       
        if(len(self.classes)==1):
            IH = -1*np.ones(len(self.X))         

        elif(len(self.classes)==2):
            sample_c1 = self.X[self.class_inxs[0]]
            sample_c2 = self.X[self.class_inxs[1]]

            IH = self.aux_F2(sample_c1,sample_c2)
        else:
            #One vs All
            IH = []
            for i in range(len(self.class_inxs)):
            
                sample_c1 = self.X[self.class_inxs[i]]
                
                c2_inx = []
                for j in range(len(self.class_inxs)):
                    if(j!=i):
                        c2_inx.append(self.class_inxs[j])
                
                sample_c2 = self.X[c2_inx]

                sample_hardness = self.aux_F2(sample_c1,sample_c2)
                IH.append(sample_hardness)          
                
        np.mean(IH,axis=0)
        return IH
    

    def calculate_instance_hardness(self,metric="N1"):
        '''
        Calculates the instance hardness of each sample in the dataset according to the chosen complexity metric
        ------
        Parameters:
        metric (str): The chosen complexity metric to measure instance hardnes
        -------
        Returns:

        '''
        if(metric=="N1"):
            instance_hardness = self.instance_hardness_N1()
        elif(metric=="N3"):
            instance_hardness = self.instance_hardness_N3(k=5)
        elif(metric=="F2"):
            instance_hardness = self.instance_hardness_F2()
        else:
            instance_hardness = self.instance_hardness_N1()
        
        instance_hardness_per_class = []
        instance_hardness_per_class_inx = []

        class_inxs = self.class_inxs

        '''
        plt.plot(self.X[class_inxs[0],0],self.X[class_inxs[0],1],"o")
        plt.plot(self.X[class_inxs[1],0],self.X[class_inxs[1],1],"o")

        
        for j in range(len(instance_hardness)):
            plt.text(self.X[j][0], self.X[j][1], str(round(instance_hardness[j],2)), fontsize=12, ha='right')

        maxmin = np.max([np.min(self.X[class_inxs[0]],axis=0),np.min(self.X[class_inxs[1]],axis=0)],axis=0)
        minmax = np.min([np.max(self.X[class_inxs[0]],axis=0),np.max(self.X[class_inxs[1]],axis=0)],axis=0)

        
        plt.axhline(y=maxmin[1], color='r', linestyle='-')
        plt.axvline(x=maxmin[0], color='r', linestyle='-')

        plt.axhline(y=minmax[1], color='r', linestyle='-')
        plt.axvline(x=minmax[0], color='r', linestyle='-')
        '''
        
        for c_count in range(len(self.classes)):
            instance_hardness_cls = instance_hardness[self.class_inxs[c_count]]
            
            
            instance_hardness_per_class.append(instance_hardness_cls)
            instance_hardness_per_class_inx.append(np.argsort(instance_hardness_cls))
        

        return instance_hardness_per_class,instance_hardness_per_class_inx

    def getClosest(self,pos,dist_matrix):
        '''
        Gets the closest sample to the one passed as parameter according to a distance matrix.
        ------
        Parameters:
        pos (int): The position of the sample in the distance matrix
        distance_matrix (np.array): An NxN distance matrix containing the distances from each samples to all others. Row n correponds the distance of sample n to all other samples.
        ------
        Returns:
        new_pos(int): The position of the closest sample
        
        '''
        #select closest sample 
        #in case of the first row the value has to be different
        if(pos!=0):
            min_val = dist_matrix[pos][0]
            new_pos = 0
        else:
            min_val = dist_matrix[pos][1]
            new_pos = 1
        

        for i in range(len(dist_matrix[pos])):
            #check if not itself
            if i!=pos:
                if dist_matrix[pos][i] < min_val:
                    min_val = dist_matrix[pos][i]
                    new_pos = i
            #get min


        #remove sample from dist matrix
        return new_pos
    


    def getMostDistant(self,pos,dist_matrix):
        '''
        Gets the most distant sample to the one passed as parameter according to a distance matrix.
        ------
        Parameters:
        pos (int): The position of the sample in the distance matrix
        distance_matrix (np.array): An NxN distance matrix containing the distances from each samples to all others. Row n correponds the distance of sample n to all other samples.
        ------
        Returns:
        new_pos (int): The position of the most distant sample
        
        '''
        #select closest sample 
        #in case of the first row the value has to be different
        if(pos!=0):
            min_val = dist_matrix[pos][0]
            new_pos = 0
        else:
            min_val = dist_matrix[pos][1]
            new_pos = 1
        

        for i in range(len(dist_matrix[pos])):
            #check if not itself
            if i!=pos:
                if dist_matrix[pos][i] > min_val:
                    min_val = dist_matrix[pos][i]
                    new_pos = i
            #get min


        #remove sample from dist matrix
        return new_pos


    def SCV(self,foldNum=5):
        '''
        Function used to perform statified cross validation (SCV), each fold will contain the same number of samples of each class, which
        makes sure to mitigate prior probability dataset shift. 
        ------
        Parameters:
        foldNum (int): Number of folds to split the dataset into. Default value of 5 folds.
        ------
        Returns:
        folds (numpy.array): An array of foldNum size containing the split dataset according to SCV. Each element of this array corresponds to a fold.
        folds_y (numpy.array): An array of foldNum size containing the class labels for the samples in each fold of the fold array. 
        folds_inx (numpy.array): An array of foldNum size containing the indexes of the samples (in relation to the complete dataset) in each fold of the fold array.
        '''
        
        folds = []
        folds_y = []
        folds_inx = []

        for i in range(foldNum):
            folds.append([])
            folds_y.append([])
            folds_inx.append([])


        class_inxs = self.class_inxs
        
        #for each class
        for c_count in range(len(self.classes)):
            
            cls_inxs = copy.copy(class_inxs[c_count])
            random.shuffle(cls_inxs)

            orgCount = len(cls_inxs)
            n = orgCount//foldNum

            for i in range(0, foldNum):
                folds_inx[i]+=list(cls_inxs[i*n:(i+1)*n])
                folds[i]+=list(self.X[cls_inxs[i*n:(i+1)*n]])
                folds_y[i]+=list(self.y[cls_inxs[i*n:(i+1)*n]])

            #for the remained of the samples
            f_i = 0
            for i in range(n*foldNum, len(cls_inxs)):

                folds_inx[f_i]+=[cls_inxs[i]]
                folds[f_i]+=[self.X[cls_inxs[i]]]
                folds_y[f_i]+=[self.y[cls_inxs[i]]]

                f_i +=1

    

        return folds,folds_y,folds_inx

    



    def DBSCV(self,foldNum=5):
        '''
        Function used to perform distribution balanced statified cross validation (DBSCV) [1], each fold will contain the same number of samples of each class, which
        makes sure to mitigate prior probability dataset shift. Additionally, this method also tries to mitigate covariate shift by assigning the samples from the 
        feature space evenly in each fold.
        ------
        Parameters:
        foldNum (int): Number of folds to split the dataset into. Default value of 5 folds.
        ------
        Returns:
        folds (numpy.array): An array of foldNum size containing the split dataset according to SCV. Each element of this array corresponds to a fold.
        folds_y (numpy.array): An array of foldNum size containing the class labels for the samples in each fold of the fold array. 
        folds_inx (numpy.array): An array of foldNum size containing the indexes of the samples (in relation to the complete dataset) in each fold of the fold array.
        ------
        References:
        [1] Xinchuan Zeng and Tony R. Martinez. Distribution-balanced stratified cross-validation for accuracy estimation. Journal of Experimental & Theoretical Artificial Intelligence,
        12(1):1–12, 2000. doi: 10.1080/095281300146272. URL https://doi.org/10.1080/095281300146272.5
        '''
        
        if(self.is_calculate_distance_matrix==False):
            self.dist_matrix_per_class,self.dist_matrix = self.__calculate_distance_matrix(distance_func=self.distance_func)
        
        
        folds = []
        folds_y = []
        folds_inx = []
        

        class_inxs = self.class_inxs
       
        
        for i in range(foldNum):
            folds.append([])
            folds_y.append([])
            folds_inx.append([])

        #for each class
        for c_count in range(len(self.classes)):
            X_cls = self.X[class_inxs[c_count]]
            y_cls = self.y[class_inxs[c_count]]
            cls_inxs = class_inxs[c_count]

            #get the distant matrix for the current class
            dist_matrix_cls = self.dist_matrix_per_class[c_count]
                
            i = 0
            cnt=len(X_cls)
            

            #select first position
            pos = random.randint(0,len(X_cls)-1)
            sample = X_cls[pos]
            sample_y = y_cls[pos]


            #while there still are samples left
            while(cnt>1): 
                #assign sample
                folds[i].append(sample)
                folds_y[i].append(sample_y)
                folds_inx[i].append(cls_inxs[pos])
                
                
                
                cnt = cnt-1

                #rotate through the folds at each iteration
                i = (i+1)%foldNum


                #remove from dataset
                X_cls = np.delete(X_cls, (pos), axis=0)
                y_cls = np.delete(y_cls, (pos), axis=0)
                cls_inxs = np.delete(cls_inxs, (pos), axis=0)


                #get the next sample, which is the closest in distance to the current fold
                new_pos = self.getClosest(pos,dist_matrix_cls)
                

                #remove form dist matrix
                dist_matrix_cls = np.delete(dist_matrix_cls, (pos), axis=0)
                dist_matrix_cls = np.delete(dist_matrix_cls, (pos), axis=1)
                

                if(new_pos > pos):
                    pos = new_pos-1
                else:
                    pos = new_pos


                #new sample
                sample = X_cls[pos]
                sample_y = y_cls[pos]

            #assign the last sample
            folds[i].append(sample)
            folds_y[i].append(sample_y)        
            folds_inx[i].append(cls_inxs[pos])    

        return folds,folds_y,folds_inx
    
    def DOBSCV(self,foldNum=5):
        '''
        Function used to perform distribution optimized balanced statified cross validation (DOBSCV) [1], each fold will contain the same number of samples of each class, which
        makes sure to mitigate prior probability dataset shift. Additionally, this method also tries to mitigate covariate shift by assigning the samples from the 
        feature space evenly in each fold. DOBSCV is less sensitive to random choices since, after assigning a sample to each of the folds, it picks a new random
        sample to restart the process.
        ------
        Parameters:
        foldNum (int): Number of folds to split the dataset into. Default value of 5 folds.
        ------
        Returns:
        folds (numpy.array): An array of foldNum size containing the split dataset according to SCV. Each element of this array corresponds to a fold.
        folds_y (numpy.array): An array of foldNum size containing the class labels for the samples in each fold of the fold array. 
        folds_inx (numpy.array): An array of foldNum size containing the indexes of the samples (in relation to the complete dataset) in each fold of the fold array.
        ------
        References:
        [1] Jose Garcıa Moreno-Torres, Jose A. Saez, and Francisco Herrera. Study on the impact of
        partition-induced dataset shift on k-fold cross-validation. IEEE Transactions on Neural
        Networks and Learning Systems, 23(8):1304–1312, 2012b. doi: 10.1109/TNNLS.2012.2199516
        '''

        if(self.is_calculate_distance_matrix==False):
            self.dist_matrix_per_class,self.dist_matrix = self.__calculate_distance_matrix(distance_func=self.distance_func)


        folds = []
        folds_y = []
        folds_inx = []

        for i in range(foldNum):
            folds.append([])
            folds_y.append([])
            folds_inx.append([])


        class_inxs = self.class_inxs
        
        for c_count in range(len(self.classes)):
            X_cls = self.X[class_inxs[c_count]]
            y_cls = self.y[class_inxs[c_count]]


            #get the distant matrix for the current class
            dist_matrix_cls = self.dist_matrix_per_class[c_count]
            cls_inxs = class_inxs[c_count]

            
            cnt=len(X_cls)

            #while there are still samples left
            while(cnt>0): 

                #select first position
                pos = random.randint(0,len(X_cls)-1)
                sample = X_cls[pos]
                sample_y = y_cls[pos]


                folds[0].append(sample)
                folds_y[0].append(sample_y)
                folds_inx[0].append(cls_inxs[pos])
                
                
                
                cnt = cnt-1

                
                if(cnt==0):
                    break
                    
                
                #to each fold it will assign the closest sample to the sample in the pos index
                for i in range(1,foldNum):
                    #assign the closest sample
                    p2 = self.getClosest(pos,dist_matrix_cls)


                    #assign sample
                    folds[i].append(X_cls[p2])
                    folds_y[i].append(y_cls[p2])
                    folds_inx[i].append(cls_inxs[p2])


                    #delete the sample selected
                    X_cls = np.delete(X_cls, (p2), axis=0)
                    y_cls = np.delete(y_cls, (p2), axis=0)
                    cls_inxs = np.delete(cls_inxs, (p2), axis=0)

                    dist_matrix_cls = np.delete(dist_matrix_cls, (p2), axis=0)
                    dist_matrix_cls = np.delete(dist_matrix_cls, (p2), axis=1)

                    cnt = cnt-1

                    if(p2 < pos):
                        pos = pos-1
                    


                    if(cnt==0):
                        break

                if(cnt==0):
                    break    
                
                #for the last sample
                dist_matrix_cls = np.delete(dist_matrix_cls, (pos), axis=0)
                dist_matrix_cls = np.delete(dist_matrix_cls, (pos), axis=1)

                X_cls = np.delete(X_cls, (pos), axis=0)
                y_cls = np.delete(y_cls, (pos), axis=0)
                cls_inxs = np.delete(cls_inxs, (pos), axis=0)

        return folds,folds_y,folds_inx

    def MSSCV(self,foldNum=5):
        '''
        Function used to perform maximally shifted statified cross validation (MSSCV) [1], each fold will contain the same number of samples of each class, which
        makes sure to mitigate prior probability dataset shift. Additionally, this method also tries distributes samples so that each fold is has different as possible.
        ------
        Parameters:
        foldNum (int): Number of folds to split the dataset into. Default value of 5 folds.
        ------
        Returns:
        folds (numpy.array): An array of foldNum size containing the split dataset according to SCV. Each element of this array corresponds to a fold.
        folds_y (numpy.array): An array of foldNum size containing the class labels for the samples in each fold of the fold array. 
        folds_inx (numpy.array): An array of foldNum size containing the indexes of the samples (in relation to the complete dataset) in each fold of the fold array.
        ------
        References:
        [1] Jose Garcıa Moreno-Torres, Jose A. Saez, and Francisco Herrera. Study on the impact of
        partition-induced dataset shift on k-fold cross-validation. IEEE Transactions on Neural
        Networks and Learning Systems, 23(8):1304–1312, 2012b. doi: 10.1109/TNNLS.2012.2199516
        '''

        if(self.is_calculate_distance_matrix==False):
            self.dist_matrix_per_class,self.dist_matrix = self.__calculate_distance_matrix(distance_func=self.distance_func)

        folds = []
        folds_y = []
        folds_inx = []
        

        class_inxs = self.class_inxs
       
        
        for i in range(foldNum):
            folds.append([])
            folds_y.append([])
            folds_inx.append([])

        #for each class
        for c_count in range(len(self.classes)):
            X_cls = self.X[class_inxs[c_count]]
            y_cls = self.y[class_inxs[c_count]]
            cls_inxs = class_inxs[c_count]

            #get the distance matrix for this class
            dist_matrix_cls = self.dist_matrix_per_class[c_count]
                
            i = 0
            cnt=len(X_cls)
            
            #select the first position
            pos = random.randint(0,len(X_cls)-1)
            sample = X_cls[pos]
            sample_y = y_cls[pos]

            while(cnt>1): 
                folds[i].append(sample)
                folds_y[i].append(sample_y)
                folds_inx[i].append(cls_inxs[pos])
                
                
                
                cnt = cnt-1

                #rotate through the folds at each iteration
                i = (i+1)%foldNum


                #remove from dataset
                X_cls = np.delete(X_cls, (pos), axis=0)
                y_cls = np.delete(y_cls, (pos), axis=0)
                cls_inxs = np.delete(cls_inxs, (pos), axis=0)


                #get the sample most distant to the current one, it will be assigned to the next fold
                new_pos = self.getMostDistant(pos,dist_matrix_cls)
                

                #remove form dist matrix
                dist_matrix_cls = np.delete(dist_matrix_cls, (pos), axis=0)
                dist_matrix_cls = np.delete(dist_matrix_cls, (pos), axis=1)
                

                if(new_pos > pos):
                    pos = new_pos-1
                else:
                    pos = new_pos

                sample = X_cls[pos]
                sample_y = y_cls[pos]

            folds[i].append(sample)
            folds_y[i].append(sample_y)        
            folds_inx[i].append(cls_inxs[pos])    

        return folds,folds_y,folds_inx
    

    
    
    
    def DBSCV_IH(self,foldNum=5,metric="F2"):
        '''
        Function used to perform distribution balanced statified cross validation (DBSCV-IH), an alteration made to DBSCV [1], each fold will contain the same number of samples of each class, which
        makes sure to mitigate prior probability dataset shift. Additionally, this method also tries to mitigate covariate shift by assigning the samples of equal instance hardness.
        ------
        Parameters:
        foldNum (int): Number of folds to split the dataset into. Default value of 5 folds.
        metric (str): The metric to measure instance hardness with
        ------
        Returns:
        folds (numpy.array): An array of foldNum size containing the split dataset according to SCV. Each element of this array corresponds to a fold.
        folds_y (numpy.array): An array of foldNum size containing the class labels for the samples in each fold of the fold array. 
        folds_inx (numpy.array): An array of foldNum size containing the indexes of the samples (in relation to the complete dataset) in each fold of the fold array.
        ------
        References:
        [1] Xinchuan Zeng and Tony R. Martinez. Distribution-balanced stratified cross-validation for accuracy estimation. Journal of Experimental & Theoretical Artificial Intelligence,
        12(1):1–12, 2000. doi: 10.1080/095281300146272. URL https://doi.org/10.1080/095281300146272.5
        '''
        
        if(self.is_calculate_distance_matrix==False and metric!="F2"):
            self.dist_matrix_per_class,self.dist_matrix = self.__calculate_distance_matrix(distance_func=self.distance_func)
        
        
        
        folds = []
        folds_y = []
        folds_inx = []
        

        class_inxs = self.class_inxs
       
        instance_hardness_per_class,instance_hardness_per_class_inxs = self.calculate_instance_hardness(metric)


        


        for i in range(foldNum):
            folds.append([])
            folds_y.append([])
            folds_inx.append([])

        #for each class
        for c_count in range(len(self.classes)):
            X_cls = self.X[class_inxs[c_count]]
            y_cls = self.y[class_inxs[c_count]]
            cls_inxs = class_inxs[c_count]

            #indexes of the samples sorted by instance hardness for the class
            IH_cls = instance_hardness_per_class_inxs[c_count]
            

            i = 0
           
            #while there still are samples left
            for pos in IH_cls:
                #assign sample
                folds[i].append(X_cls[pos])
                folds_y[i].append(y_cls[pos])
                folds_inx[i].append(cls_inxs[pos])
                
                #rotate through the folds at each iteration
                i = (i+1)%foldNum

        
                
        for i in range(len(folds)):
            folds[i] = self.decode_var(np.array(folds[i]),self.meta)

        return folds,folds_y,folds_inx
    



    def check_repeats_and_count(self, lst):
        # Flatten the list of lists
        flat_list = [item for sublist in lst for item in sublist]

        # Check for repeats
        repeats = len(flat_list) != len(set(flat_list))
        if(repeats):
            counts = Counter(flat_list)
            repeated_elements = [item for item, count in counts.items() if count > 1]
            print(f"Repeats: {repeated_elements}")

        # Count total elements
        total_elements = len(flat_list)

        return repeats, total_elements



    def feature_partitioning(self):

        transpose_X = np.transpose(self.X_num)
        resolution = 1
        feature_bounds=[]
        steps = []
        #for every feature
        for j in range(len(self.X_num[0])):
            #(max of the feature - min of the feature)/(num of intervals +1)
            min_feature = min(transpose_X[j])
            max_feature = max(transpose_X[j])

            # if(min_feature == max_feature):
            #     print("OPAHHHH", min_feature, "-", max_feature, self.file_name)

            
            # print("before step")
            # print("min_feature: ", min_feature, "max_feature: ", max_feature, "resolution: ", resolution+1)
            step = (max_feature-min_feature)/(resolution+1)
            # print("after step")
            steps.append(step)
            #calculate the hypercube bounds for each dimension

            if(max_feature==min_feature):
                feature_bounds.append([min_feature, max_feature])
            else:
                # feature_bounds.append(np.arange(min_feature,max_feature,step))
                feature_bounds.append(np.linspace(min_feature,max_feature,num=2+resolution))
       
        sample_dic = {}
        #map each cell to the cell it belongs to (sample->cell)
        for s in range(len(self.X_num)):
            sample = self.X_num[s]
            for j in range(len(self.X_num[0])):
                for k in range(len(feature_bounds[j])):
                
                    if(sample[j]>=feature_bounds[j][k] and sample[j]<=feature_bounds[j][k]+steps[j]):
                        if(str(s) not in sample_dic):
                            sample_dic[str(s)]=""+str(k)
                        else:
                            sample_dic[str(s)]+="-"+str(k)
                        break
        
        #reverse the mapping (cell->sample)
        reverse_dic = {}
        for k,v in sample_dic.items():
            reverse_dic[v]= reverse_dic.get(v,[])
            #values are the class lables 
            reverse_dic[v].append(int(k)) 

        # print("reverse_dic: ", reverse_dic)
        self.feature_partition = reverse_dic
        # return reverse_dic

    def get_feature_partitioning(self, X_num, call_number, X_num_inx):

        transpose_X = np.transpose(X_num)

        resolution = 1
        feature_bounds=[]
        steps = []
        all_samples_equal = True
        #for every feature
        for j in range(len(X_num[0])):
            #(max of the feature - min of the feature)/(num of intervals +1)
            min_feature = min(transpose_X[j])
            max_feature = max(transpose_X[j])

            # if(min_feature == max_feature):
            #     print("OPAHHHH", min_feature, "-", max_feature, self.file_name)

            
            # print("before step")
            # print("min_feature: ", min_feature, "max_feature: ", max_feature, "resolution: ", resolution+1)
            step = (max_feature-min_feature)/(resolution+1)
            # print("after step")
            steps.append(step)
            #calculate the hypercube bounds for each dimension

            if(max_feature==min_feature):
                feature_bounds.append([min_feature, max_feature])
            else:
                all_samples_equal = False
                # feature_bounds.append(np.arange(min_feature,max_feature,step))
                feature_bounds.append(np.linspace(min_feature,max_feature,num=2+resolution))

        # if(all_samples_equal):
        #     print("all samples are equal: ", X_num_inx)

        # print("feature_bounds: ", feature_bounds)
        sample_dic = {}
        #map each cell to the cell it belongs to (sample->cell)
        for s in range(len(X_num)):
            sample = X_num[s]
            global_s = X_num_inx[s]
            for j in range(len(X_num[0])):
                for k in range(len(feature_bounds[j])):
                    
                    if(sample[j]>=feature_bounds[j][k] and sample[j]<=feature_bounds[j][k]+steps[j]):
                        if(str(global_s) not in sample_dic):
                            if(all_samples_equal):
                                sample_dic[str(global_s)]="0-"+str(call_number)+">"+str(k)
                            else:
                                sample_dic[str(global_s)]=str(call_number)+">"+str(k)

                        else:
                            sample_dic[str(global_s)]+="-"+str(k)
                        break
        # print("sample_dic: ", sample_dic)
        #reverse the mapping (cell->sample)
        reverse_dic = {}
        for k,v in sample_dic.items():
            reverse_dic[v]= reverse_dic.get(v,[])
            #values are the class lables 
            reverse_dic[v].append(int(k)) 
        # print("reverse_dic: ", reverse_dic)
        return reverse_dic

    def recursive_feature_partitioning(self, X_num, call_number, X_num_inx ,threshold):
        # if call_number%50==0:
        #     print("call_number: ", call_number)
        call_number+=1
        reverse_dic = self.get_feature_partitioning(X_num, call_number, X_num_inx)
        # print("reverse_dic: ", reverse_dic)

        for k in list(reverse_dic.keys()):  # Create a copy of the keys
            v = reverse_dic[k]
            if(len(v)>threshold and not k.startswith("0")):
                new_X_num = self.X_num[v]
                call_number, temp_reverse_dic = self.recursive_feature_partitioning(new_X_num, call_number, v, threshold)
                reverse_dic.pop(k)
                reverse_dic.update(temp_reverse_dic)
        return call_number, reverse_dic



    def getClosest_inCell(self, current_cell_pos,dist_matrix_cell, cell_indices):
        if(current_cell_pos!=cell_indices[0]):
            min_val = dist_matrix_cell[current_cell_pos][cell_indices[0]]
            new_cls_pos = cell_indices[0]
        else:
            min_val = dist_matrix_cell[current_cell_pos][cell_indices[1]]
            new_cls_pos = cell_indices[1]

        for i in cell_indices:
            if i!=current_cell_pos:
                if dist_matrix_cell[current_cell_pos][i] < min_val:
                    min_val = dist_matrix_cell[current_cell_pos][i]
                    new_cls_pos = i

        return new_cls_pos
    
    def main_feature_partitioning(self, threshold):
        # print(self.X_num)
        #get the feature partitioning
        call_number, self.feature_partition = self.recursive_feature_partitioning(self.X_num, 0, np.arange(len(self.X_num)) ,threshold=threshold)
        total = 0
        for k,v in self.feature_partition.items():
            total+=len(v)
        if total != len(self.X):
            print("\n\nFP - total: ", total, "size: ", len(self.X),"\n\n")
        # print("feature_partition: ", self.feature_partition)
        #separate the feature partition for each class
        class_inxs = self.class_inxs
        self.feature_partition_clxs = [{} for _ in range(len(self.classes))]
        for c_count in range(len(self.classes)):
            X_cls = self.X[class_inxs[c_count]]
            y_cls = self.y[class_inxs[c_count]]
            cls_inxs = class_inxs[c_count]
            temp_feature_partition = self.feature_partition
            feature_partition_cls = {}
            for k,v in temp_feature_partition.items():
                feature_partition_cls[k] = []
                for i in v:
                    if i in cls_inxs:
                        feature_partition_cls[k].append(i)
                if len(feature_partition_cls[k])==0:
                    feature_partition_cls.pop(k)
            self.feature_partition_clxs[c_count] = feature_partition_cls

        #get dist_matrices for each class per cell
        self.dist_matrix_per_class_cell = [{} for _ in range(len(self.classes))]
        for c_count in range(len(self.classes)):
            X_cls = self.X[class_inxs[c_count]]
            y_cls = self.y[class_inxs[c_count]]
            cls_inxs = class_inxs[c_count]
            feature_partition_clxs = self.feature_partition_clxs[c_count]
            dist_matrix_cls = {}
            for k, v in feature_partition_clxs.items():
                # print("v: ", v)
                # print("X[v]", self.X[v])
                if(len(v)>2 and not k.startswith("0")):
                    dist_matrix_cls_cell, _ = self.__distance_HEOM(self.X[v])
                    dist_matrix_cls[k] = dist_matrix_cls_cell
            self.dist_matrix_per_class_cell[c_count] = dist_matrix_cls

        # print("\n\ndist_matrix_per_class_cell:\n", self.dist_matrix_per_class_cell[0])
        # print("distance matrix: " , self.dist_matrix)

        # samples_set = set(tuple(sample) for sample in self.X)
        # print("got: ",len(samples_set), "original: ", len(self.X))

    def featurePartitioning_DBSCV(self, foldNum=5):
        folds = []
        folds_y = []
        folds_inx = []

        class_inxs = self.class_inxs
        
        for i in range(foldNum):
            folds.append([])
            folds_y.append([])
            folds_inx.append([])

        fold_index = np.random.randint(0, foldNum)

        for c_count in range(len(self.classes)):
            X_cls = self.X[class_inxs[c_count]]
            y_cls = self.y[class_inxs[c_count]]
            cls_inxs = class_inxs[c_count]
            feature_partition_cls = self.feature_partition_clxs[c_count]
            dist_matrix_cls_cells = self.dist_matrix_per_class_cell[c_count]

            # #for each cell
            # items = list(feature_partition_cls.items())
            # random.shuffle(items)
            for k, v in feature_partition_cls.items():
                temp_v = v
                if len(temp_v)>2 and not k.startswith("0"):
                    cell_dist_matrix = dist_matrix_cls_cells[k]
                    current_indice = np.random.choice(temp_v, 1, replace=False)[0]

                    while(len(temp_v)>1):
                        # print(self.X[v], "\n", X_cls[i_cls])
                        folds[fold_index].append(self.X[current_indice])
                        folds_y[fold_index].append(self.y[current_indice])
                        folds_inx[fold_index].append(current_indice)
                        fold_index = (fold_index+1)%foldNum

                        temp_v = np.delete(temp_v, np.where(np.isin(temp_v, current_indice))[0][0]) #remove the current sample from the cell
                        cell_samples = np.where(np.isin(v, temp_v))[0] #get the index of the remaining samples relative to the original cell (v)
                        current_cell_pos = np.where(np.isin(v, current_indice))[0][0] #get the index of the current sample in the original cell (v)
                        new_cell_pos =  self.getClosest_inCell(current_cell_pos, cell_dist_matrix, cell_samples)
                        current_indice = v[new_cell_pos]
                    
                    folds[fold_index].append(self.X[current_indice])
                    folds_y[fold_index].append(self.y[current_indice])
                    folds_inx[fold_index].append(current_indice)
                    fold_index = (fold_index+1)%foldNum
                else:
                    random.shuffle(v)
                    for i in v:
                        folds[fold_index].append(self.X[i])
                        folds_y[fold_index].append(self.y[i])
                        folds_inx[fold_index].append(i)
                        fold_index = (fold_index+1)%foldNum
                    
                                       
                    



        # print("_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_\n")
        # print(folds_inx)
        # print(folds_y)
        repeats, total_elements = self.check_repeats_and_count(folds_inx)
        # print(f"Are there repeats? {repeats}")
        if(repeats):
            print(f"Are there repeats? {repeats}")
        if(total_elements != len(self.X_num)):
            print(f"\n\nCV - Total elements: {total_elements}", " size", len(self.X_num), "\n\n")

        # counts_in_dataset = [len(class_inx) for class_inx in self.class_inxs]
        # print(f"Counts in dataset: {counts_in_dataset}")

        # counts_per_fold = self.count_samples_per_class(folds_y)
        # print(f"Counts per fold: {counts_per_fold}")
        # print("\n_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_\n")

        return folds,folds_y,None