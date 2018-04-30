 #------Metrics and other sklearn tools------#
from sklearn.decomposition import PCA, IncrementalPCA, SparsePCA, FastICA, TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score, recall_score, precision_score, auc, roc_curve
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import cross_validate, cross_val_predict, train_test_split

#------Estimators------#
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import LinearSVC, SVC, NuSVC
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier

#------Other tools------#
import pandas as pd
import numpy as np
import time

class pipelines:
	def __init__(self, dataname="", filename=""):
		self.pipes = [] #list of all the combination pipes
		self.params = [] #list of all the target parameters to keep track of
		self.dataname = dataname #name of file containing dataset
		self.filename = filename #name of the created .csv file
		self.target_feature = "" #feature to predict
		self.target_val = "" #target feature's value to keep track of; maps to 1
		self.non_target_val = [] #list of target feature's values that are not kept track of; maps to 0
		self.labels = "" #labels for header of .csv file

	def data_process_sep(self, data_train, data_test, target, target_val, non_target_val, header="Exists"): #for datasets that are split to begin with (i.e. train dataset and test dataset are separate)
		#------ Read each dataset ------#
		#------ Account for Header ------#
		if header == "None":
			self.train_dataset = pd.read_csv(data_train, header=None)
			self.test_dataset = pd.read_csv(data_test, header=None)
		else:
			self.train_dataset = pd.read_csv(data_train)
			self.test_dataset = pd.read_csv(data_test)

		#------ Process data ------#
		self.target_val = target_val
		self.target = target
		self.non_target_val = list(non_target_val)

		if self.target_val != "None":
			#------ Turn target feature into binary ------#
			convert = {}
			for var in self.non_target_val: #if it is not the target variable to be predicted, map to 0
				convert[var] = 0
			convert[self.target_val] = 1 #if target variable, map to 1
			self.train_dataset = self.train_dataset.replace({self.target:convert}).infer_objects()
			self.test_dataset = self.test_dataset.replace({self.target:convert}).infer_objects()

		#----- Separate into X and Y -----#
		self.X_train = self.train_dataset.drop(target, axis=1)
		self.Y_train = self.train_dataset[target]

		self.X_test = self.test_dataset.drop(target, axis=1)
		self.Y_test = self.test_dataset[target]
		print(self.train_dataset.shape, self.test_dataset.shape, self.X_train.shape, self.Y_train.shape, self.X_test.shape, self.Y_test.shape)


		#----- Combine datasets ------#
		combine = [self.train_dataset, self.test_dataset]
		self.dataset = pd.concat(combine)
		self.X_tot = self.dataset.drop(target, axis=1)
		self.Y_tot = self.dataset[target]


	def data_process(self, target, target_val, non_target_val, test_size=0.25, header="Exists"): #Process data as pandas structure, map the last (predict) feature as binary
		print("Processing {}...".format(self.dataname))
		#------ Read each dataset ------#
		#------ Account for Header ------#
		if header == "None":
			self.dataset = pd.read_csv(self.dataname, header=None)
		else:
			self.dataset = pd.read_csv(self.dataname)

		self.target_val = target_val
		self.target = target
		self.non_target_val = list(non_target_val)
		#------ Turn target feature into binary ------#
		convert = {}
		for var in non_target_val: #if it is not the target variable to be predicted, map to 0
			convert[var] = 0
		convert[target_val] = 1 #if target variable, map to 1
		self.dataset = self.dataset.replace({target:convert}).infer_objects()
	
		#------ Separate X and Y -------#
		self.X_tot = self.dataset.drop(self.target, axis=1)
		self.Y_tot = self.dataset[self.target]
		self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X_tot, self.Y_tot, test_size=test_size)

	def create_pipeline(self, PCA, estimator):
		#------ Create Names For Each Step ------#
		PCA_name = str(PCA).split("(")[0]
		step_list = [(PCA_name, PCA)]
		estimator_name = str(estimator).split("(")[0]
		try:
			base_estimator = str(estimator.get_params()["base_estimator"]).split("(")[0]
			estimator_name = estimator_name + "_" + base_estimator
		except:
			estimator_name = estimator_name
		#------ Add to Steps List------#
		step_list.append((estimator_name, estimator))
		pipe = Pipeline(steps=step_list)
		#------ Add to Pipes List ------#
		self.pipes.append(pipe)

	def add_target_params(self, target_params):
		for target in target_params:
			if target not in self.params:
				self.params.append(target)

	def print_pipes(self):
		for pipe in self.pipes:
			self.print_pipe(pipe)

	def print_pipe(self, pipe):
		name = pipe.get_params()['steps'][0][0] + " & " + pipe.get_params()['steps'][1][0]
		print(name)

	def train_pipes(self):
		i = 1
		for pipe in self.pipes:
			print("\nStarting...{} ({}%)".format(i, round(float(i)/float(len(self.pipes)), 2)))
			self.print_pipe(pipe)
			#------ Increment Index ------#
			i += 1

			#------ Pipeline Only ------#
			print("Fitting pipeline...")
			start = time.process_time()
			try:
				pipe.fit(self.X_train, self.Y_train)
				end = time.process_time()
				y_pred = pipe.predict(self.X_test)
				a_score = accuracy_score(self.Y_test, y_pred)
				p_score = precision_score(self.Y_test, y_pred)
				r_score = recall_score(self.Y_test, y_pred)
				fit_time = end - start
			except ValueError as error:
				print(error)
	
			#------ Cross Validation ------#
			print("Cross-validating on train data...")
			start = time.process_time()
			try:
				y_pred_cv = cross_val_predict(pipe, self.X_train, self.Y_train, cv = 5)
				end = time.process_time()
				a_score_cv = accuracy_score(self.Y_train, y_pred_cv)
				p_score_cv = precision_score(self.Y_train, y_pred_cv)
				r_score_cv = recall_score(self.Y_train, y_pred_cv)
				cv_time = end - start
			except ValueError as error:
				print(error)
	
			#------ CV On Whole Dataset ------#
			print("Cross-validating on the whole dataset...")
			pipe.fit(self.X_tot, self.Y_tot)
			start = time.process_time()
			try:
				y_pred_tot = cross_val_predict(pipe, self.X_tot, self.Y_tot, cv = 5)
				end = time.process_time()
				a_score_tot = accuracy_score(self.Y_tot, y_pred_tot)
				p_score_tot = precision_score(self.Y_tot, y_pred_tot)
				r_score_tot = recall_score(self.Y_tot, y_pred_tot)
				cv_tot_time = end - start
			except ValueError as error:
				print(error)
	
			#------ Calculate ROC & AUC ------#
			print("Caculating ROC and AUC...")
			try:
				fpr, tpr, threshold = roc_curve(self.Y_test, pipe.predict_proba(self.X_test)[:,0])
				fpr_cv, tpr_cv, threshold_cv = roc_curve(self.Y_train, cross_val_predict(pipe, self.X_train, self.Y_train, cv = 5, method='predict_proba')[:,0])
				fpr_tot, tpr_tot, threshold_tot = roc_curve(self.Y_tot, cross_val_predict(pipe, self.X_tot, self.Y_tot, cv = 5, method='predict_proba')[:,0])
				auc_roc = auc(fpr, tpr)
				auc_roc_cv = auc(fpr_cv, tpr_cv)
				auc_roc_tot = auc(fpr_tot, tpr_tot)
			except:
				auc_roc = "No attribute 'predict_proba'"
				auc_roc_cv = "No attribute 'predict_proba'"
				auc_roc_tot = "No attribute 'predict_proba'"
			
			#------ Create Result String ------#
			results_fit = self.result(pipe, 'PIPELINE', fit_time, a_score, p_score, r_score, auc_roc)
			results_cv = self.result(pipe, 'CV', cv_time, a_score_cv, p_score_cv, r_score_cv, auc_roc_cv)
			results_tot = self.result(pipe, 'CV_TOT', cv_tot_time, a_score_tot, p_score_tot, r_score_tot, auc_roc_tot)

			#------ Add Projection Step to Results ------#
			results = pipe.get_params()['steps'][1][0] + ", "

			results = results + results_fit + '\n' + " ," + results_cv + '\n'+ " ," + results_tot + '\n'

			#------ Write results to file ------#
			self.write_to_file(results)

	def result(self, pipe, name, fit_time, a_score, p_score, r_score, auc_roc): #given all the metrics
		#Name (Pipeline), base_estimator, target arguments, metrics

		result_str = name + ", "
		pipe_params = pipe.get_params()

		#------ Add Base Estimator ------#
		be_key = estimator = pipe_params['steps'][1][0] + "__" + "base_estimator"
		try:
			result_str = result_str + str(pipe_params[be_key]).split("(")[0] + ", "
		except:
			result_str = result_str + "n/a, "

		#------ Add Target Arguments -------#
		for target in self.params:
			try:
				result_str = result_str + str(pipe_params[target]) + ", "
			except:
				result_str = result_str + "n/a, "

		#------ Add Metrics ------#
		try:
			result_str = result_str + str(fit_time[0]) + ", " 
		except:
			result_str = result_str + str(fit_time) + ", " 

		result_str = result_str + str(a_score) + ", " + str(p_score) + ", " + str(r_score) + ", " + str(auc_roc)
		return result_str

	def create_header(self): #create header for the csv output file
		#Name (Pipeline), PCA, Estimator, base_estimator, target arguments, metrics
		self.labels = "Combination" + ", Mode" + ", base_estimator"
		for target in self.params:
			self.labels = self.labels + ", " + target
		self.labels = self.labels + ", Time" + ", Accuracy Score" + ", Precision Score" + ", Recall Score" + ", AUC\n"

	def write_to_file(self, result):
		print("Writing to File...")
		with open(self.filename, 'a') as file:
			file.write(result)

	def create_file(self):
		self.create_header()
		print("Creating File {}".format(self.filename))
		with open(self.filename, 'w') as file:
			file.write(self.labels)



