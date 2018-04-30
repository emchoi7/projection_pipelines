from pipelines_generator import *

dataname = 'data/abalone.csv'
filename = 'output/abalone_test_combination.csv'

pipe = pipelines(dataname, filename)
pipe.data_process(0, 'F', ['M', 'I'], header="None")

prj_list = [PCA(), IncrementalPCA(), SparsePCA(), FastICA(), TruncatedSVD(), KMeans()]
alg_list = [RandomForestClassifier(), ExtraTreesClassifier(), GradientBoostingClassifier(), LinearSVC(), SVC(), NuSVC(), SGDClassifier(), KNeighborsClassifier(), RadiusNeighborsClassifier(radius=10)]
#base_list = []Ca

for alg in alg_list:
	alg_name = str(alg).split("(")[0]
	print('\n *---------------- {} ----------------*'.format(alg_name))
	for prj in prj_list:
		pipe.create_pipeline(prj, alg)

target_params = ['PCA__n_components']
pipe.add_target_params(target_params)
pipe.create_file()
pipe.train_pipes()