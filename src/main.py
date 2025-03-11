import json
import sys
import numpy as np
import pickle as pk
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import time
import tensorflow as tf
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input

from utils import *
from FACEGroup import *
from kernel import *
from dataLoader import *


def get_FACEGroup_Directory():
    """Get the path of the 'FACEGroup-Feasible-and-Actionable-Counterfactual-Explanations-for-Group-Fairness-Auditing' directory."""
    current_dir = os.getcwd()
    target_dir = 'FACEGroup-Feasible-and-Actionable-Counterfactual-Explanations-for-Group-Fairness-Auditing'
    
    while os.path.basename(current_dir) != target_dir:
        current_dir = os.path.dirname(current_dir)
        if current_dir == os.path.dirname(current_dir):
            return None
        
    return current_dir

def get_path_separator():
    """Get the system-specific directory separator."""
    return os.sep

FACEGroup_DIR = get_FACEGroup_Directory()
sys.path.append(FACEGroup_DIR)
sep = get_path_separator()

def initialize_FACEGroup(epsilon=3,datasetName='Student',
	group_identifier='sex', classifier="lr", bandwith_approch="mean_scotts_rule", group_identifier_value=None, 
	skip_model_training=True, skip_distance_calculation=True, skip_graph_creation=True,
	skip_bandwith_calculation=True, representation=64, verbose=True):
	"""
	Initialize the FACEGroup algorithm
	
	# Parameters:
	----------------
	- epsilon: (float)
		margin for creating connections in graphs
	- datasetName: (str)
		name of the dataset
	- group_identifier: (str)
		the column name of the group identifier
	- classifier: (str)
		the classifier to use for the FACEGroup algorithm
	- bandwith_approch: (str)
		the method to use for calculating the bandwith
	- group_identifier_value: (int)
		the value of the group identifier
	- skip_model_training: (boolean)
		load model or not
	- skip_distance_calculation: (boolean)
		load distances or not
	- skip_graph_creation: (boolean)	
		load graph or not
	- skip_bandwith_calculation: (boolean)
		load bandwith or not
	- roundb: (int)
		the number of decimal places to round calculated distances
	
	# Returns:
	----------------
	- facegroup: (FACEGroup object)
		the initialized FACEGroup object
	- graph: (networkx graph)
		the graph object
	- distances: (dict)
		the pairwise distances
	- data: (pandas DataFrame)
		the data
	- data_np: (numpy array)
		the data points
	- data_df_copy: (pandas DataFrame)
		the copy of the original data
	- attr_col_mapping: (dict)
		the mapping of the column names to their indices
	- normalized_group_identifer_value: (float)
		the normalized value of the group identifier
	- numeric_columns: (list)
		the list of numeric columns
	- candidate_counterfactuals: (list)
		the list of candidate counterfactuals
	- Factuals: (list)
		the list of false negatives
	- Factuals_by_group: (dict)
		the false negatives by group
	- node_connectivity: (float)
		the node connectivity
	- edge_connectivity: (float)
		the edge connectivity
	"""
	data, FEATURE_COLUMNS, TARGET_COLUMNS, numeric_columns, categorical_columns, min_max_scaler, data_df_copy, continuous_featues, one_hot_encode_features = load_dataset(datasetName=datasetName)
	
	if 'GermanCredit' in datasetName:
		datasetName = 'GermanCredit'
	TEST_SIZE = 0.3

	X_train, X_test, y_train, y_test = train_test_split(
		data[FEATURE_COLUMNS],
		data[TARGET_COLUMNS],
		test_size=TEST_SIZE,
		random_state=utils.random_seed,
		shuffle=True
	)
	if verbose:
		print("Data shape:", data.shape)
		print("Feature columns:", data.columns)
		print("Target columns:", TARGET_COLUMNS)

	if not os.path.exists(f"{FACEGroup_DIR}{sep}tmp"):
		os.makedirs(os.path.join(FACEGroup_DIR, 'tmp'))

	if not os.path.exists(f"{FACEGroup_DIR}{sep}tmp{sep}{datasetName}"):
		os.makedirs(f"{FACEGroup_DIR}{sep}tmp{sep}{datasetName}")

	train_model = None
	param_grid = None
	model = None
	if classifier == "lr":
		if skip_model_training and "LR_classifier_data.pk" in os.listdir(f"{FACEGroup_DIR}{sep}tmp{sep}{datasetName}{sep}"):
			if verbose:
				print("Loading classifier from file ...")
			model = pk.load(open(f"{FACEGroup_DIR}{sep}tmp{sep}{datasetName}{sep}LR_classifier_data.pk", "rb"))
		else:
			param_grid = {
				'C': [0.001, 0.01, 0.1, 1, 5, 10, 50, 100, 200], 
				'solver': ['newton-cg', 'lbfgs', 'liblinear']
			}
			model = LogisticRegression(max_iter=10000)
			train_model = 'lr'
	elif classifier == "xgb":
		if skip_model_training and "XGB_classifier_data.pk" in os.listdir(f"{FACEGroup_DIR}{sep}tmp{sep}{datasetName}{sep}"):
			if verbose:
				print("Loading classifier from file ...")
			model = pk.load(open(f"{FACEGroup_DIR}{sep}tmp{sep}{datasetName}{sep}XGB_classifier_data.pk", "rb"))
		else:
			param_grid = {
			'n_estimators': [50, 100, 200, 500],
			'max_depth': [3, 5, 7, 10, 15],
			'learning_rate': [0.01, 0.05, 0.1, 0.2],
			'subsample': [0.5, 0.7, 0.9, 1],
			'colsample_bytree': [0.5, 0.7, 0.9, 1],
			'gamma': [0, 0.1, 0.5, 1, 5],
			'reg_alpha': [0, 0.01, 0.1, 1],
			'reg_lambda': [1, 5, 10],
			}

			model = xgb.XGBClassifier(
				objective="binary:logistic",
				eval_metric="logloss"
			)
			train_model = 'xgb'
	elif classifier == "rf":
		if skip_model_training and "RF_classifier_data.pk" in os.listdir(f"{FACEGroup_DIR}{sep}tmp{sep}{datasetName}{sep}"):
			if verbose:
				print("Loading classifier from file ...")
			model = pk.load(open(f"{FACEGroup_DIR}{sep}tmp{sep}{datasetName}{sep}RF_classifier_data.pk", "rb"))
		else:
			param_grid = {
			'n_estimators': [100, 200, 300, 400],
			'max_depth': [None, 10, 20, 30],
			'min_samples_split': [2, 5, 10],
			'min_samples_leaf': [1, 2, 4],
			'bootstrap': [True, False],
			}
			model = RandomForestClassifier(random_state=42)
			train_model = 'rf'
	elif classifier == "dnn":
		if skip_model_training and "DNN_classifier_data.h5" in os.listdir(f"{FACEGroup_DIR}{sep}tmp{sep}{datasetName}{sep}"):
			if verbose:
				print("Loading classifier from file ...")
			model = tf.keras.models.load_model(f"{FACEGroup_DIR}{sep}tmp{sep}{datasetName}{sep}DNN_classifier_data.keras")
		else:
			def create_dnn_model(optimizer='adam', dropout_rate=0.5, hidden_units=32):
				model = Sequential()
				model.add(Input(shape=(X_train.shape[1],)))
				model.add(Dense(hidden_units, activation='relu'))
				model.add(Dropout(dropout_rate))
				model.add(Dense(hidden_units // 2, activation='relu'))
				model.add(Dense(1, activation='sigmoid'))
				model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
				return model
			model = KerasClassifier(
				model=create_dnn_model,
				verbose=0,
				epochs=10,
				batch_size=32
			)
			param_grid = {
				'model__optimizer': ['adam', 'rmsprop'],
				'model__dropout_rate': [0.3, 0.5],
				'model__hidden_units': [32, 64],
				'batch_size': [8, 16],
				'epochs': [5, 10]
			}
			train_model = 'dnn'
	else:
		raise ValueError("Invalid classifier type. Supported types are 'lr', 'xgb', and 'dnn'.")

	if train_model != None:	
		random_search = RandomizedSearchCV(
			estimator=model,
			param_distributions=param_grid,
			n_iter=15, 
			cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
			scoring='balanced_accuracy',
			n_jobs=max(1, (os.cpu_count() or 1) - 5),
			verbose=0,
			random_state=42
		)
		if verbose:
			print(f"Starting {classifier} hyperparameter search...")
		random_search.fit(X_train, y_train)
		model = random_search.best_estimator_

		if verbose:
			print(f"\nBest {classifier} Hyperparameters: {random_search.best_params_}")
			print(f"Best cross-validated accuracy: {random_search.best_score_:.4f}")
			print(f"Training Accuracy: {model.score(X_train, y_train):.4f}")
			print(f"Testing Accuracy: {model.score(X_test, y_test):.4f}")

	if not os.path.exists(f"{FACEGroup_DIR}{sep}tmp{sep}{datasetName}{sep}"):
		os.makedirs(f"{FACEGroup_DIR}{sep}tmp{sep}{datasetName}{sep}")
	if train_model == 'lr':
		pk.dump(model, open(f"{FACEGroup_DIR}{sep}tmp{sep}{datasetName}{sep}LR_classifier_data.pk", 'wb'))
	elif train_model == 'xgb':
		pk.dump(model, open(f"{FACEGroup_DIR}{sep}tmp{sep}{datasetName}{sep}XGB_classifier_data.pk", 'wb'))
	elif train_model == 'rf':
		pk.dump(model, open(f"{FACEGroup_DIR}{sep}tmp{sep}{datasetName}{sep}RF_classifier_data.pk", 'wb'))
	elif train_model == 'dnn':
		model.save(f"{FACEGroup_DIR}{sep}tmp{sep}{datasetName}{sep}DNN_classifier_data.keras")

	start_time = time.time()

	data = data.drop_duplicates()
	data_df_copy = data_df_copy.loc[data.index]
	data = data.reset_index(drop=True)
	data_df_copy = data_df_copy.reset_index(drop=True)
	data_np = data.to_numpy()
	attr_col_mapping = {col: i for i, col in enumerate(data.columns)}
	X = data_np[:, [attr_col_mapping[col] for col in FEATURE_COLUMNS]]

	X_train, X_test, y_train, y_test = train_test_split(
		data[FEATURE_COLUMNS],
		data[TARGET_COLUMNS],
		test_size=TEST_SIZE,
		random_state=utils.random_seed,
		shuffle=True
	)
	candidate_counterfactuals = data[model.predict(data[FEATURE_COLUMNS]) == 1]
	factuals = X_test[model.predict(X_test[FEATURE_COLUMNS]) == 0]
	common_indices = factuals.index.intersection(y_test[y_test == 1].index)
	Factuals = factuals.loc[common_indices]
	if verbose:
		print(f"Positive points: {len(candidate_counterfactuals)}")
		print(f"Factuals: {len(Factuals)}")
	candidate_counterfactuals = {index: row.to_numpy() for index, row in candidate_counterfactuals.iterrows()}
	factuals = {index: row.to_numpy() for index, row in factuals.iterrows()}
	Factuals = {index: row.to_numpy() for index, row in Factuals.iterrows()}

	normalized_group_identifer_value = None
	if group_identifier in numeric_columns and group_identifier_value is not None:
		normalized_group_identifer_value = utils.get_normalized_group_identifier_value(group_identifier, group_identifier_value, min_max_scaler, data_df_copy)
		if verbose:
			print(f"Group identifier value: {group_identifier_value}")
			print(f"Normalized group identifier value: {normalized_group_identifer_value}")
	elif group_identifier in numeric_columns and group_identifier_value is None:
		raise ValueError(f"The group_identifier column {group_identifier} does not contain numerical values")

	Factuals_by_group = utils.get_false_negatives_by_group(Factuals, group_identifier, normalized_group_identifer_value, data, numeric_columns)

	if skip_distance_calculation and skip_graph_creation and\
				os.path.exists(f"{FACEGroup_DIR}{sep}tmp{sep}{datasetName}{sep}Graphs{sep}Graph_{epsilon}.pkl")\
					and os.path.exists(f"{FACEGroup_DIR}{sep}tmp{sep}{datasetName}{sep}Distances.pkl") and os.path.exists(f"{FACEGroup_DIR}{sep}tmp{sep}{datasetName}{sep}Densities{sep}Densities_{epsilon}.pkl"):
		if verbose:
			print("Loading graph from file ...")
		graph = pk.load(open(f"{FACEGroup_DIR}{sep}tmp{sep}{datasetName}{sep}Graphs{sep}Graph_{epsilon}.pkl", "rb"))
		kernel = Kernel(datasetName, X, skip_bandwith_calculation=skip_bandwith_calculation, bandwith_approch=bandwith_approch)
		kernel.fitKernel(X)
		facegroup = FACEGroup(data_np, kernel, FEATURE_COLUMNS, TARGET_COLUMNS, epsilon, model)
		feasibility_constraints = utils.getFeasibilityConstraints(FEATURE_COLUMNS, dataset_name=datasetName)
		facegroup.set_graph(graph)
		if verbose:
			print("Loading distances from file ...")
		distances = pk.load(open(f"{FACEGroup_DIR}{sep}tmp{sep}{datasetName}{sep}Distances.pkl", "rb"))
		if verbose:
				print(f"Max distance in the dataset: {np.sqrt(len(FEATURE_COLUMNS))}")
				print(f"Max possible distance considered in graph: {np.max(distances)}")
	else:
		kernel = Kernel(datasetName, X, skip_bandwith_calculation=skip_bandwith_calculation, bandwith_approch=bandwith_approch)
		kernel.fitKernel(X)
		facegroup = FACEGroup(data_np, kernel, FEATURE_COLUMNS, TARGET_COLUMNS, epsilon, model)
		feasibility_constraints = utils.getFeasibilityConstraints(FEATURE_COLUMNS, dataset_name=datasetName)

		start_time = time.time()
		dng_obj = GraphBuilder(feasibility_constraints, FEATURE_COLUMNS, X, kernel, exclude_columns=True)
		distances, graph, densities = dng_obj.compute_pairwise_distances_within_subgroups_and_graph(datasetName, data[FEATURE_COLUMNS], epsilon, feasibility_constraints, representation)
		facegroup.set_graph(graph)
		end_time = time.time()
		execution_time = end_time - start_time
		if verbose:		
			print("Distances and graph initialization: ", execution_time, " seconds")

		if not os.path.exists(f"{FACEGroup_DIR}{sep}tmp{sep}{datasetName}{sep}Graphs{sep}"):
			os.makedirs(f"{FACEGroup_DIR}{sep}tmp{sep}{datasetName}{sep}Graphs{sep}")
		if not os.path.exists(f"{FACEGroup_DIR}{sep}tmp{sep}{datasetName}{sep}Densities{sep}"):
			os.makedirs(f"{FACEGroup_DIR}{sep}tmp{sep}{datasetName}{sep}Densities")

		pk.dump(graph, open(f"{FACEGroup_DIR}{sep}tmp{sep}{datasetName}{sep}Graphs{sep}Graph_{epsilon}.pkl", "wb"))
		pk.dump(distances, open(f"{FACEGroup_DIR}{sep}tmp{sep}{datasetName}{sep}Distances.pkl", "wb"))
		pk.dump(densities, open(f"{FACEGroup_DIR}{sep}tmp{sep}{datasetName}{sep}Densities{sep}Densities_{epsilon}.pkl", "wb"))
		if verbose:
			print(f"Max distance in the dataset: {np.sqrt(len(FEATURE_COLUMNS))}")
			print(f"Max possible distance considered in graph: {np.max(distances)}")
	
	fully_connected_nodes = len(X)
	singleton_nodes = [node for node, degree in graph.degree() if degree == 0]
	connected_nodes = graph.nodes()-singleton_nodes
	node_connectivity = len(graph.nodes()-singleton_nodes) / len(graph.nodes()) * 100
	density = nx.density(graph) * 100
	end_time = time.time()
	execution_time = end_time - start_time
	if verbose:
		print(f"{len(connected_nodes)} nodes are connected out of {fully_connected_nodes} nodes. Connectivity: {node_connectivity}%")
		print(f"Density: {density}%")
		print("FACEGroup initialization: ", execution_time, " seconds")

	return facegroup, graph, distances, data, data_np, data_df_copy, attr_col_mapping, normalized_group_identifer_value,\
		numeric_columns, candidate_counterfactuals, Factuals, Factuals_by_group, node_connectivity, density, feasibility_constraints


# =====================================================================================================================
# =====================================================================================================================
#                 		 					cost-constrained group counterfactuals
# =====================================================================================================================
# =====================================================================================================================
def main_cost_constrained_GCFEs(epsilon=3, datasetName='Student',
					group_identifier='sex', classifier="lr", bandwith_approch="mean_scotts_rule",
					k=5, max_d = 1, cost_function = "max_vector_distance", k_selection_method="accross_all_ccs",
					group_identifier_value=None, skip_model_training=True, skip_distance_calculation=True, skip_graph_creation=True,
					skip_bandwith_calculation=True,  skip_FACEGroup_calculation=True, compare_with_Face=False, representation=64,
     				fgce_init_dict=None, verbose=False, cfe_selection_method='greedy'):
	"""
	This function is used to solve the cost-constrained group counterfactuals problem using the greedy coverage algorithm

	# Parameters:
	----------------
	- epsilon: (float)
		margin for creating connections in graphs
	- datasetName: (str)
		name of the dataset
	- group_identifier: (str)
		the column name of the group identifier
	- classifier: (str)
		the classifier to use for the FACEGroup algorithm
	- k: (int)
		maximum number of cfes to return for each group
	- max_d: (int)
		maximum path cost for reachability (d-reachable threshold)
	- cost_function: (str)
		the cost function to use for the FACEGroup algorithm
	- k_selection_method: (str)
		the method to use for selecting the k cfes
	- group_identifier_value: (int)
		the value of the group identifier
	- skip_model_training: (boolean)
		if it will skip the model training and load it if it exists or not
	- skip_distance_calculation: (boolean)
		if it will skip the distance calculation and load it if it exists or not
	- skip_graph_creation: (boolean)
		if it will skip the graph creation and load it if it exists or not
	- skip_bandwith_calculation: (boolean)
		if it will skip the bandwith calculation and load it if it exists or not
	- skip_FACEGroup_calculation: (boolean)
		if it will skip the group cfes calculation and load it if it exists or not
	- compare_with_Face: (boolean)
		if it will compare the results with the Face algorithm or not
	- roundb: (int)
		the number of decimal places to round calculated distances
	

	# Returns:
	----------------
	- results: (dict)
		dictionary containing the final results of the FACEGroup-Group algorithm
	- data_np: (numpy array)
		the data points
	- attr_col_mapping: (dict)
		the mapping of the column names to their indices
	- data_df_copy: (pandas DataFrame)
		the copy of the original data
	In case of comparing with the Face algorithm, the following values are returned:
	- face_vector_distances: (float)
		the average vector distance of the Face algorithm
	- gfce_vector_distances: (float)
		the average vector distance of the FACEGroup algorithm
	- face_wij_distances: (float)
		the average path cost of the Face algorithm
	- gfce_wij_distances: (float)
		the average path cost of the FACEGroup algorithm
	"""
	if fgce_init_dict:
		facegroup, graph, distances, data, data_np, data_df_copy, attr_col_mapping, normalized_group_identifer_value, numeric_columns, candidate_counterfactuals,\
			  Factuals, Factuals_by_group, node_connectivity, edge_connectivity, feasibility_constraints  = fgce_init_dict["facegroup"],\
			  fgce_init_dict["graph"], fgce_init_dict["distances"], fgce_init_dict["data"],\
			  fgce_init_dict["data_np"], fgce_init_dict["data_df_copy"], fgce_init_dict["attr_col_mapping"],\
			  fgce_init_dict["normalized_group_identifer_value"], fgce_init_dict["numeric_columns"], fgce_init_dict["candidate_counterfactuals"],\
			  fgce_init_dict["Factuals"], fgce_init_dict["Factuals_by_group"], fgce_init_dict["node_connectivity"],\
			  fgce_init_dict["edge_connectivity"], fgce_init_dict["feasibility_constraints"]
	else:
		facegroup, graph, distances, data, data_np, data_df_copy, attr_col_mapping, normalized_group_identifer_value, numeric_columns, candidate_counterfactuals,\
			  Factuals, Factuals_by_group, node_connectivity, edge_connectivity, feasibility_constraints  = initialize_FACEGroup(epsilon, datasetName, group_identifier, classifier, bandwith_approch, group_identifier_value, 
					skip_model_training, skip_distance_calculation, skip_graph_creation, skip_bandwith_calculation, representation, verbose=verbose)
	# =========================================================================================================================
	# 												GROUP CFES
	# =========================================================================================================================
	results = {}

	if cost_function == "max_vector_distance":
		max_d_store = round(max_d, 2)	
	elif cost_function == "max_path_cost":
		max_d_store = f"{max_d:.2e}"
	else:
		max_d_store = max_d
		
	if not os.path.exists(f"{FACEGroup_DIR}{sep}tmp{sep}{datasetName}{sep}figs"):
		os.makedirs(f"{FACEGroup_DIR}{sep}tmp{sep}{datasetName}{sep}figs")

	file_path = f"{FACEGroup_DIR}{sep}tmp{sep}{datasetName}{sep}cost_constrained_GCFEs{sep}results_{datasetName}_eps{epsilon}_k_{k}_cost_function_{cost_function}_d_{max_d_store}_kmethod_{k_selection_method}.json"
	if skip_FACEGroup_calculation and os.path.exists(file_path):
		results = json.load(open(file_path, "r"))
		return results, data, data_np, attr_col_mapping, data_df_copy, [], [], 0, 0
	else:
		start_time = time.time()
		subgroups = utils.get_subgraphs_by_group(graph, data_np, data, attr_col_mapping, group_identifier, normalized_group_identifer_value, numeric_columns)
		print(f"\n{len(subgroups)} subgroups created based on group identifier: {group_identifier}")
		
		if verbose:
			print(f"Computing group cfes...")
		gcfes, not_possible_to_cover_fns_group, time_gcfes = facegroup.compute_gcfes(subgroups, candidate_counterfactuals, Factuals, max_d, cost_function, k, distances, k_selection_method, verbose=verbose, cfe_selection_method=cfe_selection_method)

		stats = {}
		stats["Node Connectivity"] = node_connectivity
		stats["Edge Connectivity"] = edge_connectivity

		results = facegroup.apply_cfes(gcfes, Factuals_by_group, distances, not_possible_to_cover_fns_group, k_selection_method, cost_function, stats, verbose=verbose)

		# ensure all keys are strings of the results dict to avoid such errors: "TypeError: keys must be str, int, float, bool or None, not int64"
		results = {str(key): value for key, value in results.items()}

		## do the same to the dubdicts too
		for key in results:
			if key in ["Node Connectivity", "Edge Connectivity", "Total coverage", "Graph Stats"]: continue
			results[key] = {str(k): v for k, v in results[key].items()}

		end_time = time.time()
		execution_time = end_time - start_time
		results['Time'] = time_gcfes
		if verbose:
			print("Group Cfes - Time:", execution_time, "seconds")

		if not os.path.exists(f"{FACEGroup_DIR}{sep}tmp{sep}{datasetName}{sep}cost_constrained_GCFEs"):
			os.makedirs(f"{FACEGroup_DIR}{sep}tmp{sep}{datasetName}{sep}cost_constrained_GCFEs")

		try:
			with open(file_path, "w") as outfile:
				json.dump(results, outfile)
		except Exception as e:
			if verbose:
				print("Error saving results:", e)
				print("Trying to serialize the results...")
			results_serializable = serialize_json(results)
			with open(file_path, "w") as outfile:
				json.dump(results_serializable, outfile)
    # =========================================================================================================================
    # 											COMPARE WITH INDIVIDUAL CFES FROM FACE
    # =========================================================================================================================
	if compare_with_Face:
		facegroup.set_candidates(candidate_counterfactuals)

		face_vector_distances = 0
		face_wij_distances = 0
		face_max_wij_distance = 0
		face_max_vector_distance = 0

		gfce_vector_distances = 0
		gfce_wij_distances = 0
		gfce_max_wij_distance = 0
		gfce_max_vector_distance = 0
		factuals_found = 0

		## create individual recourses for each Fn point for each group
		if k_selection_method == "accross_all_ccs":
			for key in results:
				if key in ["Node Connectivity", "Edge Connectivity", "Total coverage", "Graph Stats"]: continue

				for factual in results[key]:
					if factual in ["Coverage", "Avg. distance", "Median distance", "Avg. path cost", "Median path cost"]: continue
					
					factuals_found += 1
					shortest_paths_info, min_target_id = facegroup.compute_recourse(graph, int(factual), feasibility_constraints, True)
					
					print("* Factual Point:", factual)

					if shortest_paths_info is not None and min_target_id is not None and shortest_paths_info != {}:
						print("	Face Info: ")
						shortest_path_info = shortest_paths_info[min_target_id]
						
						print(f"		CFE: {min_target_id}")
						print(f"		Path cost: {shortest_path_info['path_cost']}")
						print(f"		Path cost dist: {shortest_path_info['path_cost_dist']}")
						print(f"		Vector Distance: {distances[int(factual)][min_target_id]}")
						fvd = distances[int(factual)][min_target_id]
						face_vector_distances += fvd	

						if fvd > face_max_vector_distance:
							face_max_vector_distance = fvd
						fwd = shortest_path_info['path_cost']

						if fwd > face_max_wij_distance:
							face_max_wij_distance = fwd
						face_wij_distances += fwd
					
					if verbose:
						print("	FACEGroup Info:")
						print(f"		CFE: {results[key][factual]['CFE_name']}")
						print(f"		Shortest Path cost: {results[key][factual]['Shortest_path_cost']}")
						print(f"		Shortest Path cost dist: {results[key][factual]['Shortest_paths_distance_cost']}")
						print(f"		Vector Distance: {results[key][factual]['Vector_distance']}")
					gfvd = results[key][factual]['Vector_distance']
					gfce_vector_distances += gfvd

					if gfvd > gfce_max_vector_distance:
						gfce_max_vector_distance = gfvd

					gfwd = results[key][factual]['Shortest_path_cost']
					if gfwd > gfce_max_wij_distance:
						gfce_max_wij_distance = gfwd
					gfce_wij_distances += gfwd

		if factuals_found != 0:
			face_wij_distances = (face_wij_distances / face_max_wij_distance) / factuals_found
			gfce_wij_distances = (gfce_wij_distances / gfce_max_wij_distance) /factuals_found

			face_vector_distances = (face_vector_distances / face_max_vector_distance)  / factuals_found
			gfce_vector_distances = (gfce_vector_distances / gfce_max_vector_distance) / factuals_found
		else:
			return results, data, data_np, attr_col_mapping, data_df_copy, None, None, 0, 0

		return results, data, data_np, attr_col_mapping, data_df_copy, face_vector_distances, gfce_vector_distances, face_wij_distances, gfce_wij_distances
	return results, data, data_np, attr_col_mapping, data_df_copy, [], [], 0, 0


# =====================================================================================================================
# =====================================================================================================================
#                 		 					coverage-constrained group counterfactuals
# =====================================================================================================================
# =====================================================================================================================
def main_coverage_constrained_GCFEs(epsilon=0.2, datasetName='Student', group_identifier='sex',
					classifier="lr", cost_function="max_path_cost", k=2,
					min_d=0, max_d=2, bst=1e-3, bandwith_approch="mean_scotts_rule",
					group_identifier_value=None, skip_model_training=True, skip_distance_calculation=True, skip_graph_creation=True,
					compare_with_Face=False, skip_FACEGroup_calculation=False,  skip_bandwith_calculation=True, find_k0=True, representation=64, fgce_init_dict=None):
	"""
	This function is used to solve the coverage-constrained group counterfactuals problem using binary search

	# Parameters:
	----------------
	- epsilon: (float)
		margin for creating connections in graphs
	- datasetName: (str)
		name of the dataset
	- group_identifier: (str)
		the column name of the group identifier
	- classifier: (str)
		the classifier to use for the FACEGroup algorithm
	- k: (int)
		maximum number of cfes to return for each group
	- cost_function: (str)
		the cost function to use for the FACEGroup algorithm
	- group_identifier_value: (int)
		the value of the group identifier
	- skip_model_training: (boolean)
		if it will skip the model training and load it if it exists or not
	- skip_distance_calculation: (boolean)
		if it will skip the distance calculation and load it if it exists or not
	- skip_graph_creation: (boolean)
		if it will skip the graph creation and load it if it exists or not
	- skip_bandwith_calculation: (boolean)
		if it will skip the bandwith calculation and load it if it exists or not
	- skip_FACEGroup_calculation: (boolean)
		if it will skip the group cfes calculation and load it if it exists or not
	- compare_with_Face: (boolean)
		if it will compare the results with the Face algorithm or not
	- find_k0: (boolean)
		if it will find the optimal k0 or not
	- roundb: (int)
		the number of decimal places to round calculated distances

	# Returns:
	----------------
	- results: (dict)
		dictionary containing the final results of the FACEGroup-Group algorithm
	- data_np: (numpy array)
		the data points
	- attr_col_mapping: (dict)
		the mapping of the column names to their indices
	- data_df_copy: (pandas DataFrame)
		the copy of the original data
	In case of comparing with the Face algorithm, the following values are returned:
	- face_vector_distances: (float)
		the average vector distance of the Face algorithm
	- gfce_vector_distances: (float)
		the average vector distance of the FACEGroup algorithm
	- face_wij_distances: (float)
		the average path cost of the Face algorithm
	- gfce_wij_distances: (float)
		the average path cost of the FACEGroup algorithm
	"""
	if fgce_init_dict:
		facegroup, graph, distances, data, data_np, data_df_copy, attr_col_mapping, normalized_group_identifer_value, numeric_columns, candidate_counterfactuals,\
			  Factuals, Factuals_by_group, node_connectivity, edge_connectivity, feasibility_constraints = fgce_init_dict["facegroup"],\
			  fgce_init_dict["graph"], fgce_init_dict["distances"], fgce_init_dict["data"],\
			  fgce_init_dict["data_np"], fgce_init_dict["data_df_copy"], fgce_init_dict["attr_col_mapping"],\
			  fgce_init_dict["normalized_group_identifer_value"], fgce_init_dict["numeric_columns"], fgce_init_dict["candidate_counterfactuals"],\
			  fgce_init_dict["Factuals"], fgce_init_dict["Factuals_by_group"], fgce_init_dict["node_connectivity"],\
			  fgce_init_dict["edge_connectivity"], fgce_init_dict["feasibility_constraints"]
	else:
		facegroup, graph, distances, data, data_np, data_df_copy, attr_col_mapping, normalized_group_identifer_value, numeric_columns, candidate_counterfactuals,\
			  Factuals, Factuals_by_group, node_connectivity, edge_connectivity, feasibility_constraints  = initialize_FACEGroup(epsilon, datasetName, group_identifier, classifier, bandwith_approch, group_identifier_value, 
					skip_model_training, skip_distance_calculation, skip_graph_creation, skip_bandwith_calculation, representation)
	# =========================================================================================================================
	# 												GROUP CFES
	# =========================================================================================================================
	results = {}

	if cost_function == "max_vector_distance":
		max_d_store = round(max_d, 2)
		min_d_store = round(min_d, 2)
		bst_store = round(bst, 2)
	elif cost_function == "max_path_cost":
		max_d_store = f"{max_d:.2e}"
		min_d_store = f"{min_d:.2e}"
		bst_store = f"{bst:.2e}"
	else:
		max_d_store = max_d
		min_d_store = min_d
		bst_store = bst
	
	if not os.path.exists(f"{FACEGroup_DIR}{sep}tmp{sep}{datasetName}{sep}figs"):
		os.makedirs(f"{FACEGroup_DIR}{sep}tmp{sep}{datasetName}{sep}figs")

	file_path = f"{FACEGroup_DIR}{sep}tmp{sep}{datasetName}{sep}coverage_constrained_GCFEs{sep}results_{datasetName}_eps{epsilon}_k_{k}_cost_function_{cost_function}_max_d_store_{max_d_store}_min_d_store_{min_d_store}_bst_{bst_store}.json"
	
	if skip_FACEGroup_calculation and os.path.exists(file_path):
		results = json.load(open(file_path, "r"))
		return results, data, data_np, attr_col_mapping, data_df_copy, [], [], 0, 0

	else:
		start_time = time.time()
		print(f"Computing group cfes...")

		subgroups = utils.get_subgraphs_by_group(graph, data_np, data, attr_col_mapping, group_identifier, normalized_group_identifer_value, numeric_columns)
		print(f"\n{len(subgroups)} subgroups created based on group identifier: {group_identifier}")

		gcfes, gcfes_with_ccs, not_possible_to_cover_fns_group = facegroup.compute_gcfes_binary(subgroups, candidate_counterfactuals, Factuals, k, min_d, max_d, bst, cost_function, distances, find_k0)

		stats = {}
		stats["Node Connectivity"] = node_connectivity
		stats["Edge Connectivity"] = edge_connectivity
		stats['Optimal d'] = {}

		for group in gcfes_with_ccs:
			d0s = []
			for cc in gcfes_with_ccs[group]:
				d0s.append(gcfes_with_ccs[group][cc]['optimal_d'])
			stats['Optimal d'][group] = np.max(d0s)
		
		for group in stats['Optimal d']:
			print(f"Group: {group} - Optimal d0: {np.round(stats['Optimal d'][group], 2)}")
		
		stats = facegroup.apply_cfes(gcfes, Factuals_by_group, distances, not_possible_to_cover_fns_group, "same_k_for_all_ccs", cost_function, stats, binary_implementation=True)

		end_time = time.time()
		execution_time = end_time - start_time
		print("Group Cfes - Time:", execution_time, "seconds")

		if not os.path.exists(f"{FACEGroup_DIR}{sep}tmp{sep}{datasetName}{sep}coverage_constrained_GCFEs"):
			os.makedirs(f"{FACEGroup_DIR}{sep}tmp{sep}{datasetName}{sep}coverage_constrained_GCFEs")

		try:
			with open(file_path, "w") as outfile:
				json.dump(results, outfile)
		except Exception as e:
			print("Error saving results:", e)
			print("Trying to serialize the results...")
			results_serializable = serialize_json(results)
			with open(file_path, "w") as outfile:
				json.dump(results_serializable, outfile)
	# =========================================================================================================================
    # 											COMPARE WITH INDIVIDUAL CFES FROM FACE
    # =========================================================================================================================
	if compare_with_Face:
		facegroup.set_candidates(candidate_counterfactuals)

		face_vector_distances = 0
		face_wij_distances = 0
		face_max_wij_distance = 0
		face_max_vector_distance = 0

		gfce_vector_distances = 0
		gfce_wij_distances = 0
		gfce_max_wij_distance = 0
		gfce_max_vector_distance = 0
		factuals_found = 0

		## create individual recourses for each Fn point for each group
		for key in stats:
			if isinstance(key, str):
				if key in ["Node Connectivity", "Edge Connectivity", "Total coverage", "Graph Stats"] or "Optimal d" in key:
					continue

			for factual in stats[key]:
				if factual in ["Coverage", "Avg. distance", "Median distance", "Avg. path cost", "Median path cost"]: continue

				for subkey in stats[key][factual]:
					if subkey != "CFE_name": continue
					cfe_name = stats[key][factual][subkey]

					if factual in ["Coverage", "Avg. distance", "Median distance", "Avg. path cost", "Median path cost"]: continue
					factuals_found += 1
					shortest_paths_info, min_target_id = facegroup.compute_recourse(graph, int(factual), feasibility_constraints, True)
					
					print("* Factual Point:", factual)

					if shortest_paths_info is not None and min_target_id is not None and shortest_paths_info != {}:
						print("	Face Info: ")
						shortest_path_info = shortest_paths_info[min_target_id]
						
						print(f"		CFE: {min_target_id}")
						fwd = shortest_path_info['path_cost']
						print(f"		Path cost: {fwd}")
						print(f"		Path cost dist: {shortest_path_info['path_cost_dist']}")
						fvd = distances[int(factual)][min_target_id]
						print(f"		Vector Distance: {fvd}")
						
						face_vector_distances += fvd	

						if fvd > face_max_vector_distance:
							face_max_vector_distance = fvd

						if fwd > face_max_wij_distance:
							face_max_wij_distance = fwd
						face_wij_distances += fwd
					
					print("	Face-Group Info:")
					print(f"		CFE: {cfe_name}")
					gfwd = stats[key][factual]['Shortest_path_cost']
					print(f"		Shortest Path cost: {gfwd}")
					print(f"		Shortest Path cost dist: {stats[key][factual]['Shortest_paths_distance_cost']}")
					gfvd = stats[key][factual]['Vector_distance']
					print(f"		Vector Distance: {gfvd}")
					
					gfce_vector_distances += gfvd

					if gfvd > gfce_max_vector_distance:
						gfce_max_vector_distance = gfvd

					if gfwd > gfce_max_wij_distance:
						gfce_max_wij_distance = gfwd
					gfce_wij_distances += gfwd

		if factuals_found != 0:
			face_wij_distances = (face_wij_distances /face_max_wij_distance) /factuals_found
			gfce_wij_distances = (gfce_wij_distances /gfce_max_wij_distance) /factuals_found

			face_vector_distances = (face_vector_distances /face_max_vector_distance)   /factuals_found
			gfce_vector_distances = (gfce_vector_distances /gfce_max_vector_distance) /factuals_found
		else:
			return stats, data, data_np, attr_col_mapping, data_df_copy, None, None, 0, 0
		return stats, data, data_np, attr_col_mapping, data_df_copy, face_vector_distances, gfce_vector_distances, face_wij_distances, gfce_wij_distances
	return stats, data, data_np, attr_col_mapping, data_df_copy, [], [], 0, 0



# =====================================================================================================================
#                 		 					coverage-constrained group counterfactuals-MIP
# =====================================================================================================================
def main_coverage_constrained_GCFEs_Greedy_MIP(epsilon=3, datasetName='Student', 
					group_identifier='sex', classifier="lr", bandwith_approch="mean_scotts_rule", k=5, cost_function = "max_vector_distance",
					group_identifier_value=None, skip_model_training=True, skip_distance_calculation=True, skip_graph_creation=True,
					skip_fgce_calculation=False,  skip_bandwith_calculation=True, cov_constr_approach="local", cov = 1,  representation=64, fgce_init_dict=None, alg='MIP'):
	"""
	This function is used to solve the coverage-constrained group counterfactuals problem using binary search

	# Parameters:
	----------------
	- epsilon: (float)
		margin for creating connections in graphs
	- datasetName: (str)
		name of the dataset
	- group_identifier: (str)	
		the column name of the group identifier
	- classifier: (str)
		the classifier to use for the FGCE algorithm
	- k: (int)
		maximum number of cfes to return for each group
	- cost_function: (str)
		the cost function to use for the FGCE algorithm
	- group_identifier_value: (int)
		the value of the group identifier
	- skip_model_training: (boolean)
	- skip_distance_calculation: (boolean)
		if it will skip the distance calculation and load it if it exists or not
	- skip_graph_creation: (boolean)
		if it will skip the graph creation and load it if it exists or not
	- skip_bandwith_calculation: (boolean)
		if it will skip the bandwith calculation and load it if it exists or not
	- skip_fgce_calculation: (boolean)
		if it will skip the group cfes calculation and load it if it exists or not
	- cov_constr_approach: (str)
		the approach to use for the coverage constraint
	- roundb: (int)
		the number of decimal places to round calculated distances
	# Returns:
	----------------
	- results: (dict)
		dictionary containing the final results of the FGCE-Group algorithm
	"""
	if fgce_init_dict:
		facegroup, graph, distances, data, data_np, data_df_copy, attr_col_mapping, normalized_group_identifer_value, numeric_columns, candidate_counterfactuals,\
			  Factuals, Factuals_by_group, node_connectivity, edge_connectivity, feasibility_constraints = fgce_init_dict["facegroup"],\
			  fgce_init_dict["graph"], fgce_init_dict["distances"], fgce_init_dict["data"],\
			  fgce_init_dict["data_np"], fgce_init_dict["data_df_copy"], fgce_init_dict["attr_col_mapping"],\
			  fgce_init_dict["normalized_group_identifer_value"], fgce_init_dict["numeric_columns"], fgce_init_dict["candidate_counterfactuals"],\
			  fgce_init_dict["Factuals"], fgce_init_dict["Factuals_by_group"], fgce_init_dict["node_connectivity"],\
			  fgce_init_dict["edge_connectivity"], fgce_init_dict["feasibility_constraints"]
	else:
		facegroup, graph, distances, data, data_np, data_df_copy, attr_col_mapping, normalized_group_identifer_value, numeric_columns, candidate_counterfactuals,\
			  Factuals, Factuals_by_group, node_connectivity, edge_connectivity, feasibility_constraints  = initialize_FACEGroup(epsilon, datasetName, group_identifier, classifier, bandwith_approch, group_identifier_value, 
					skip_model_training, skip_distance_calculation, skip_graph_creation, skip_bandwith_calculation, representation)
	# =========================================================================================================================
	# 												GROUP CFES
	# =========================================================================================================================
	results = {}	
	file_path = f"{FACEGroup_DIR}{sep}tmp{sep}{datasetName}{sep}coverage_constrained_GCFEs{sep}{cov_constr_approach}{sep}stats{sep}results_{datasetName}_eps{epsilon}_k_{k}_cov_{cov}_cost_function_{cost_function}.json"
	gcfes_path = f"{FACEGroup_DIR}{sep}tmp{sep}{datasetName}{sep}coverage_constrained_GCFEs{sep}{cov_constr_approach}{sep}gcfes{sep}gcfes_{datasetName}_eps{epsilon}_k_{k}_cov_{cov}_cost_function_{cost_function}.json"
	os.makedirs(os.path.dirname(file_path), exist_ok=True)
	os.makedirs(os.path.dirname(gcfes_path), exist_ok=True)

	if skip_fgce_calculation and os.path.exists(file_path):
		results = json.load(open(file_path, "r"))
		return results
	else:
		start_time = time.time()
		subgroups = utils.get_subgraphs_by_group(graph, data_np, data, attr_col_mapping, group_identifier, normalized_group_identifer_value, numeric_columns)
		print(f"\n{len(subgroups)} subgroups created based on group identifier: {group_identifier}")		
		print(f"Computing group cfes...")

		if cov_constr_approach == "local":
			gcfes, results, _, _, _, _, max_cost = facegroup.get_gcfes_approach_integer_prog_local(subgroups, distances, candidate_counterfactuals, Factuals) 
			
		elif cov_constr_approach == "global":
			gcfes, results, max_cost = facegroup.get_gcfes_approach_coverage_constrained_global(subgroups, distances, candidate_counterfactuals, Factuals, k, cov, alg)
		
		end_time = time.time()
		if not os.path.exists(f"{FACEGroup_DIR}{sep}tmp{sep}{datasetName}{sep}coverage_constrained_GCFEs"):
			os.makedirs(f"{FACEGroup_DIR}{sep}tmp{sep}{datasetName}{sep}coverage_constrained_GCFEs")

		with open(file_path, "w") as outfile:
			json.dump(results, outfile)
		with open(gcfes_path, "w") as outfile:
			json.dump(gcfes, outfile)
	return results, max_cost