import sys
import os
import utils
import pickle as pk
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from FACEGroup import *
from main import *
from kernel import *
from dataLoader import *

import xgboost as xgb
from scikeras.wrappers import KerasClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input


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


def initialize_FACEGroup_attributes(datasetName='Student', skip_bandwith_calculation=True, bandwith_approch='optimal', classifier='xgb', skip_model_training=False):
    data, FEATURE_COLUMNS, TARGET_COLUMNS, _, _, \
        _, _, _, _ = load_dataset(datasetName=datasetName)
    if 'GermanCredit' in datasetName:
        datasetName = 'GermanCredit'
    X = data[FEATURE_COLUMNS]
    TEST_SIZE = 0.3

    X_train, X_test, y_train, y_test = train_test_split(
        data[FEATURE_COLUMNS],
        data[TARGET_COLUMNS],
        test_size=TEST_SIZE,
        random_state=utils.random_seed,
        shuffle=True
    )

    if not os.path.exists(f"{FACEGroup_DIR}{sep}tmp"):
        os.makedirs(os.path.join(FACEGroup_DIR, 'tmp'))

    if not os.path.exists(f"{FACEGroup_DIR}{sep}tmp{sep}{datasetName}"):
        os.makedirs(f"{FACEGroup_DIR}{sep}tmp{sep}{datasetName}")

    train_model = None
    param_grid = None
    model = None
    if classifier == "lr":
        if skip_model_training and "LR_classifier_data.pk" in os.listdir(f"{FACEGroup_DIR}{sep}tmp{sep}{datasetName}{sep}"):
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

        print(f"Starting {classifier} hyperparameter search...")
        random_search.fit(X_train, y_train)
        model = random_search.best_estimator_
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
        model.model.save(f"{FACEGroup_DIR}{sep}tmp{sep}{datasetName}{sep}DNN_classifier_data.keras")

    data = data.drop_duplicates()
    data = data.reset_index(drop=True)
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
    positive_points = data[model.predict(data[FEATURE_COLUMNS]) == 1]
    print(f"Positive points: {len(positive_points)}")
    negative_points = X_test[model.predict(X_test[FEATURE_COLUMNS]) == 0]
    common_indices = negative_points.index.intersection(y_test[y_test == 1].index)
    FN = negative_points.loc[common_indices]
    print(f"FN: {len(FN)}")

    kernel = Kernel(datasetName, X, skip_bandwith_calculation=skip_bandwith_calculation, bandwith_approch=bandwith_approch)
    kernel.fitKernel(X)

    return data, data_np, X, FEATURE_COLUMNS, TARGET_COLUMNS, kernel, model

def face_plot(datasetName, face_dists, gfce_dists, face_wij, gfce_wij, d_method, max_d, k_values, x_size, y_size, tick_params_size,
              ax1_ylabel=True, ax2_ylabel=True, legend_inside=True, legend_fontsize=16, loc="best", round_precision_wij=3, round_precision_v=2):
    # plt.style.use('seaborn-muted')
    plt.rcParams.update({
        "font.family": "serif",
        "axes.titlesize": tick_params_size,
        "axes.labelsize": tick_params_size,
        "xtick.labelsize": tick_params_size,
        "ytick.labelsize": tick_params_size,
        "legend.fontsize": tick_params_size
    })

    fig, ax1 = plt.subplots(figsize=(x_size, y_size))

    x_values = np.arange(1, len(face_wij) + 1)
    x_values_offset = x_values + 0.15

    color_face_wij = '#55A868'  # Muted green
    color_gfce_wij = '#C44E52'  # Muted red
    color_face_dists = '#4C72B0'  # Muted blue
    color_gfce_dists = '#8172B3'  # Muted purple

    # First axis
    ax1.plot(x_values, face_wij, '--o', color=color_face_wij, label="Face Path Cost", markersize=8, alpha=0.9, linewidth=6)
    ax1.plot(x_values_offset, gfce_wij, '--o', color=color_gfce_wij, label="FACEGroup Path Cost", markersize=8, alpha=0.9, linewidth=6)

    if ax1_ylabel:
        ax1.set_ylabel("Avg Path Cost", fontsize=tick_params_size)

    ax1.set_xticks(x_values)
    ax1.set_xticklabels([int(k) for k in k_values])
    ax1.set_xlabel("k", fontsize=tick_params_size)
    y1_min = min(min(face_wij), min(gfce_wij))
    y1_max = max(max(face_wij), max(gfce_wij))
    y1ticks_raw = nice_numbers(y1_min, y1_max, 4, score='d', round_precision=round_precision_wij)
    ax1.set_yticks(y1ticks_raw)

    ax2 = ax1.twinx()
    ax2.plot(x_values, face_dists, '-o', color=color_face_dists, label="Face L2 Cost", markersize=8, alpha=0.9, linewidth=6)
    ax2.plot(x_values_offset, gfce_dists, '-o', color=color_gfce_dists, label="FACEGroup L2 Cost", markersize=8, alpha=0.9, linewidth=6)
    if ax2_ylabel:
        ax2.set_ylabel("Avg $L_2$ Cost", fontsize=tick_params_size)

    y2_min = min(min(face_dists), min(gfce_dists))
    y2_max = max(max(face_dists), max(gfce_dists))
    y2ticks_raw = nice_numbers(y2_min, y2_max, 4, score='d', round_precision=round_precision_v)
    ax2.set_yticks(y2ticks_raw)
    if legend_inside:
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles = handles1 + handles2
        labels = labels1 + labels2
        ax1.legend(handles, labels, loc=loc, fontsize=legend_fontsize)
        fig.savefig(f"{FACEGroup_DIR}/tmp/{datasetName}/figs/Coverage_constrained_face_gface_comparison_d_method_{d_method}_maxd_{max_d}_normalized.pdf",
                    bbox_inches='tight', dpi=300)
    else:
        fig.savefig(f"{FACEGroup_DIR}/tmp/{datasetName}/figs/Coverage_constrained_face_gface_comparison_d_method_{d_method}_maxd_{max_d}_normalized_no_legend.pdf",
                    bbox_inches='tight', dpi=300)

        fig_legend = plt.figure(figsize=(4, 2))
        ax_legend = fig_legend.add_subplot(111)

        face_wij_line = Line2D([0, 0], [0, 2], linestyle="dashed", marker='o', color=color_face_wij, markersize=8, linewidth=4, label="Face Path Cost")
        gfce_wij_line = Line2D([0, 0], [0, 1], linestyle='dashed', marker='o', color=color_gfce_wij, markersize=8, linewidth=4, label="FACEGroup Path Cost")
        face_dists_line = Line2D([0, 1], [0, 0], linestyle='solid', marker='o', color=color_face_dists, markersize=8, linewidth=4, label="Face $L_2$ Cost")
        gfce_dists_line = Line2D([0, 1], [0, 0], linestyle='solid', marker='o', color=color_gfce_dists, markersize=8, linewidth=4, label="FACEGroup $L_2$ Cost")

        ax_legend.legend([face_wij_line, gfce_wij_line, face_dists_line, gfce_dists_line],
                            [line.get_label() for line in [face_wij_line, gfce_wij_line, face_dists_line, gfce_dists_line]], 
                            loc='center', fontsize=legend_fontsize, frameon=False, ncol=4)
        ax_legend.axis('off')

        fig_legend.savefig(f"{FACEGroup_DIR}/tmp/{datasetName}/figs/{datasetName}_legend.pdf",
                            bbox_inches='tight', dpi=300)
    plt.show()

def nice_numbers(range_min, range_max, num_ticks, score='k', round_precision=2):
    """
    Generate "nice" numbers for a given range.

    Parameters
    ----------
    range_min : float
        The minimum value of the range.
    range_max : float
        The maximum value of the range.
    num_ticks : int
        The number of ticks to generate.
    score : str
        The score type ('k' or 'd').

    Returns
    -------
    ticks : numpy.ndarray
        An array of "nice" numbers for the given range.
    """
    if range_min >= range_max:
        raise ValueError("range_min must be less than range_max.")
    # if score == 'k' and num_ticks > range_max - range_min:
    #     raise ValueError("num_ticks must be greater than the range size.")
        
    range_size = range_max - range_min
    raw_spacing = range_size / (num_ticks - 1)
    
    exponent = np.floor(np.log10(raw_spacing))
    fraction = raw_spacing / (10**exponent)

    if score == 'k':
        if fraction < 1.5:
            nice_fraction = 1
        elif fraction < 2.5:
            nice_fraction = 2
        elif fraction < 3.5:
            nice_fraction = 3
        elif fraction < 4.5:
            nice_fraction = 4
        elif fraction < 7:
            nice_fraction = 5
        else:
            nice_fraction = 10

        nice_tick_spacing = nice_fraction * 10**exponent
        ticks = range_min + np.arange(num_ticks)*nice_tick_spacing
        ticks = ticks.astype(int)

    elif score == 'd':
        nice_options = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for option in nice_options:
            if fraction <= option:
                nice_fraction = option
                break
        nice_tick_spacing = nice_fraction * 10**exponent
        if range_min + (num_ticks - 1) * nice_tick_spacing < range_max:
            nice_tick_spacing = (range_max - range_min) / (num_ticks - 1)
        ticks = range_min + np.arange(num_ticks)*nice_tick_spacing
        ticks = np.round(ticks, round_precision)
    return ticks

def face_comparison(datasetName="Student", epsilon=3, bandwith_approch="mean_scotts_rule", classifier="xgb",\
                    group_identifier='sex', upper_limit_for_k=10, steps=10, group_identifier_value=None,\
                    skip_model_training=True, skip_bandwith_calculation=True, skip_graph_creation=True, skip_distance_calculation=True,\
                    max_d=1000000000, representation=64, bst=0.1):
    face_dists = []
    face_wij = []
    gfce_dists = []
    gfce_wij = []

    facegroup, graph, distances, data, data_np, data_df_copy, attr_col_mapping, normalized_group_identifer_value, numeric_columns, candidate_counterfactuals,\
                Factuals, Factuals_by_group, node_connectivity, edge_connectivity, feasibility_constraints  = initialize_FACEGroup(epsilon=epsilon,\
                    datasetName=datasetName, group_identifier=group_identifier, classifier=classifier, bandwith_approch=bandwith_approch,\
                    group_identifier_value=group_identifier_value, skip_model_training=skip_model_training, skip_bandwith_calculation=skip_bandwith_calculation,\
                    skip_graph_creation=skip_graph_creation, skip_distance_calculation=skip_distance_calculation, representation=representation)
    fgce_init_dict = {"facegroup": facegroup, "graph": graph, "distances": distances, "data": data, "data_np": data_np, "data_df_copy": data_df_copy,\
            "attr_col_mapping": attr_col_mapping, "normalized_group_identifer_value": normalized_group_identifer_value, "numeric_columns": numeric_columns,\
            "candidate_counterfactuals": candidate_counterfactuals, "Factuals": Factuals, "Factuals_by_group": Factuals_by_group, "node_connectivity": node_connectivity,\
                "edge_connectivity": edge_connectivity, "feasibility_constraints": feasibility_constraints}

    k_values = nice_numbers(1, upper_limit_for_k, steps, score='k')
    face_comparison_results = {}
    for i, cfes in enumerate(k_values):
        print(f"Running for {i}-th time")

        results, data, data_np, attr_col_mapping, data_df_copy, face_vector_distances, gfce_vector_distances,\
              face_wij_distances, gfce_wij_distances = main_coverage_constrained_GCFEs(epsilon=epsilon,
                                datasetName=datasetName, group_identifier=group_identifier,
                                classifier='xgb', compare_with_Face=True, skip_distance_calculation=skip_distance_calculation,
                                skip_model_training=skip_model_training, skip_graph_creation=skip_graph_creation, skip_FACEGroup_calculation=False,
                                k=cfes, max_d = max_d, cost_function="max_path_cost", fgce_init_dict=fgce_init_dict, bst=bst)

        if face_vector_distances == None:
            continue
        face_dists.append(face_vector_distances)
        gfce_dists.append(gfce_vector_distances)
        face_wij.append(face_wij_distances)
        gfce_wij.append(gfce_wij_distances)
        face_comparison_results[cfes] = {"face_vector_distances": face_vector_distances, "gfce_vector_distances": gfce_vector_distances,\
                                         "face_wij_distances": face_wij_distances, "gfce_wij_distances": gfce_wij_distances}
    return face_comparison_results

def get_graph_stats(epsilon=0.4,\
        datasetName='Adult', group_identifier='sex', group_identifier_value=None, bandwith_approch="mean_scotts_rule", classifier='xgb',\
		skip_model_training=True, skip_distance_calculation=True, skip_graph_creation=True, skip_bandwith_calculation=True, verbose=False):
  
  facegroup, graph, distances, data, data_np, data_df_copy, attr_col_mapping, normalized_group_identifer_value, numeric_columns, candidate_counterfactuals,\
                Factuals, Factuals_by_group, node_connectivity, edge_connectivity, feasibility_constraints = initialize_FACEGroup(epsilon=epsilon,\
    datasetName=datasetName, group_identifier=group_identifier, bandwith_approch=bandwith_approch, classifier=classifier,\
    group_identifier_value=group_identifier_value, skip_model_training=skip_model_training, skip_distance_calculation=skip_distance_calculation,\
    skip_graph_creation=skip_graph_creation, skip_bandwith_calculation=skip_bandwith_calculation, verbose=verbose)
      
  subgroups = utils.get_subgraphs_by_group(graph, data_np, data, attr_col_mapping, group_identifier, normalized_group_identifer_value, numeric_columns)
  weakly_connected_components = {}
  subgroup_nodes = {}
  stats = {}
  for group, subgraph in subgroups.items():
    weakly_connected_components[group] = list(nx.weakly_connected_components(subgraph))
    subgroup_nodes[group] = list(subgraph.nodes())

    strongly_connected_components = list(nx.strongly_connected_components(subgraph))
    
    density = nx.density(subgraph) 
    stats[group] = {'num_nodes': len(subgroup_nodes[group]), 'num_strongly_connected_components': len(strongly_connected_components),
            'num_weakly_connected_components': len(weakly_connected_components[group]), 'density': f'{density*100:.2f}'}
  return stats

def mip_to_greedy_comparison(epsilon=3, datasetName='Student',
                    group_identifier='sex', classifier="xgb", bandwith_approch="mean_scotts_rule",
                    k_range=None, k_lower=1, k_upper=10, max_d=3.61, cost_function="max_vector_distance", 
                    k_selection_method="accross_all_ccs", group_identifier_value=None, 
                    skip_model_training=True, skip_distance_calculation=True, skip_graph_creation=True, 
                    skip_bandwith_calculation=True, skip_FACEGroup_calculation=False, 
                    representation=64, fgce_init_dict=None, 
                    verbose=False, mip_runs=5):
    if os.path.exists(f"{FACEGroup_DIR}{sep}tmp{sep}{datasetName}{sep}Mip_greedy_comparison{sep}{str(k_range)}_{max_d}_mipruns{mip_runs}.pkl"):
        with open(f"{FACEGroup_DIR}{sep}tmp{sep}{datasetName}{sep}Mip_greedy_comparison{sep}{str(k_range)}_{max_d}_mipruns{mip_runs}.pkl", "rb") as f:
            all_results = pickle.load(f)
        return all_results['results'], all_results['mips_time'], all_results['greedy_time'], all_results['total_coverage_mip'], all_results['total_coverage_greedy']

    facegroup, graph, distances, data, data_np, data_df_copy, attr_col_mapping, normalized_group_identifer_value, numeric_columns, candidate_counterfactuals,\
			  Factuals, Factuals_by_group, node_connectivity, edge_connectivity, feasibility_constraints  = initialize_FACEGroup(epsilon,\
                datasetName=datasetName, group_identifier=group_identifier, classifier=classifier, bandwith_approch=bandwith_approch, verbose=verbose,\
                group_identifier_value=group_identifier_value, skip_model_training=skip_model_training, skip_distance_calculation=skip_distance_calculation,\
                skip_graph_creation=skip_graph_creation, representation=representation, skip_bandwith_calculation=skip_bandwith_calculation)

    fgce_init_dict = {"facegroup": facegroup, "graph": graph, "distances": distances, "data": data, "data_np": data_np, "data_df_copy": data_df_copy,\
         "attr_col_mapping": attr_col_mapping, "normalized_group_identifer_value": normalized_group_identifer_value, "numeric_columns": numeric_columns,\
         "candidate_counterfactuals": candidate_counterfactuals, "Factuals": Factuals, "Factuals_by_group": Factuals_by_group, "node_connectivity": node_connectivity,\
              "edge_connectivity": edge_connectivity, "feasibility_constraints": feasibility_constraints}

    group_ids = []
    mips_time = {}
    mips_total_coverage = {}
    greedy_time = {}
    greedy_total_coverage = {}
    results = {}

    if k_range is None:
        k_range = range(k_lower, k_upper + 1)

    for k in tqdm(k_range, desc=f"Comparing MIP and Greedy selection for k: {k_range}"):
        # Storage for multiple MIP runs
        mip_coverage_per_group = {}
        mip_avg_cost_per_group = {}
        mip_cfes_count_per_group = {}
        mip_total_coverage = []
        mip_exec_times = []

        ## Run MIP multiple times and collect statistics
        for _ in range(mip_runs):
            mip_res = main_cost_constrained_GCFEs(
                epsilon=epsilon, datasetName=datasetName, group_identifier=group_identifier,
                group_identifier_value=group_identifier_value, skip_model_training=skip_model_training,
                skip_FACEGroup_calculation=skip_FACEGroup_calculation, skip_graph_creation=skip_graph_creation,
                max_d=max_d, cost_function=cost_function, k=k,
                k_selection_method=k_selection_method, fgce_init_dict=fgce_init_dict, cfe_selection_method="mip"
            )[0]

            mip_total_coverage.append(mip_res['Total coverage'])
            mip_exec_times.append(mip_res['Time'])

            # Process per-group statistics
            for group_id in mip_res:
                if group_id in ["Node Connectivity", "Edge Connectivity", "Total coverage", "Graph Stats", "Time"]:
                    continue

                # Initialize if first run
                if group_id not in mip_coverage_per_group:
                    mip_coverage_per_group[group_id] = []
                    mip_avg_cost_per_group[group_id] = []
                    mip_cfes_count_per_group[group_id] = []

                mip_coverage_per_group[group_id].append(mip_res[group_id]['Coverage'])
                mip_avg_cost_per_group[group_id].append(mip_res[group_id]['Avg. distance'])

                # Compute unique CFEs per group
                mip_group_cfes = set()
                for factual in mip_res[group_id]:
                    if factual in ["Coverage", "Avg. distance", "Median distance", "Avg. path cost", "Median path cost"]:
                        continue
                    mip_group_cfes.add(mip_res[group_id][factual]['CFE_name'])
                mip_cfes_count_per_group[group_id].append(len(mip_group_cfes))

        # Compute averages
        avg_mip_exec_time = np.mean(mip_exec_times)
        avg_mip_total_coverage = np.mean(mip_total_coverage)

        # Run Greedy once
        greedy_res = main_cost_constrained_GCFEs(
            epsilon=epsilon, datasetName=datasetName, group_identifier=group_identifier,
            group_identifier_value=group_identifier_value, skip_model_training=skip_model_training,
            skip_FACEGroup_calculation=skip_FACEGroup_calculation, skip_graph_creation=skip_graph_creation,
            max_d=max_d, cost_function=cost_function, k=k,
            k_selection_method=k_selection_method, fgce_init_dict=fgce_init_dict, cfe_selection_method="greedy"
        )[0]

        # Compute number of CFEs for Greedy
        greedy_cfes_count_per_group = {}
        for group_id in greedy_res:
            if group_id in ["Node Connectivity", "Edge Connectivity", "Total coverage", "Graph Stats", "Time"]:
                continue

            greedy_group_cfes = set()
            for factual in greedy_res[group_id]:
                if factual in ["Coverage", "Avg. distance", "Median distance", "Avg. path cost", "Median path cost"]:
                    continue
                greedy_group_cfes.add(greedy_res[group_id][factual]['CFE_name'])
            greedy_cfes_count_per_group[group_id] = len(greedy_group_cfes)

        # Collect group IDs (only in first iteration)
        if not group_ids:
            for group_id in mip_res.keys():
                if group_id in ["Node Connectivity", "Edge Connectivity", "Total coverage", "Graph Stats", "Time"]:
                    continue
                group_ids.append(group_id)

        # Store results per group and per k
        for group_id in group_ids:
            if group_id not in results:
                results[group_id] = {}

            results[group_id][k] = {
                'coverage_mip': np.mean(mip_coverage_per_group[group_id]),
                'avg_cost_mip': np.mean(mip_avg_cost_per_group[group_id]),
                'num_cfes_mip': np.mean(mip_cfes_count_per_group[group_id]),  # Avg CFEs per group in MIP
                'coverage_greedy': greedy_res[group_id]['Coverage'],
                'avg_cost_greedy': greedy_res[group_id]['Avg. distance'],
                'num_cfes_greedy': greedy_cfes_count_per_group[group_id]  # CFEs per group in Greedy
            }

        mips_time[k] = avg_mip_exec_time
        mips_total_coverage[k] = avg_mip_total_coverage
        greedy_time[k] = greedy_res['Time']
        greedy_total_coverage[k] = greedy_res['Total coverage']
    
    all_results = {"results": results, "mips_time": mips_time, "greedy_time": greedy_time, "total_coverage_mip": mips_total_coverage, "total_coverage_greedy": greedy_total_coverage}
    if not os.path.exists(f"{FACEGroup_DIR}{sep}tmp{sep}{datasetName}{sep}Mip_greedy_comparison{sep}"):
        os.makedirs(f"{FACEGroup_DIR}{sep}tmp{sep}{datasetName}{sep}Mip_greedy_comparison{sep}")
    with open(f"{FACEGroup_DIR}{sep}tmp{sep}{datasetName}{sep}Mip_greedy_comparison{sep}{str(k_range)}_{max_d}_mipruns{mip_runs}.pkl", "wb") as f:
        pickle.dump(all_results, f)

    return results, mips_time, greedy_time, mips_total_coverage, greedy_total_coverage

def mip_vs_greedy_plot(datasetName="Student", results=None, mips_time=None, greedy_time=None,
                        mips_total_coverage=None, greedy_total_coverage=None, k_lower=1, k_upper=10, k_range=None, max_d=3):
    plt.style.use('seaborn-muted')
    plt.rcParams.update({
        "font.family": "serif",
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16
    })

    group_ids = list(results.keys())
    if k_range is None:
        k_values = list(range(k_lower, k_upper + 1))
    else:
        k_values = k_range

    # color_mip = '#CC6677'  
    # color_greedy = '#882255'

    color_mip = '#CC6677'  
    color_greedy = '#4C72B0'  

    save_path = f"{FACEGroup_DIR}{sep}tmp{sep}{datasetName}{sep}figs{sep}/greedy_vs_mip/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    ### 1. Total Coverage Plot ###
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(k_values, mips_total_coverage.values(), label='MIP', linestyle='-', marker='o', color=color_mip, linewidth=2, markersize=6)
    ax.plot(k_values, greedy_total_coverage.values(), label='Greedy', linestyle='-', marker='x', color=color_greedy, linewidth=2, markersize=6)
    ax.set_xticks(k_values)
    ax.set_xlabel("k", fontsize=18)
    ax.set_ylabel("Coverage", fontsize=18)
    plt.tight_layout()
    ax.legend(loc='best', frameon=True, fontsize=15)
    plt.savefig(f"{save_path}total_coverage_{max_d}.pdf", bbox_inches='tight', dpi=300)
    plt.show()

    ### 2. Coverage per Group ###
    fig, ax = plt.subplots(figsize=(8, 6))
    for group_id in group_ids:
        linestyle = '-' if group_id == '0.0' else '--'
        ax.plot(k_values, [results[group_id][k]['coverage_mip'] for k in k_values], label=f'Group {int(float(group_id))} (MIP)', linestyle=linestyle, marker='o', color=color_mip, linewidth=2, markersize=6)
        ax.plot(k_values, [results[group_id][k]['coverage_greedy'] for k in k_values], label=f'Group {int(float(group_id))} (Greedy)', linestyle=linestyle, marker='x', color=color_greedy, linewidth=2, markersize=6)
    ax.set_xticks(k_values)
    ax.set_xlabel("k", fontsize=18)
    ax.set_ylabel("Coverage", fontsize=18)
    plt.tight_layout()
    ax.legend(loc='best', frameon=True, fontsize=15)
    plt.savefig(f"{save_path}coverage_{max_d}.pdf", bbox_inches='tight', dpi=300)
    plt.show()

    ### 3. Time Plot ###
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(k_values, [mips_time[k] for k in k_values], label='MIP', linestyle='-', marker='o', color=color_mip, linewidth=2, markersize=6)
    ax.plot(k_values, [greedy_time[k] for k in k_values], label='Greedy', linestyle='-', marker='x', color=color_greedy, linewidth=2, markersize=6)
    ax.set_xticks(k_values)
    ax.set_xlabel("k", fontsize=18)
    ax.set_ylabel("Time (s)", fontsize=18)
    ax.set_yscale('log')
    plt.tight_layout()
    ax.legend(loc='best', frameon=True, fontsize=15)
    plt.savefig(f"{save_path}time_{max_d}.pdf", bbox_inches='tight', dpi=300)
    plt.show()

    ### 4. CFEs Count ###
    fig, ax = plt.subplots(figsize=(8, 6))
    for group_id in group_ids:
        linestyle = '-' if group_id == '0.0' else '--'
        ax.plot(k_values, [results[group_id][k]['num_cfes_mip'] for k in k_values], label=f'Group {int(float(group_id))} (MIP)', linestyle=linestyle, marker='o', color=color_mip, linewidth=2, markersize=6)
        ax.plot(k_values, [results[group_id][k]['num_cfes_greedy'] for k in k_values], label=f'Group {int(float(group_id))} (Greedy)', linestyle=linestyle, marker='x', color=color_greedy, linewidth=2, markersize=6)
    ax.set_xticks(k_values)
    ax.set_xlabel("k", fontsize=18)
    ax.set_ylabel("CFEs Count", fontsize=18)
    ax.legend(loc='best', frameon=True, fontsize=15)
    plt.tight_layout()
    plt.savefig(f"{save_path}cfes_count_{max_d}.pdf", bbox_inches='tight', dpi=300)
    plt.show()