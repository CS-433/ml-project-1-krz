import numpy as np
import matplotlib.pyplot as plt
import itertools as it
from implementations import ridge_regression, logistic_regression, least_squares
from proj1_helpers import *
from model_imp import *

############################# HELPER FUNCTIONS #################################

def extract_features_and_labels(data, labels, ls):
    extracted = data[:, ls]
    extracted_labels = [labels[i] for i in ls]
    return extracted, extracted_labels

def drop_features_and_labels(data, labels, ls):
    clean_ = np.delete(data.T, ls, axis=0).T
    clean_labels = np.delete(labels, ls)
    return clean_, clean_labels

def extract_relevant_features_and_labels(category):
    return drop_features_and_labels(tx_cats[category], labels, ir_ind[category])

def replace_unknown_values(data, labels, replace_median):
    """replace all unknown (`-999.0`) values with the mean, except for features specified by `replace_median`"""
    dat = data.copy()
    for ind, f in enumerate(labels):
        known_values = np.where(dat.T[ind] != -999.0)
        unknown_values = np.where(dat.T[ind] == -999.0)
        col = dat.T[ind, known_values]
        #replace_value = np.median(col) if f in replace_median else np.mean(col)
        replace_value = np.mean(col)
        dat.T[ind, unknown_values] = replace_value
    return dat

def filter_relevant_features(tx_cats):
    for ind, cat in enumerate(tx_cats):
        tx_cats[ind] = tx_cats[ind][:, rel_ind[ind]]

    return tx_cats

def build_final_features(tx_cats):
    final_features = []

    for ind in jet_cat:
        engineered_features = []
        engineered_labels = []
        rel_labels = [labels[rel_ind[ind][j]] for j in range(len(rel_ind[ind]))]

        # Get all 2-combinations of relevant labels
        combinations = it.combinations(rel_labels, 2)
        corr_combinations = [i for i in combinations if i[0] in low_corr_features[ind] or i[1]
                            in low_corr_features]  # Only keep those involving at least one low-correlated feature
        # Now, create all new features and labels by multiplying feature columns elementwise
        for comb in corr_combinations:
            ind1, ind2 = rel_labels.index(comb[0]), rel_labels.index(comb[1])
            engineered_labels.append('{a} * {b}'.format(a=comb[0], b=comb[1]))
            engineered_features.append(tx_cats[ind].T[ind1] * tx_cats[ind].T[ind2])

        engineered_labels = np.array(engineered_labels)
        engineered_features = np.array(engineered_features).T

        # plot_feature_graphs(y_cats[ind], engineered_features, engineered_labels, 10)

        final_features.append(np.hstack((tx_cats[ind], engineered_features)))

    return final_features

def recombine_data(predictions, original_data):
    """Format predictions as they originally were, i.e. the right prediction at the right row"""
    indexed_predictions = []
    for i in jet_cat:
        indexes = np.where(original_data[:, lab_dict['PRI_jet_num']] == i)[0]
        ind_pred = np.vstack((indexes, predictions[i]))
        indexed_predictions.append(ind_pred)

    indexed_predictions = np.concatenate(indexed_predictions, axis=1)
    sorted_predictions = indexed_predictions.T[np.argsort(
        indexed_predictions.T[:, 0])]
    return sorted_predictions[:, 1]

def predict_category_labels(weights_arr, tX_test):
    """Perform category-wise prediction"""
    _, tx_cats = split_data_by_jet_num(tX_test)
    tx_cats = filter_relevant_features(tx_cats)
    final_features = build_final_features(tx_cats)
    predictions = []
    for ind, cat in enumerate(final_features):
        predictions.append(predict_labels(weights_arr[ind], poly_expansion(cat, 3)))

    return recombine_data(predictions, tX_test)

def split_data_by_jet_num(tx, y=None):
    y_cats = []
    tx_cats = []
    for i in jet_cat:
        indexes = np.where(tx[:, lab_dict['PRI_jet_num']] == i)
        if (y is not None):
            y_cats.append(y[indexes])
        tx_cats.append(tx[indexes])

    return y_cats, tx_cats

################################################################################

DATA_TRAIN_PATH = '../data/train.csv'
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

# Get label names
labels = []
with open('../data/train.csv', 'r') as f:
    labels = f.readline().rstrip()
labels = labels.split(',')[2:] # remove first two columns (which are ID and prediction respectively)

lab_dict = {}
for ind, lab in enumerate(labels):
    lab_dict.update({lab: ind})

jet_cat = [0, 1, 2, 3]

y_cats, tx_cats = split_data_by_jet_num(tX, y)

irrelevant_features = {
    0: ['DER_deltaeta_jet_jet', 'DER_mass_jet_jet', 'DER_prodeta_jet_jet', 'DER_lep_eta_centrality', 'PRI_jet_num',
        'PRI_jet_leading_pt', 'PRI_jet_leading_eta', 'PRI_jet_leading_phi', 'PRI_jet_subleading_pt', 'PRI_jet_subleading_eta',
        'PRI_jet_subleading_phi', 'PRI_jet_all_pt'],
    1: ['DER_deltaeta_jet_jet', 'DER_mass_jet_jet', 'DER_prodeta_jet_jet', 'DER_lep_eta_centrality', 'PRI_jet_num',
        'PRI_jet_subleading_pt', 'PRI_jet_subleading_eta', 'PRI_jet_subleading_phi'],
    2: ['PRI_jet_num'],
    3: ['PRI_jet_num']
}

# Get irrelevant feature indexes
ir_ind = {}
for k in irrelevant_features.keys():
    ir_ind.update({k: []})
    for v in irrelevant_features[k]:
        ir_ind[k].append(lab_dict[v])

# Get relevant feature indexes
all_indexes = np.array(list(range(len(labels))))
rel_ind = {}
for k in ir_ind.keys():
    rel_ind.update({k: list(np.delete(all_indexes, ir_ind[k]))})

to_replace = set()
for i in jet_cat:
    for ind in rel_ind[i]:
        needs_replacing = np.any(tX[:, ind] == -999.0)
        if (needs_replacing):
            to_replace.add(labels[ind])

to_replace_ind = []
for r in to_replace:
    to_replace_ind.append(lab_dict[r])
to_replace_ind.sort()

clean_, clean_labels = extract_features_and_labels(tX, labels, to_replace_ind)

replace_median = ['DER_mass_MMC', 'DER_mass_jet_jet',
                  'PRI_jet_leading_pt', 'PRI_jet_subleading_pt']

replaced_ = replace_unknown_values(clean_, clean_labels, replace_median)

# Replace updated data in a new X array
cleaned_xt = []
for label in labels:
    if label in clean_labels:
        cleaned_xt.append(replaced_.T[clean_labels.index(label)])
    else:
        cleaned_xt.append(tX.T[ind])

cleaned_x = np.array(cleaned_xt).T

# Sanity check: in relevant indexes of the data, there should no longer be any unknown values
truth_values = []
for i in jet_cat:
    truth_values.append(np.all(cleaned_x.T[rel_ind[i]] != -999.0))

tx_cats = []
for i in jet_cat:
    subset = tX[np.where(tX[:, lab_dict['PRI_jet_num']] == i)]
    relevant_subset = subset.T[rel_ind[i]]
    tx_cats.append(relevant_subset.T)

# Sanity check: number of features of categories is same as number of relevant features
truth_values = []
for i in jet_cat:
    truth_values.append(len(tx_cats[i].T) == len(rel_ind[i]))

low_corr_features = {
    0: ['PRI_tau_eta', 'PRI_tau_phi', 'PRI_lep_eta', 'PRI_lep_phi', 'PRI_met_phi'],
    1: ['PRI_tau_eta', 'PRI_tau_phi', 'PRI_lep_eta', 'PRI_lep_phi', 'PRI_met_phi',
        'PRI_jet_leading_eta', 'PRI_jet_leading_phi'],
    2: ['PRI_tau_eta', 'PRI_tau_phi', 'PRI_lep_eta', 'PRI_lep_phi', 'PRI_met_phi',
        'PRI_jet_leading_eta', 'PRI_jet_leading_phi', 'PRI_jet_subleading_eta', 'PRI_jet_subleading_phi'],
    3: ['PRI_tau_eta', 'PRI_tau_phi', 'PRI_lep_eta', 'PRI_lep_phi', 'PRI_met_phi',
        'PRI_jet_leading_eta', 'PRI_jet_leading_phi', 'PRI_jet_subleading_eta', 'PRI_jet_subleading_phi'],
}

final_features = build_final_features(tx_cats)

truth_values = []
for i in jet_cat:
    tx_cats[i] = (tx_cats[i] - np.mean(tx_cats[i], axis=0)) / np.std(tx_cats[i], axis=0)
    truth_values.append(np.allclose(0, np.mean(tx_cats[i], axis=0)) and np.allclose(1, np.std(tx_cats[i], axis=0)))


weights_arr = []
lambda_ = 1e-6
gamma = 0.01
max_iters = 5

for i in jet_cat:
    initial_w = np.zeros(shape=(final_features[i].shape[1], 1))
    #weights, loss = logistic_regression(y_cats[i], final_features[i], initial_w, max_iters, gamma)
    weights, loss = ridge_regression(y_cats[i], poly_expansion(final_features[i], 3), lambda_)
    weights_arr.append(weights)

DATA_TEST_PATH = '../data/test.csv' # TODO: download train data and supply path here
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

OUTPUT_PATH = '../output/prediction.csv' # TODO: fill in desired name of output file for submission
#y_pred = predict_labels(weights, tX_test)
y_pred = predict_category_labels(weights_arr, tX_test)
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
