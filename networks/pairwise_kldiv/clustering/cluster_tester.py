"""
    A suite for testing the MR calculation

    EDIT Heusser:
    - Create method plot_mr_clusters from "main" content.

    Work of Lukic and Vogt, adapted by Heusser
"""
import pickle

from matplotlib import pyplot as plt

from .cluster import generate_embedings, calc_MR


def plot_mr_clusters(base_path, train_file_name, test_file_name):
    rel_path = base_path + 'output/'
    calc_mr_for_layer(rel_path + train_file_name, rel_path + test_file_name, 'Output')

    rel_path = base_path + 'dense_2/'
    calc_mr_for_layer(rel_path + train_file_name, rel_path + test_file_name, 'Dense 2')

    rel_path = base_path + 'dense_1/'
    calc_mr_for_layer(rel_path + train_file_name, rel_path + test_file_name, 'Dense 1')

    plt.xlabel('Clusters')
    plt.ylabel('Misclassification Rate (MR)')
    plt.grid()
    plt.legend(loc='lower right', shadow=False)
    plt.ylim(0, 1)
    plt.show()


def calc_mr_for_layer(train_file, test_file, label):
    train_output, test_output, train_speakers, test_speakers = load_data(train_file, test_file)
    X, y, num_speakers = generate_embedings(train_output, test_output, train_speakers, test_speakers, 590)
    MRs = calc_MR(X, y, num_speakers, 'cosine')
    plt.plot(MRs, label=label, linewidth=2)


def load_data(train_file, test_file):
    with open(train_file, 'rb') as f:
        train_output, train_speakers, train_speaker_names = pickle.load(f)
    with open(test_file, 'rb') as f:
        test_output, test_speakers, test_speaker_names = pickle.load(f)
    return train_output, test_output, train_speakers, test_speakers


if __name__ == "__main__":
    PATH = '../../data/experiments/01_speaker_identification_for_clustering/outputs_4422_v2_1000_valid/'
    TRAIN_FILE = 'output_train_20_clustering_vs_reynolds_44_22_2000_valid.pickle'
    TEST_FILE = 'output_test_20_clustering_vs_reynolds_44_22_2000_valid.pickle'

    plot_mr_clusters(PATH, TRAIN_FILE, TEST_FILE)


# plt.savefig('/Users/yanicklukic/Google Drive/Carlo+Yanick/BA/experimente/01/diagrams/known_speakers/all_layers_40.png')

# import output_vector_plotter as ovp
#
# tsne = manifold.TSNE(n_components=2, perplexity=30, early_exaggeration=1.0, learning_rate=100, metric="cityblock", init='pca', random_state=10)
# Y_train = tsne.fit_transform(X_train)
# Y_test = tsne.fit_transform(X_test)
#
# for i in range(len(Y_train)):
#     name = train_speaker_names[i]
#     plt.scatter(Y_train[i, 0], Y_train[i, 1], c=ovp.hex_to_rgb(ovp.COLOR_VALUES[i]), label=name, s=50)
#     plt.scatter(Y_test[i, 0], Y_test[i, 1], c=ovp.hex_to_rgb(ovp.COLOR_VALUES[i]), s=50)
#
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend(scatterpoints=1, loc='upper right', ncol=1, fontsize=8)
# plt.show()
