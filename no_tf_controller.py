from networks.lstm_arc_face.arc_face_controller import ArcFaceController
from networks.pairwise_lstm.lstm_controller import LSTMController

inp = input('[1] arcface [2] lstm?')
if int(inp) == 1:
    controller = ArcFaceController('speakers_100_50w_50m_not_reynolds_cluster', 'speakers_40_clustering_vs_reynolds')
    controller.train_network()
else:
    controller = LSTMController(2, 15, 512)
    controller.train_network()
