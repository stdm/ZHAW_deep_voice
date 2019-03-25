from networks.lstm_arc_face.arc_face_controller import ArcFaceController

controller = ArcFaceController('speakers_100_50w_50m_not_reynolds_cluster', 'speakers_40_clustering_vs_reynolds')
controller.train_network()
