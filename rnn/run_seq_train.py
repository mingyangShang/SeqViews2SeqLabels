import os

try:
    os.system('python seq_train.py --train=%s --data_path=%s --data_dir=%s --train_feature_file=%s --train_label_file=%s --test_feature_file=%s --test_label_file=%s --save_seq_mvmodel_path=%s --seq_mvmodel_path=%s'
              %('True', '/home/shangmingyang/PycharmProjects/MVModel/ignore/data/', '/home/shangmingyang/PycharmProjects/MVModel/ignore/data/', 'example_feature.npy', 'example_label.npy', 'example_feature.npy', 'example_label.npy',
                '/home/shangmingyang/PycharmProjects/MVModel/ignore/data/model/seq_mvmodel.ckpt', '/home/shangmingyang/PycharmProjects/MVModel/ignore/data/model/seq_mvmodel.ckpt-10'))
except Exception as e:
    print("Exception:", e)
