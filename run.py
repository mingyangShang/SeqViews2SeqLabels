import os
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='modelnet10', help='dataset used to train and test')
parser.add_argument('--train', help='train mode or test mode')
args = parser.parse_args()


train_cmd_base = 'python train.py --n_hidden=128 --decoder_embedding_size=256 --n_views=12 --use_lstm=False --keep_prob=0.5 --training_epoches=200 --save_epoches=1 --learning_rate=0.0002 --batch_size=32 --n_max_keep_model=200 '
train_cmd = train_cmd_base

data_paths = {"modelnet10": ["/home3/lhl/tensorflow-vgg-master-total/feature/train_12p_vgg19_epo48_do05_sigmoid7_feature_class10.npy", "/home3/lhl/modelnet10_v2/feature10/train_labels_modelnet10.npy", "/home3/lhl/tensorflow-vgg-master-total/feature/test_12p_vgg19_epo48_do05_sigmoid7_feature_class10.npy", "/home3/lhl/modelnet10_v2/feature10/test_labels_modelnet10.npy", "/home1/shangmingyang/data/3dmodel/trained_seq_mvmodel/modelnet10/", "modelnet10.csv"],
                "modelnet40": ["/home3/lhl/tensorflow-vgg-master-total/feature/train_12p_vgg19_epo10_do05_sigmoid7_feature_total.npy", "/home3/lhl/modelnet40_total_v2/train_label.npy", "/home3/lhl/tensorflow-vgg-master-total/feature/test_12p_vgg19_epo10_do05_sigmoid7_feature_total.npy", "/home3/lhl/modelnet40_total_v2/test_label.npy", "/home1/shangmingyang/data/3dmodel/trained_seq_mvmodel/modelnet40_adam/", "modelnet40.csv"],
                "shapenet55": ["/home3/lhl/tensorflow-vgg-master-shapenet/feature/train_feature_SN55_epo17.npy", "/home1/shangmingyang/data/3dmodel/shapenet/shapenet55_v1_train_labels.npy", "/home1/shangmingyang/data/ImgJoint3D/feature/eval_shape_img_feature.npy", "/home1/shangmingyang/data/ImgJoint3D/feature/fake_labels.npy", "/home1/shangmingyang/data/3dmodel/trained_seq_mvmodel/shapenet55_256_512_0.0002_0.5/", "shapenet55_256_512_0.0002_0.5.csv"]}

if args.dataset == "modelnet10":
    path = data_paths["modelnet10"]
    train_cmd = train_cmd_base + '--n_classes=10 --train_feature_file=%s --train_label_file=%s --test_feature_file=%s --test_label_file=%s --save_seq_embeddingmvmodel_path=%s --checkpoint_path=%s --test_acc_file=%s '%(path[0], path[1], path[2], path[3], os.path.join(path[4], "mvmodel.ckpt"), os.path.join(path[4], "checkpoint"), path[5])
elif args.dataset == "modelnet40":
    path = data_paths["modelnet40"]
    train_cmd = train_cmd_base + '--n_classes=40 --train_feature_file=%s --train_label_file=%s --test_feature_file=%s --test_label_file=%s --save_seq_embeddingmvmodel_path=%s --checkpoint_path=%s --test_acc_file=%s '%(path[0], path[1], path[2], path[3], os.path.join(path[4], "mvmodel.ckpt"), os.path.join(path[4], "checkpoint"), path[5])
elif args.dataset == 'shapenet55':
    path = data_paths["shapenet55"]
    train_cmd = train_cmd_base + '--n_classes=55 --train_feature_file=%s --train_label_file=%s --test_feature_file=%s --test_label_file=%s --save_seq_embeddingmvmodel_path=%s --checkpoint_path=%s --test_acc_file=%s --enrich_shapenet=%s'%(path[0], path[1], path[2], path[3], os.path.join(path[4], "mvmodel.ckpt"), os.path.join(path[4], "checkpoint"), path[5], False)
else:
    print("dataset muse one of [modelnet10, modelnet40, shapenet55], can not be %s" %(args.dataset))
    sys.exit()
train_cmd = train_cmd + " --train=%s"%args.train

print(train_cmd)
os.system(train_cmd)

