from scipy.spatial.distance import euclidean, cosine
from sklearn.metrics import auc
import numpy as np
from scipy.interpolate import interp1d
import os
import threading
from utils.rank_metrics import mean_average_precision, average_precision

def generate_retrival_distance(test_feature_file, train_feature_file, savepath='test_train'):
    test_data, train_data = np.load(test_feature_file), np.load(train_feature_file)
    result = [euclidean(test, train) for test in test_data for train in train_data]
    result = np.reshape(np.array(result), [test_data.shape[0], train_data.shape[0]])
    np.save(savepath, result)

def generate_distance_test2test(test_feature_file, savepath='test_test'):
    test_data = np.load(test_feature_file)
    result = [cosine(test1, test2) for test1 in test_data for test2 in test_data]
    result = np.reshape(np.array(result), [test_data.shape[0], test_data.shape[0]])
    np.save(savepath, result)

def generate_retrival_all_distance(test_feature_file, train_feature_file, savepath='all_all'):
    test_data, train_data = np.load(test_feature_file), np.load(train_feature_file)
    all_data = np.concatenate((test_data, train_data), axis=0)
    result = [euclidean(test1, test2) for test2 in all_data for test1 in all_data]
    result = np.reshape(np.array(result), [all_data.shape[0], all_data.shape[0]])
    np.save(savepath, result)

def generate_labels_all(labels_test_file, labels_train_file, save_all_file):
    labels_test, labels_train = np.load(labels_test_file), np.load(labels_train_file)
    labels_all = np.concatenate((labels_test, labels_train))
    np.save(save_all_file, labels_all)

def retrival_metrics_all(sims_file, labels_file):
    sims, labels = np.load(sims_file), np.load(labels_file)
    #sims = np.array([[0,0.5,1],[0.5,0,2],[1,2,0]])
    #labels = np.array([0,1,0])
    mAP_input = np.zeros(sims.shape)
    for i in range(sims.shape[0]):
        sims_row, curr_label = sims[i], labels[i]
        labels_row = labels[np.argsort(sims_row)]
        for j in range(labels_row.shape[0]):
            if labels_row[j] == curr_label:
                mAP_input[i][j] = 1
    print(mAP_input)
    return mean_average_precision(mAP_input.tolist())

def retrival_metrics_test2train(sims_file, test_labels_file, train_labels_file):
    sims, labels_test, labels_train = np.load(sims_file), np.load(test_labels_file), np.load(train_labels_file)
    mAP_input = np.zeros(sims.shape)
    for i in range(sims.shape[0]):
        sims_row, curr_label = sims[i], labels_test[i]
        labels_row = labels_train[np.argsort(sims_row)]
        for j in range(labels_row.shape[0]):
            if labels_row[j] == curr_label:
                mAP_input[i][j] = 1
    print(mAP_input)
    return mean_average_precision(mAP_input.tolist())

def PR_test2test(sims_file, labels_file, save_prefix="mean"):
    sims, labels = np.load(sims_file), np.load(labels_file)
    Ps, Rs = [], []
    for i in range(sims.shape[0]):
        sims_row, curr_label = sims[i], labels[i]
        labels_row = labels[np.argsort(sims_row)]
        y_true = [1 if curr_label==l else 0 for l in labels_row]
        y_pred = [1 for _ in range(len(y_true))]
        P, R = PR(y_true, y_pred, save=(True if i==20 else False))
        Ps.append(P)
        Rs.append(R)
    Ps, Rs = np.array(Ps), np.array(Rs)
    mean_P, mean_R = np.mean(Ps, axis=0), np.mean(Rs, axis=0)
    mean_P[0] = 1.0
    mean_R[0] = 0.0
    #mean_P, mean_R = np.insert(mean_P, 0, 1.0), np.insert(mean_R, 0, 0.0)
    np.save(save_prefix+"_P", mean_P)
    np.save(save_prefix+"_R", mean_R)
    with open(save_prefix+'_P.txt', 'w') as f:
        f.write('\n'.join([str(p) for p in mean_P]))
    with open(save_prefix+'_R.txt', 'w') as f:
        f.write("\n".join([str(r) for r in mean_R]))

    area = auc(mean_R, mean_P)
    return Ps, Rs, area

def PR_test2train(sims_file, test_labels_file, train_labels_file, save_prefix="mean"):
    sims, test_labels, train_labels = np.load(sims_file), np.load(test_labels_file), np.load(train_labels_file)
    Ps, Rs = [], []
    for i in range(sims.shape[0]):
        sims_row, curr_label = sims[i], test_labels[i]
        labels_row = train_labels[np.argsort(sims_row)]
        y_true = [1 if curr_label==l else 0 for l in labels_row]
        y_pred = [1 for _ in range(len(y_true))]
        P, R = PR(y_true, y_pred, save=(True if i==20 else False))
        Ps.append(P)
        Rs.append(R)
    Ps, Rs = np.array(Ps), np.array(Rs)
    np.save("Ps", Ps)
    np.save("Rs", Rs)
    mean_P, mean_R = np.mean(Ps, axis=0), np.mean(Rs, axis=0)
    mean_P[0] = 1.0
    mean_R[0] = 0.0
    #mean_P, mean_R = np.insert(mean_P, 0, 1.0), np.insert(mean_R, 0, 0.0)
    np.save(save_prefix+"_P", mean_P)
    np.save(save_prefix+"_R", mean_R)
    with open(save_prefix+'_P.txt', 'w') as f:
        f.write('\n'.join([str(p) for p in mean_P]))
    with open(save_prefix+'_R.txt', 'w') as f:
        f.write("\n".join([str(r) for r in mean_R]))

    area = auc(mean_R, mean_P)
    return Ps, Rs, area


def PR(y_true, y_pred, save=False):
    """
    generate precison and recall array for drawing PR-curve
    y_true: array, value is 0 or 1, 1 means the model has same class with query
    y_pred: array, value is 0 or 1, 1 means the model has same class with query
    @return:
        P: array, size is 1 bigger than y_true and y_pred, first is 1
        R: array, size is 1 bigger than y_true and y_pred, first is 0
    example:
        y_true=[1,1,0,0]
        y_pred=[1,0,1,0]
        P=[1,1,1,1/2,1/2]
        R=[0,1/2,1/2,1/2,1/2]
    """
    P, R = [], []
    sum_true, TP = np.count_nonzero(y_true), 0
    TP, FP = 0, 0
    for gd, pred in zip(y_true, y_pred):
        if gd == pred == 1:
            TP += 1
        elif pred == 1 and gd == 0:
            FP += 1
        P.append(TP*1.0/(TP+FP))
        R.append(TP)
        if TP == sum_true:
            break
    R = [r*1.0/sum_true for r in R]
    P, R = [1.0] + P, [0.0] + R
    if save:
        print(P)
    # do linear interpolate
    f = interp1d(R, P)
    new_R = np.linspace(0, 1.0, 1001, endpoint=True)
    return f(new_R), new_R

def PR_curve(P_file, R_file):
    P, R = np.load(P_file), np.load(R_file)

    import pylab as pl
    pl.plot(R, P)
    pl.show()

def retrival_results(train_feature_file, train_label_file, test_feature_file, test_label_file, save_dir="/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/modelnet10"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    generate_distance_test2test(train_feature_file, os.path.join(save_dir, "train2train_euclidean"))
    generate_distance_test2test(test_feature_file, os.path.join(save_dir, "test2test_euclidean"))
    generate_retrival_distance(test_feature_file, train_feature_file, os.path.join(save_dir, "test2train_euclidean"))
    generate_retrival_all_distance(test_feature_file, train_feature_file, os.path.join(save_dir, "all2all_euclidean"))

    generate_labels_all(test_label_file, train_label_file, os.path.join(save_dir, "all_labels"))

    mAP_test2test = retrival_metrics_all(os.path.join(save_dir, "test2test_euclidean.npy"), test_label_file)
    mAP_train2train = retrival_metrics_all(os.path.join(save_dir, "train2train_euclidean.npy"), train_label_file)
    mAP_all2all = retrival_metrics_all(os.path.join(save_dir, "all2all_euclidean.npy"), os.path.join(save_dir, "all_labels.npy"))
    mAP_test2train = retrival_metrics_test2train(os.path.join(save_dir, "test2train_euclidean.npy"), test_label_file, train_label_file)

    mAPs = [mAP_test2test, mAP_train2train, mAP_all2all, mAP_test2train]

    P_test2test, R_test2test, auc_test2test = PR_test2test(os.path.join(save_dir, "test2test_euclidean.npy"), test_label_file)
    np.save(os.path.join(save_dir, "P_test2test"), P_test2test)
    np.save(os.path.join(save_dir, "R_test2test"), R_test2test)
    aucs = [auc_test2test]
    try:
        PR_curve(os.path.join(save_dir, "P_test2test.npy"), os.path.join(save_dir, "R_test2test.npy"))
    except Exception as e:
        print(e)
    print(mAPs, aucs)
    return mAPs, aucs

def PR_modelnet10():
    P_test2test, R_test2test, auc_test2test = PR_test2test("/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/lhl/modelnet10/test2test_euclidean.npy", "/home3/lhl/modelnet10_v2/feature10/test_labels_modelnet10.npy", save_prefix='lhl_modelnet10_test2test')
    P_train2train, R_train2train, auc_train2train = PR_test2test("/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/lhl/modelnet10/train2train_euclidean.npy", "/home3/lhl/modelnet10_v2/feature10/train_labels_modelnet10.npy", save_prefix='lhl_modelnet10_train2train')
    P_all2all, R_all2all, auc_all2all = PR_test2test("/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/lhl/modelnet10/all2all_euclidean.npy", "/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/modelnet10/all_labels.npy", save_prefix='lhl_modelnet10_all2all')
    P_test2train, R_test2train, auc_test2train = PR_test2train("/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/lhl/modelnet10/test2train_euclidean.npy", "/home3/lhl/modelnet10_v2/feature10/test_labels_modelnet10.npy", '/home3/lhl/modelnet10_v2/feature10/train_labels_modelnet10.npy', save_prefix='lhl_modelnet10_test2train')
    print(auc_test2test, auc_train2train, auc_all2all, auc_test2train)

def PR_modelnet40():
    P_test2test, R_test2test, auc_test2test = PR_test2test("/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/lhl/modelnet40/test2test_euclidean.npy", "/home3/lhl/modelnet40_total_v2/test_label.npy", save_prefix='lhl_modelnet40_test2test')
    P_train2train, R_train2train, auc_train2train = PR_test2test("/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/lhl/modelnet40/train2train_euclidean.npy", "/home3/lhl/modelnet40_total_v2/train_label.npy", save_prefix='lhl_modelnet40_train2train')
    P_all2all, R_all2all, auc_all2all = PR_test2test("/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/lhl/modelnet40/all2all_euclidean.npy", "/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/modelnet40/all_labels.npy", save_prefix='lhl_modelnet40_all2all')
    P_test2train, R_test2train, auc_test2train = PR_test2train("/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/lhl/modelnet40/test2train_euclidean.npy", "/home3/lhl/modelnet40_total_v2/test_label.npy", '/home3/lhl/modelnet40_total_v2/train_label.npy', save_prefix='lhl_modelnet40_test2train')
    print(auc_test2test, auc_train2train, auc_all2all, auc_test2train)

def retrival_shapenet(sims_file, ids_file, save_dir='/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/shapenet55', max_n=1000):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    sims, ids = np.load(sims_file), np.load(ids_file)
    for i in range(sims.shape[0]):
        with open(os.path.join(save_dir, ids[i]), 'w') as f:
            sims_row, curr_id = sims[i], ids[i]
            ids_row = ids[np.argsort(sims_row)][:min(max_n, sims.shape[1])]
            sims_row = np.sort(sims_row)[:min(max_n, sims.shape[1])]
            for modelid, dis in zip(ids_row, sims_row):
                f.write(' '.join([modelid, str(dis)])+'\n')


def shapenet55_color():
    def test2test():
        generate_distance_test2test('/home3/lhl/cnn-text-classification-tf-shapenet/feature/test_feature_v10_SNC_filternums1024.npy', '/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/lhl/shapenet55/color/shapenet55_color_test2test_euclidean')
        retrival_shapenet('/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/lhl/shapenet55/color/shapenet55_color_test2test_euclidean.npy', '/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/shapenet55/shapenet55_v1_test_ids.npy', save_dir='/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/lhl/shapenet55/color/test_normal', max_n=1000)

    def train2train():
        generate_distance_test2test('/home3/lhl/cnn-text-classification-tf-shapenet/feature/train_feature_v10_SNC_filternums1024.npy', '/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/lhl/shapenet55/color/shapenet55_color_train2train_euclidean')
        retrival_shapenet('/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/lhl/shapenet55/color/shapenet55_color_train2train_euclidean.npy', '/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/shapenet55/shapenet55_v1_train_ids.npy', save_dir='/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/lhl/shapenet55/color/train_normal', max_n=1000)

    def val2val():
        generate_distance_test2test('/home3/lhl/cnn-text-classification-tf-shapenet/feature/val_feature_v10_SNC_filternums1024.npy', '/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/lhl/shapenet55/color/shapenet55_color_val2val_euclidean')
        retrival_shapenet('/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/lhl/shapenet55/color/shapenet55_color_val2val_euclidean.npy', '/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/shapenet55/shapenet55_v1_val_ids.npy', save_dir='/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/lhl/shapenet55/color/val_normal', max_n=1000)
    t_test = threading.Thread(target=test2test)
    t_test.start()
    t_train = threading.Thread(target=train2train)
    t_train.start()
    t_val = threading.Thread(target=val2val)
    t_val.start()





    #generate_distance_test2test('/home3/lhl/cnn-text-classification-tf-shapenet/feature/test_feature_v10_SNC_filternums1024.npy', '/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/lhl/shapenet55/color/shapenet55_color_test2test_euclidean')
    #generate_distance_test2test('/home3/lhl/cnn-text-classification-tf-shapenet/feature/train_feature_v10_SNC_filternums1024.npy', '/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/lhl/shapenet55/color/shapenet55_color_train2train_euclidean')
    #generate_distance_test2test('/home3/lhl/cnn-text-classification-tf-shapenet/feature/val_feature_v10_SNC_filternums1024.npy', '/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/lhl/shapenet55/color/shapenet55_color_val2val_euclidean')
    #retrival_shapenet('/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/lhl/shapenet55/color/shapenet55_color_test2test_euclidean.npy', '/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/shapenet55/shapenet55_v1_test_ids.npy', save_dir='/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/lhl/shapenet55/color/test_normal', max_n=1000)
    #retrival_shapenet('/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/lhl/shapenet55/color/shapenet55_color_train2train_euclidean.npy', '/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/shapenet55/shapenet55_v1_train_ids.npy', save_dir='/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/lhl/shapenet55/color/train_normal', max_n=1000)
    #retrival_shapenet('/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/lhl/shapenet55/color/shapenet55_color_val2val_euclidean.npy', '/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/shapenet55/shapenet55_v1_val_ids.npy', save_dir='/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/lhl/shapenet55/color/val_normal', max_n=1000)

def shapenet55_nocolor():
    #generate_distance_test2test('/home3/lhl/cnn-text-classification-tf-shapenet/feature/train_feature_v10_SN_filternums512.npy', '/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/lhl/shapenet55/nocolor/lhl_shapenet55_nocolor_train2train_euclidean')
    #generate_distance_test2test('/home3/lhl/cnn-text-classification-tf-shapenet/feature/test_feature_v10_SN_filternums512.npy', '/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/lhl/shapenet55/nocolor/lhl_shapenet55_nocolor_test2test_euclidean')
    #generate_distance_test2test('/home3/lhl/cnn-text-classification-tf-shapenet/feature/val_feature_v10_SN_filternums512.npy', '/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/lhl/shapenet55/nocolor/lhl_shapenet55_nocolor_val2val_euclidean')
    retrival_shapenet('/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/lhl/shapenet55/nocolor/lhl_shapenet55_nocolor_train2train_euclidean.npy', '/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/shapenet55/shapenet55_v1_train_ids.npy', save_dir='/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/lhl/shapenet55/nocolor/train_normal', max_n=1000)
    print("retrival shapenet nocolor train2train finished")
    retrival_shapenet('/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/lhl/shapenet55/nocolor/lhl_shapenet55_nocolor_test2test_euclidean.npy', '/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/shapenet55/shapenet55_v1_test_ids.npy', save_dir='/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/lhl/shapenet55/nocolor/test_normal', max_n=1000)
    print("retrival shapenet nocolor test2test finished")
    retrival_shapenet('/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/lhl/shapenet55/nocolor/lhl_shapenet55_nocolor_val2val_euclidean.npy', '/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/shapenet55/shapenet55_v1_val_ids.npy', save_dir='/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/lhl/shapenet55/nocolor/val_normal', max_n=1000)
    print("retrival shapenet nocolor val2val finished")


def merge_metrics_shapenet55(metrics_dir, save_dir):
    import csv, glob
    metrics_files = glob.glob(os.path.join(metrics_dir, 'mvmodel.summary-*.csv'))
    metrics_files.sort(key=lambda name: int(name[name.rindex('-')+1 : name.rindex('.csv')]))
    micro_p, micro_r, micro_f1, micro_mAP, micro_ndcg = [], [], [], [], []
    macro_p, macro_r, macro_f1, macro_mAP, macro_ndcg = [], [], [], [], []

    for file in metrics_files:
        print(file)
        with open(file, 'rb') as f:
            metrics = f.readline()
            reader = csv.reader(f)
            for row in reader:
                if row[0] == "microALL":
                    micro_p.append(row[1])
                    micro_r.append(row[2])
                    micro_f1.append(row[3])
                    micro_mAP.append(row[4])
                    micro_ndcg.append(row[5])
                elif row[0] == 'macroALL':
                    macro_p.append(row[1])
                    macro_r.append(row[2])
                    macro_f1.append(row[3])
                    macro_mAP.append(row[4])
                    macro_ndcg.append(row[5])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(os.path.join(save_dir, 'micro_p'), np.array(micro_p))
    np.save(os.path.join(save_dir, 'micro_r'), np.array(micro_r))
    np.save(os.path.join(save_dir, 'micro_f1'), np.array(micro_f1))
    np.save(os.path.join(save_dir, 'micro_mAP'), np.array(micro_mAP))
    np.save(os.path.join(save_dir, 'micro_ndcg'), np.array(micro_ndcg))

    np.save(os.path.join(save_dir, 'macro_p'), np.array(macro_p))
    np.save(os.path.join(save_dir, 'macro_r'), np.array(macro_r))
    np.save(os.path.join(save_dir, 'macro_f1'), np.array(macro_f1))
    np.save(os.path.join(save_dir, 'macro_mAP'), np.array(macro_mAP))
    np.save(os.path.join(save_dir, 'macro_ndcg'), np.array(macro_ndcg))






if __name__ == '__main__':

    #P_test2test, R_test2test, auc_test2test = PR_test2test("/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/modelnet40/test2test_euclidean.npy", "/home3/lhl/modelnet40_total_v2/test_label.npy", save_prefix='modelnet40_test2test')
    #PR_modelnet40()
    #P_test2test, R_test2test, auc_test2test = PR_test2train("/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/modelnet10/test2train_euclidean.npy", "/home3/lhl/modelnet10_v2/feature10/test_labels_modelnet10.npy", '/home3/lhl/modelnet10_v2/feature10/train_labels_modelnet10.npy', save_prefix='modelnet10_test2train')
    #retrival_results("/home3/lhl/cnn-text-classification-tf-modelnet10/feature/train_feature_v10_modelnet10.npy",
                     #"/home3/lhl/modelnet10_v2/feature10/train_labels_modelnet10.npy",
                     #"/home3/lhl/cnn-text-classification-tf-modelnet10/feature/test_feature_v10_modelnet10.npy",
                     #"/home3/lhl/modelnet10_v2/feature10/test_labels_modelnet10.npy",
                     #save_dir="/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/lhl/modelnet10")
    #PR_modelnet40()
    #PR_test2test("/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/modelnet10/test2test_euclidean.npy", "/home3/lhl/modelnet10_v2/feature10/test_labels_modelnet10.npy")
    #retrival_distance('/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/modelnet10_test_hidden.npy', '/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/modelnet10_train_hidden.npy', '/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/modelnet10_test_train_euclidean')
    shapenet55_nocolor()
    #shapenet55_color()
    #merge_metrics_shapenet55('/home/shangmingyang/wuque/projects/evaluator', '/home/shangmingyang/wuque/projects/evaluator/test_normal')
    #retrival_all_distance('/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/modelnet10_test_hidden.npy', '/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/modelnet10_train_hidden.npy', '/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/modelnet10_all2all_euclidean')
    #generate_labels_all('/home3/lhl/modelnet40_total_v2/test_label.npy', '/home3/lhl/modelnet40_total_v2/train_label.npy', '/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/all_labels')
    #retrival_metrics_all('/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/all_all_euclidean.npy', '/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/all_labels.npy')
    #retrival_metrics_test2train('/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/test_train_euclidean.npy', '/home3/lhl/modelnet40_total_v2/test_label.npy', '/home3/lhl/modelnet40_total_v2/train_label.npy')
    #retrival_metrics_all('/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/modelnet10_train2train_euclidean.npy', '/home3/lhl/modelnet10_v2/feature10/train_labels_modelnet10.npy')
    #_,_,area=PR_test2test('/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/test_test_euclidean.npy', '/home3/lhl/modelnet40_total_v2/test_label.npy')
    #print area
