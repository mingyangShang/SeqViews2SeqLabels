import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import euclidean, cosine


random_image = np.random.random([100, 500])
# print(np.shape(random_image))

# plt.imshow(random_image, cmap='gray', interpolation='nearest')

# linear0 = np.linspace(0, 1, 2500).reshape((50, 50))
# linear1 = np.linspace(0, 255, 2500).reshape((50, 50)).astype(np.uint8)
#
# print("Linear0:", linear0.dtype, linear0.min(), linear0.max())
# print("Linear1:", linear1.dtype, linear1.min(), linear1.max())
#
# fig, (ax0, ax1) = plt.subplots(1, 2)
# ax0.imshow(linear0, cmap='gray')
# ax1.imshow(linear1, cmap='gray')
# plt.show()

embeddings_file = '../ignore/data/embedding.npy'
proj_file = '../ignore/data/proj_w.npy'

def show_model_features(index=0, type='tanh', normalize=True):
    # do transformation
    features_relu = np.load('../ignore/test_12p_vgg19_29epo_feature.npy')
    features_tanh = np.load('../ignore/test_12p_vgg19_epo29_tanh7_feature.npy')
    data_relu = features_relu[index]
    data_tanh = features_tanh[index]

    max_f_relu, min_f_relu = np.max(data_relu), np.min(data_relu)
    max_f_tanh, min_f_tanh = np.max(data_tanh), np.min(data_tanh)
    if normalize:
        for i in range(np.shape(data_relu)[0]):
            for j in range(np.shape(data_relu)[1]):
                data_relu[i][j] = normalization(min_f_relu, max_f_relu, data_relu[i][j])
        max_f_relu, min_f_relu = np.max(data_relu), np.min(data_relu)
        for i in range(np.shape(data_tanh)[0]):
            for j in range(np.shape(data_tanh)[1]):
                data_tanh[i][j] = normalization(min_f_tanh, max_f_tanh, data_tanh[i][j])
        max_f_tanh, min_f_tanh = np.max(data_tanh), np.min(data_tanh)
    # for i in range(7):
    #     data_relu = np.concatenate((data_relu, data_relu))
    # for i in range(7):
    #     data_tanh = np.concatenate((data_tanh, data_tanh))
    data_relu = np.repeat(data_relu, 200, axis=0)
    data_tanh = np.repeat(data_tanh, 200, axis=0)

    fig_relu, ax_relu = plt.subplots()
    fig_relu.suptitle("relu")
    cax_relu = ax_relu.imshow(data_relu, cmap='gray', interpolation='nearest')
    cbar_relu = fig_relu.colorbar(cax_relu, ticks=[min_f_relu, max_f_relu])
    cbar_relu.ax.set_yticklabels([str(min_f_relu), str(max_f_relu)])  # vertically oriented colorbar

    fig_tanh, ax_tanh = plt.subplots()
    fig_tanh.suptitle("tanh")
    cax_tanh = ax_tanh.imshow(data_tanh, cmap='gray', interpolation='nearest')
    cbar_tanh = fig_tanh.colorbar(cax_tanh, ticks=[min_f_tanh, max_f_tanh])
    cbar_tanh.ax.set_yticklabels([str(min_f_tanh), str(max_f_tanh)])

def normalization(min_v, max_v, v):
    return 1.0*(v-min_v)/(max_v-min_v)

def show_embedding_dis(embedding_file, distance=euclidean):
    embeddings = np.load(embedding_file)
    class_embeddings = embeddings[1::2]
    n_class = class_embeddings.shape[0]
    embedding_dis = []
    for embedding_i in class_embeddings:
        for embedding_j in class_embeddings:
            embedding_dis.append(distance(embedding_i, embedding_j))
            embedding_dis[-1] = 0 if embedding_dis[-1] < 0 else embedding_dis[-1]
    embedding_dis = np.reshape(np.array(embedding_dis), [n_class, n_class])
    min_dis, max_dis = np.min(embedding_dis), np.max(embedding_dis)
    fig, ax = plt.subplots()
    fig.suptitle(distance.__str__())
    cax = ax.imshow(embedding_dis, cmap='gray', interpolation='nearest')
    cbar = fig.colorbar(cax, ticks=[min_dis, max_dis])
    cbar.ax.set_yticklabels([str(min_dis), str(max_dis)])

def show_proj_w(proj_file):
    proj_w = np.load(proj_file)
    proj_w_class = proj_w[np.arange(proj_w.shape[0]), 1::2]
    min_val, max_val = np.min(proj_w_class), np.max(proj_w_class)
    fig, ax = plt.subplots()
    fig.suptitle("Proj_w")
    cax = ax.imshow(proj_w_class, cmap='gray', interpolation='nearest')
    cbar = fig.colorbar(cax, ticks=[min_val, max_val])
    cbar.ax.set_yticklabels([str(min_val), str(max_val)])

def show_proj_w_dis(proj_file, distance=euclidean):
    proj_w = np.load(proj_file)
    proj_w_class = proj_w[np.arange(proj_w.shape[0]), 1::2]
    proj_w_class = np.transpose(proj_w_class)
    proj_dis = []
    for feature_i in proj_w_class:
        for feature_j in proj_w_class:
            proj_dis.append(distance(feature_i, feature_j))
            proj_dis[-1] = 0 if proj_dis[-1] < 0 else proj_dis[-1]
    proj_dis = np.reshape(np.array(proj_dis), [proj_w_class.shape[0], proj_w_class.shape[0]])
    min_dis, max_dis = np.min(proj_dis), np.max(proj_dis)
    fig, ax = plt.subplots()
    fig.suptitle(distance.__str__())
    cax = ax.imshow(proj_dis, cmap='gray', interpolation='nearest')
    cbar = fig.colorbar(cax, ticks=[min_dis, max_dis])
    cbar.ax.set_yticklabels([str(min_dis), str(max_dis)])

def show_attention(attn_file, model_index=0):
    attns = np.load(attn_file)
    model_attn = attns[model_index]
    print(model_attn)
    fig, ax = plt.subplots()
    fig.suptitle("%s-%d"%(attn_file[attn_file.rfind('/')+1:], model_index))
    cax = ax.imshow(model_attn, cmap="gray", interpolation="nearest", vmin=0, vmax=1)
    cbar = fig.colorbar(cax, ticks=[np.min(model_attn), np.max(model_attn)])
    cbar.ax.set_yticklabels([str(np.min(model_attn)), str(np.max(model_attn))])

def show_attention2(attn_file, model_index=0):
    attns = np.load(attn_file)
    model_attn = attns[model_index]
    classes, views = model_attn.shape[0], model_attn.shape[1]
    for i in range(classes):
        s = 0.0
        for j in range(views):
            s += (model_attn[i][j] * model_attn[i][j])
        for j in range(views):
            model_attn[i][j] = model_attn[i][j] * model_attn[i][j] / s
    fig, ax = plt.subplots()
    fig.suptitle("%s-%d" % (attn_file[attn_file.rfind('/') + 1:], model_index))
    cax = ax.imshow(model_attn, cmap="gray", interpolation="nearest")
    cbar = fig.colorbar(cax, ticks=[np.min(model_attn), np.max(model_attn)])
    cbar.ax.set_yticklabels([str(np.min(model_attn)), str(np.max(model_attn))])

def attn2txt(attn_weights, name="attn"):
    with open(name+'.txt', 'w') as f:
        for attn in attn_weights:
            f.write(' '.join([str(a) for a in attn])+'\n')

def show_metric_N(metric_dir, metric_name, N=1000):
    import os
    micro_p = np.load(os.path.join(metric_dir, "micro_p.npy"))
    micro_r = np.load(os.path.join(metric_dir, "micro_r.npy"))
    micro_f1 = np.load(os.path.join(metric_dir, "micro_f1.npy"))
    micro_mAP = np.load(os.path.join(metric_dir, "micro_mAP.npy"))
    micro_ndcg = np.load(os.path.join(metric_dir, "micro_ndcg.npy"))

    macro_p = np.load(os.path.join(metric_dir, "macro_p.npy"))
    macro_r = np.load(os.path.join(metric_dir, "macro_r.npy"))
    macro_f1 = np.load(os.path.join(metric_dir, "macro_f1.npy"))
    macro_mAP = np.load(os.path.join(metric_dir, "macro_mAP.npy"))
    macro_ndcg = np.load(os.path.join(metric_dir, "macro_ndcg.npy"))

    plt.subplots(1,1)
    plt.plot([i+1 for i in range(micro_p.shape[0])], micro_p.tolist(), label="micro_p")
    plt.plot([i + 1 for i in range(micro_r.shape[0])], micro_r.tolist(), label="micro_r")
    plt.plot([i + 1 for i in range(micro_f1.shape[0])], micro_f1.tolist(), label="micro_f1")
    plt.plot([i + 1 for i in range(micro_mAP.shape[0])], micro_mAP.tolist(), label="micro_mAP")
    plt.plot([i + 1 for i in range(micro_ndcg.shape[0])], micro_ndcg.tolist(), label="micro_ndcg")
    plt.plot([i + 1 for i in range(macro_ndcg.shape[0])], [0.77] * 1000, label="rmvcnn-micro-r")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)

    plt.subplots(1,1)
    plt.plot([i + 1 for i in range(macro_p.shape[0])], macro_p.tolist(), label="macro_p")
    plt.plot([i + 1 for i in range(macro_r.shape[0])], macro_r.tolist(), label="macro_r")
    plt.plot([i + 1 for i in range(macro_f1.shape[0])], macro_f1.tolist(), label="macro_f1")
    plt.plot([i + 1 for i in range(macro_mAP.shape[0])], macro_mAP.tolist(), label="macro_mAP")
    plt.plot([i + 1 for i in range(macro_ndcg.shape[0])], macro_ndcg.tolist(), label="macro_ndcg")
    plt.plot([i + 1 for i in range(macro_ndcg.shape[0])], [0.625] * 1000, label="rmvcnn-macro-r")


    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)

    plt.xlabel("N")


# 4, 7, 12
# 64,65,66
if __name__ == '__main__':
    # show_model_features(index=66, normalize=False)
    # plt.show()
    # show_embedding_dis(embeddings_file, distance=euclidean)
    # show_embedding_dis(embeddings_file, distance=cosine)
    # show_proj_w_dis(proj_file, distance=euclidean)
    # show_proj_w_dis("../ignore/data/proj_w_attn.npy", distance=euclidean)
    # show_proj_w_dis(proj_file, distance=cosine)
    # show_proj_w_dis("../ignore/data/proj_w_attn.npy", distance=cosine)
    # show_attention('../ignore/data/attention_weights_richdata.npy', model_index=567)
    show_metric_N('/home/shangmingyang/wuque/projects/evaluator/test_normal', 'macro_p')
    plt.show()
