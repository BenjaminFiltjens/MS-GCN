#!/usr/bin/python2.7
# adapted from: https://github.com/yabufarha/ms-tcn/blob/master/eval.py
# who adapted from: https://github.com/colincsl/TemporalConvolutionalNetworks/blob/master/code/metrics.py

import math

import numpy as np
import argparse


def read_file(path):
    with open(path, 'r') as f:
        content = f.read()
        f.close()
    return content


def get_labels_start_end_time(frame_wise_labels, bg_class=["background"]):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i + 1)
    return labels, starts, ends


def levenstein(p, y, norm=False):
    m_row = len(p)
    n_col = len(y)
    D = np.zeros([m_row + 1, n_col + 1], np.float)
    for i in range(m_row + 1):
        D[i, 0] = i
    for i in range(n_col + 1):
        D[0, i] = i

    for j in range(1, n_col + 1):
        for i in range(1, m_row + 1):
            if y[j - 1] == p[i - 1]:
                D[i, j] = D[i - 1, j - 1]
            else:
                D[i, j] = min(D[i - 1, j] + 1,
                              D[i, j - 1] + 1,
                              D[i - 1, j - 1] + 1)

    if norm:
        score = (1 - D[-1, -1] / max(m_row, n_col)) * 100
    else:
        score = D[-1, -1]

    return score


def edit_score(recognized, ground_truth, norm=True, bg_class=["background"]):
    P, _, _ = get_labels_start_end_time(recognized, bg_class)
    Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, norm)


def f_score(recognized, ground_truth, overlap, bg_class=["background"]):
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)

    tp = 0
    fp = 0

    hits = np.zeros(len(y_label))

    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0 * intersection / union) * ([p_label[j] == y_label[x] for x in range(len(y_label))])
        # Get the best scoring segment
        idx = np.array(IoU).argmax()

        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)


def mcc_score(recognized, ground_truth, overlap, bg_class=["background"]):
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    hits = np.zeros(len(y_label))

    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0 * intersection / union) * ([p_label[j] == y_label[x] for x in range(len(y_label))])
        # Get the best scoring segment
        idx = np.array(IoU).argmax()

        if IoU[idx] >= overlap and not hits[idx]:
            if(p_label[j]==0):
                tn += 1
            else:
                tp += 1
            hits[idx] = 1
        else:
            if (p_label[j] == 0):
                fn += 1
            else:
                fp += 1
    #fn = len(y_label) - sum(hits)
    # for i in range(0,len(y_label)):
    #     if(hits[i]==0):
    #         if(y_label[i]==1):
    #             fn += 1
    #         else:
    #             fp += 1
    #     else:
    #         if (y_label[i] == 1):
    #             tp += 1
    #         else:
    #             tn += 1

    return float(tp), float(fp), float(fn), float(tn)


def run_eval(split):
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default="final_dataset_6")
    args = parser.parse_args()
    print(args.dataset)
    ground_truth_path = "./data/" + args.dataset + "/groundTruth_/"
    recog_path = "./results/" + args.dataset + "/split_" + str(split) + "/"
    file_list = "./data/" + args.dataset + "/splits_loso_validation/test.split" + str(split) + ".bundle"

    list_of_videos = read_file(file_list).split('\n')[:-1]

    overlap=[0.5]
    tp2, fp2, fn2,tn2 = np.zeros(len(overlap)), np.zeros(len(overlap)), np.zeros(len(overlap)),np.zeros(len(overlap))
    tp, fp, fn = np.zeros(len(overlap)), np.zeros(len(overlap)), np.zeros(len(overlap))
    correct = 0
    total = 0
    edit = 0
    mcctn = 0
    mcctp = 0
    mccfn = 0
    mccfp = 0

    for vid in list_of_videos:

        gt_file = ground_truth_path + vid
        gt_content = list(np.loadtxt(gt_file).astype(int)[:-1])

        recog_file = recog_path + vid[:-4]
        recog_content = list(np.loadtxt(recog_file).astype(int)[:len(gt_content)])
        for i in range(len(gt_content)):
            total += 1
            if gt_content[i] == recog_content[i]:
                correct += 1
                if (gt_content[i] == 1):
                    mcctp += 1
                else:
                    mcctn += 1
            else:
                if (gt_content[i] == 1):
                    mccfn += 1
                else:
                    mccfp += 1

        edit += edit_score(recog_content, gt_content)

        for s in range(len(overlap)):
            tp1, fp1, fn1 = f_score(recog_content, gt_content, overlap[s])
            tp[s] += tp1
            fp[s] += fp1
            fn[s] += fn1

        for s in range(len(overlap)):
            tp1, fp1, fn1, tn1 = mcc_score(recog_content, gt_content, overlap[s])
            tp2[s] += tp1
            fp2[s] += fp1
            fn2[s] += fn1
            tn2[s] += tn1

        outputs = np.column_stack((np.asarray(recog_content), np.asarray(gt_content)))
        np.savetxt(recog_path + vid + '_annot.csv', outputs, delimiter=',')

    print('Split '+str(split))
    print("Acc: %.4f" % (100 * float(correct) / total))
    print('Edit: %.4f' % ((1.0 * edit) / len(list_of_videos)))
    for s in range(len(overlap)):
        precision = tp[s] / float(tp[s] + fp[s])
        recall = tp[s] / float(tp[s] + fn[s])

        f1 = 2.0 * (precision * recall) / (precision + recall)

        f1 = np.nan_to_num(f1) * 100
        mcc = ((tp2[s]*tn2[s])-(fn2[s]*fp2[s]))/(math.sqrt((tp2[s]+fp2[s])*(tp2[s]+fn2[s])*(tn2[s]+fp2[s])*(tn2[s]+fn2[s]))+0.0000001) #no div by zero
        print('Precision@%0.2f: %.4f' % (overlap[s], precision))
        print('Recall@%0.2f: %.4f' % (overlap[s], recall))
        print('F1@%0.2f: %.4f' % (overlap[s], f1))
        print('MCC@%0.2f: %.4f' % (overlap[s], mcc))
    frame_mcc = ((mcctp * mcctn) - (mccfn * mccfp)) / (math.sqrt((mcctp + mccfp) * (mcctp + mccfn) * (mcctn + mccfp) * (mcctn + mccfn)) + 0.0000001)
    print('Frame MCC: %.4f' % (frame_mcc))
    acc = 100 * float(correct) / total
    print('\n')
    return frame_mcc, f1, acc


def main():
    score = []
    f1s = []
    mccs = []
    accs = []
    for i in range(1,2):
        mcc_fr, f1, acc = run_eval(i)
        metric = ((mcc_fr * 100) + f1) / 2
        accs.append(acc)
        score.append(metric)
        f1s.append(f1)
        mccs.append(mcc_fr)
    print('score: ' + str(np.mean(score)))
    print('f1: ' + str(np.mean(f1s)))
    print('f1 sd: ' + str(np.std(f1s)))
    print('mcc: ' + str(np.mean(mccs)))
    print('mcc sd: ' + str(np.std(mccs)))
    print('Acc: ' + str(np.mean(accs)))


if __name__ == '__main__':
    main()
