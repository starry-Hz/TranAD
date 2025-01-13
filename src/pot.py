import numpy as np

from src.spot import SPOT
from src.constants import *
from sklearn.metrics import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def calculate_metric(label, score,fig_name='roc_curve_1'):
    # ---------------------------------------------------------------------------------------------
    # After adjusting the prediction labels, calculate ROC curve and AUC:
    fpr, tpr, thresholds = roc_curve(label, score)
    roc_auc = auc(fpr, tpr)
    # 把fpr,tpr,roc_auc保存到roc_curve.csv文件中
    df = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'Thresholds': thresholds})
    df['AUC'] = roc_auc
    df.to_csv('roc_curve.csv', index=False)
    # print('ROC curve and AUC saved to roc_curve.csv')
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line (random classifier)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig(f'{fig_name}.png')

    return roc_auc
    # ---------------------------------------------------------------------------------------------


def plot_roc_curve(fpr, tpr,fig_name='TranAD_els'):
    # plt.figure(figsize=(8,6))
    # plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve')
    # plt.plot([0,1], [0,1], color='red', lw=2, linestyle='--', label='Random guess')
    # plt.xlabel('False Positive Rate (FPR)')
    # plt.ylabel('True Positive Rate (TPR)')
    # plt.title('ROC Curve')
    # plt.legend(loc='lower right')
    # plt.grid(True)
    # # plt.show()
    # plt.savefig('roc_curve_3.png')
    roc_auc = auc(fpr, tpr)
    # 把fpr,tpr,roc_auc保存到roc_curve.csv文件中
    df = pd.DataFrame({'fpr': fpr, 'tpr': tpr})
    df['AUC'] = roc_auc
    df.to_csv('roc_curve.csv', index=False)
    # print('ROC curve and AUC saved to roc_curve.csv')
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.6f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line (random classifier)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig(f'{fig_name}.png')


def compute_tpr_fpr(score, label, thresholds):
    tpr_list = []
    fpr_list = []
    for thresh in thresholds:
        preds = adjust_predicts(score, label, threshold=thresh)
        # 计算 TP, FP, FN, TN
        TP = np.sum((preds == 1) & (label > 0.1))
        FP = np.sum((preds == 1) & (label <= 0.1))
        FN = np.sum((preds == 0) & (label > 0.1))
        TN = np.sum((preds == 0) & (label <= 0.1))
        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
        tpr_list.append(TPR)
        fpr_list.append(FPR)
    return tpr_list, fpr_list



def calc_point2point(predict, actual):
    """
    calculate f1 score by predict and actual.
    Args:
        predict (np.ndarray): the predict label
        actual (np.ndarray): np.ndarray
    """
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)
    accuracy = (TP + TN) / (TP + TN + FP + FN + 0.00001)
    try:
        roc_auc = roc_auc_score(actual, predict)
        # roc_auc = calculate_metric(actual, predict, fig_name='roc_curve_2')
        # print('roc_auc:', roc_auc)
    except:
        roc_auc = 0
    return f1, precision, recall, TP, TN, FP, FN, roc_auc, accuracy


# the below function is taken from OmniAnomaly code base directly
# # 以下函数直接取自 OmniAnomaly 代码库
def adjust_predicts(score, label,
                    threshold=None,
                    pred=None,
                    calc_latency=False):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.
    Args:
        score (np.ndarray): The anomaly score
        label (np.ndarray): The ground-truth label
        threshold (float): The threshold of anomaly score.
            A point is labeled as "anomaly" if its score is lower than the threshold.
        pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
        calc_latency (bool):
    Returns:
        np.ndarray: predict labels
    score:异常分数
    label:真实标签
    threshold:异常分数的阈值
    pred:预测标签
    calc_latency:是否计算延迟
    """
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    if pred is None:
        predict = score > threshold
    else:
        predict = pred
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
                anomaly_state = True
                anomaly_count += 1
                for j in range(i, 0, -1):
                    if not actual[j]:
                        break
                    else:
                        if not predict[j]:
                            predict[j] = True
                            latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict


def calc_seq(score, label, threshold, calc_latency=False):
    """
    Calculate f1 score for a score sequence
    """
    if calc_latency:
        predict, latency = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        t = list(calc_point2point(predict, label))
        t.append(latency)
        return t
    else:
        predict = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        return calc_point2point(predict, label)


def bf_search(score, label, start, end=None, step_num=1, display_freq=1, verbose=True):
    """
    Find the best-f1 score by searching best `threshold` in [`start`, `end`).
    Returns:
        list: list for results
        float: the `threshold` for best-f1
    """
    if step_num is None or end is None:
        end = start
        step_num = 1
    search_step, search_range, search_lower_bound = step_num, end - start, start
    if verbose:
        print("search range: ", search_lower_bound, search_lower_bound + search_range)
    threshold = search_lower_bound
    m = (-1., -1., -1.)
    m_t = 0.0
    for i in range(search_step):
        threshold += search_range / float(search_step)
        target = calc_seq(score, label, threshold, calc_latency=True)
        if target[0] > m[0]:
            m_t = threshold
            m = target
        if verbose and i % display_freq == 0:
            print("cur thr: ", threshold, target, m, m_t)
    print(m, m_t)
    return m, m_t


def pot_eval(init_score, score, label, q=1e-5, level=0.02):
    """
    Run POT method on given score.      # 实现了基于峰值超阈值(POT)的异常检测方法
    Args:
        init_score (np.ndarray): The data to get init threshold.
            it should be the anomaly score of train set.
        score (np.ndarray): The data to run POT method.
            it should be the anomaly score of test set.
        label:
        q (float): Detection level (risk)
        level (float): Probability associated with the initial threshold t
    Returns:
        dict: pot result dict
    """
    # lms表示风险阈值和常量值，lm的第一个值是风险阈值，第二个值是常量值
    # q:什么是风险阈值，什么是常量值
    # 
    lms = lm[0]
    # print('lms:', lms)  # els数据集 TranAD模型 lms: 0.63
    while True:
        try:
            s = SPOT(q)  # SPOT object,SPOT对象，q 是风险阈值
            # print(f"s_142:{s}")
            s.fit(init_score, score)  # data import,用 训练集(init_score)和 测试集(score)训练模型
            # print(f"s_144:{s}")
            s.initialize(level=lms, min_extrema=False, verbose=False)  # initialization step
            # print('lms:', lms)
        except: lms = lms * 0.999
        else: break
    ret = s.run(dynamic=False)  # run   执行SPOT方法的结果中提取阈值列表
    # print(len(ret['alarms']))
    # print(len(ret['thresholds']))
    # lm[1]是常量值,调整最终的阈值
    pot_th = np.mean(ret['thresholds']) * lm[1]
    # print('pot_th:', pot_th)
    # pot_th = np.percentile(score, 100 * lm[0])
    # np.percentile(score, 100 * lm[0])
    #---------------------------------------------------------------------------------------------
    # 绘制ROC曲线和计算AUC
    # roc_auc_score = calculate_metric(label, score)
    # print('roc_auc_score:', roc_auc_score)
    # score_plot = np.asarray(score)
    # label_plot = np.asarray(label)
    # pot_plot = np.arange(0, 1.01, 0.01)
    # tpr, fpr = compute_tpr_fpr(score_plot, label_plot, pot_plot)
    # plot_roc_curve(fpr, tpr)
    #---------------------------------------------------------------------------------------------
    pred, p_latency = adjust_predicts(score, label, pot_th, calc_latency=True)
    # DEBUG - np.save(f'{debug}.npy', np.array(pred))
    # DEBUG - print(np.argwhere(np.array(pred)))
    p_t = calc_point2point(pred, label)
    # print('POT result: ', p_t, pot_th, p_latency)
    return {
        'f1': p_t[0],
        'precision': p_t[1],
        'recall': p_t[2],
        'TP': p_t[3],
        'TN': p_t[4],
        'FP': p_t[5],
        'FN': p_t[6],
        'ROC/AUC': p_t[7],
        'threshold': pot_th,
        # 'pot-latency': p_latency
        'accuracy': p_t[8]
    }, np.array(pred)
