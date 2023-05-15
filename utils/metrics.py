from sklearn.metrics import confusion_matrix

# 这段代码定义了一个函数accuracy，它用来计算分类器的准确率。
# 它的输入是y_true和y_pred，两者均为numpy数组或Pandas Series，分别表示真实标签和预测标签。
# 函数的计算过程是先计算y_true和y_pred差值的绝对值，然后对它们求平均值，并用1减去这个平均值得到准确率。
# 因为分类器的预测结果是0或1，所以差值的绝对值即为分类器的错误率，1减去错误率即为准确率。
def accuracy(y_true, y_pred):
    return 1-(abs(y_true - y_pred)).mean()

def discrimination(y_real,y_pred,SensitiveCat,privileged,unprivileged):
    y_priv = y_pred[y_real[SensitiveCat] == privileged]
    y_unpriv = y_pred[y_real[SensitiveCat] == unprivileged]
    return abs(y_priv.mean()-y_unpriv.mean())

def consistency(X,y_pred,k=5):
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(X)
    y=0
    N = X.shape[0]
    for i in range(N):
        distances, indices = nbrs.kneighbors(X[i,:].reshape(1,-1))
        #THE FIRST INDEX IS ALWAYS THE SAME SAMPLE -> REMOVE [1:]\|/
        y+=abs(y_pred.iloc[i] - y_pred.iloc[indices.tolist()[0][1:]].sum())
    return 1-y/(N*k)

def DifferenceEqualOpportunity(y_pred,y_real,SensitiveCat, outcome, privileged, unprivileged, labels):
    '''
    ABS Difference in True positive Rate between the two groups
    :param y_pred: prediction
    :param y_real: real label
    :param SensitiveCat: Sensitive feature name
    :param outcome: Outcome feature name
    :param privileged: value of the privileged group
    :param unprivileged: value of the unprivileged group
    :param labels: both priv-unpriv value for CFmatrix
    :return:
    '''
    y_priv = y_pred[y_real[SensitiveCat]==privileged]
    y_real_priv = y_real[y_real[SensitiveCat]==privileged]
    y_unpriv = y_pred[y_real[SensitiveCat]==unprivileged]
    y_real_unpriv = y_real[y_real[SensitiveCat]==unprivileged]
    TN_priv, FP_priv, FN_priv, TP_priv = confusion_matrix(y_real_priv[outcome],y_priv, labels=labels).ravel()
    TN_unpriv, FP_unpriv, FN_unpriv, TP_unpriv = confusion_matrix(y_real_unpriv[outcome], y_unpriv, labels=labels).ravel()

    # 添加一个小的正常数以防止除以零
    epsilon = 1e-10
    return abs(TP_unpriv/(TP_unpriv+FN_unpriv+epsilon) - TP_priv/(TP_priv+FN_priv+epsilon))

def DifferenceAverageOdds(y_pred,y_real,SensitiveCat, outcome, privileged, unprivileged,labels):
    '''
    Mean ABS difference in True positive rate and False positive rate of the two groups
    :param y_pred:
    :param y_real:
    :param SensitiveCat:
    :param outcome:
    :param privileged:
    :param unprivileged:
    :param labels:
    :return:
    '''
    y_priv = y_pred[y_real[SensitiveCat] == privileged]
    y_real_priv = y_real[y_real[SensitiveCat] == privileged]
    y_unpriv = y_pred[y_real[SensitiveCat] == unprivileged]
    y_real_unpriv = y_real[y_real[SensitiveCat] == unprivileged]
    TN_priv, FP_priv, FN_priv, TP_priv = confusion_matrix(y_real_priv[outcome], y_priv,  labels=labels).ravel()
    TN_unpriv, FP_unpriv, FN_unpriv, TP_unpriv = confusion_matrix(y_real_unpriv[outcome], y_unpriv,  labels=labels).ravel()

    # 添加一个小的正常数以防止除以零
    epsilon = 1e-10
    return 0.5*(abs(FP_unpriv/(FP_unpriv+TN_unpriv+epsilon)-FP_priv/(FP_priv+TN_priv+epsilon))+abs(TP_unpriv/(TP_unpriv+FN_unpriv+epsilon)-TP_priv/(TP_priv+FN_priv+epsilon)))


# def DifferenceEqualOpportunity(y_pred,y_real,SensitiveCat, outcome, privileged, unprivileged, labels):
#     '''
#     ABS Difference in True positive Rate between the two groups
#     :param y_pred: prediction
#     :param y_real: real label
#     :param SensitiveCat: Sensitive feature name
#     :param outcome: Outcome feature name
#     :param privileged: value of the privileged group
#     :param unprivileged: value of the unprivileged group
#     :param labels: both priv-unpriv value for CFmatrix
#     :return:
#     '''
#     y_priv = y_pred[y_real[SensitiveCat]==privileged]
#     y_real_priv = y_real[y_real[SensitiveCat]==privileged]
#     y_unpriv = y_pred[y_real[SensitiveCat]==unprivileged]
#     y_real_unpriv = y_real[y_real[SensitiveCat]==unprivileged]
#     TN_priv, FP_priv, FN_priv, TP_priv = confusion_matrix(y_real_priv[outcome],y_priv, labels=labels).ravel()
#     TN_unpriv, FP_unpriv, FN_unpriv, TP_unpriv = confusion_matrix(y_real_unpriv[outcome], y_unpriv, labels=labels).ravel()
#
#     # 添加一个小的正常数以防止除以零
#     epsilon = 1e-10
#     return abs(TP_unpriv/(TP_unpriv+FN_unpriv) - TP_priv/(TP_priv+FN_priv+epsilon))



# def DifferenceAverageOdds(y_pred,y_real,SensitiveCat, outcome, privileged, unprivileged,labels):
#     '''
#     Mean ABS difference in True positive rate and False positive rate of the two groups
#     :param y_pred:
#     :param y_real:
#     :param SensitiveCat:
#     :param outcome:
#     :param privileged:
#     :param unprivileged:
#     :param labels:
#     :return:
#     '''
#     y_priv = y_pred[y_real[SensitiveCat] == privileged]
#     y_real_priv = y_real[y_real[SensitiveCat] == privileged]
#     y_unpriv = y_pred[y_real[SensitiveCat] == unprivileged]
#     y_real_unpriv = y_real[y_real[SensitiveCat] == unprivileged]
#     TN_priv, FP_priv, FN_priv, TP_priv = confusion_matrix(y_real_priv[outcome], y_priv,  labels=labels).ravel()
#     TN_unpriv, FP_unpriv, FN_unpriv, TP_unpriv = confusion_matrix(y_real_unpriv[outcome], y_unpriv,  labels=labels).ravel()
#     return 0.5*(abs(FP_unpriv/(FP_unpriv+TN_unpriv)-FP_priv/(FP_priv+TN_priv))+abs(TP_unpriv/(TP_unpriv+FN_unpriv)-TP_priv/(TP_priv+FN_priv)))
