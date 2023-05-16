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

def confusion_matrix_torch(y_true, y_pred, labels):
    # This is a simple implementation of confusion matrix using PyTorch
    # It assumes that y_true and y_pred are of the same shape and labels are [0, 1]
    TP = ((y_true == labels[1]) & (y_pred == labels[1])).sum()
    TN = ((y_true == labels[0]) & (y_pred == labels[0])).sum()
    FP = ((y_true == labels[0]) & (y_pred == labels[1])).sum()
    FN = ((y_true == labels[1]) & (y_pred == labels[0])).sum()
    return TN, FP, FN, TP

def DifferenceEqualOpportunity(y_pred, y_real, SensitiveCat, outcome, privileged, unprivileged, labels):
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
    y_priv = y_pred[y_real[SensitiveCat] == privileged]
    y_real_priv = y_real[y_real[SensitiveCat] == privileged][outcome]
    y_unpriv = y_pred[y_real[SensitiveCat] == unprivileged]
    y_real_unpriv = y_real[y_real[SensitiveCat] == unprivileged][outcome]
    TN_priv, FP_priv, FN_priv, TP_priv = confusion_matrix_torch(y_real_priv, y_priv, labels)
    TN_unpriv, FP_unpriv, FN_unpriv, TP_unpriv = confusion_matrix_torch(y_real_unpriv, y_unpriv, labels)

    epsilon = 1e-10
    return abs(TP_unpriv.float() / (TP_unpriv + FN_unpriv + epsilon) - TP_priv.float() / (TP_priv + FN_priv + epsilon))


def DifferenceAverageOdds(y_pred, y_real, SensitiveCat, outcome, privileged, unprivileged, labels):
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
    y_real_priv = y_real[y_real[SensitiveCat] == privileged][outcome]
    y_unpriv = y_pred[y_real[SensitiveCat] == unprivileged]
    y_real_unpriv = y_real[y_real[SensitiveCat] == unprivileged][outcome]
    TN_priv, FP_priv, FN_priv, TP_priv = confusion_matrix_torch(y_real_priv, y_priv, labels)
    TN_unpriv, FP_unpriv, FN_unpriv, TP_unpriv = confusion_matrix_torch(y_real_unpriv, y_unpriv, labels)

    epsilon = 1e-10
    return 0.5 * (abs(FP_unpriv.float() / (FP_unpriv + TN_unpriv + epsilon) - FP_priv.float() / (
                FP_priv + TN_priv + epsilon)) + abs(
        TP_unpriv.float() / (TP_unpriv + FN_unpriv + epsilon) - TP_priv.float() / (TP_priv + FN_priv + epsilon)))

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
