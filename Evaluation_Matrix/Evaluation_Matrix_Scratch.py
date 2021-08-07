
#For Both Multiclass and Binaryclass Variables
import matplotlib.pyplot as plt

class evaluation_metric:
    CM=[]
    macro_pre=[]
    macro_rec=[]
    def __init__(self):
        pass
    def confusion_matrix(self,oy,py): #confusion matrix
        cm=[]
        for i in range(len(np.unique(oy))):
            cm.append([0]*len(np.unique(oy)))
        for i in range(len(oy)):
            cm[py[i]][oy[i]] += 1
        return cm            
    def accuracy(self,oy,py): #accuracy 
        sc = 0
        for i in range(len(oy)):
            if(oy[i] == py[i]):
                sc += 1
        return (sc/len(oy))*100
    #Macro Average Values
    def Macro_recall(self,oy,py):
        p=[]
        self.CM=self.confusion_matrix(oy,py)
        column_sm=list(map(sum, zip(*self.CM)))
        for i in range(len(column_sm)):
            p.append(self.CM[i][i]/column_sm[i])
        self.macro_rec=p
        return sum(p)/len(p)
    def Macro_precision(self,oy,py):
        p = []
        self.CM = self.confusion_matrix(oy, py)
        row_sm=[]
        for i in range(len(self.CM)):
            row_sm.append(sum(self.CM[i]))
        for i in range(len(row_sm)):
            p.append(self.CM[i][i]/row_sm[i])
        self.macro_pre=p
        return sum(p)/len(p)
    def Macro_F1_score(self,oy,py):
        r=self.Macro_recall(oy,py)
        p=self.Macro_precision(oy,py)
        f1s=0
        for i in range(len(self.macro_pre)):
            f1s+=(2*((self.macro_pre[i]*self.macro_rec[i])/(self.macro_pre[i]+self.macro_rec[i])))
        return f1s/len(self.macro_pre)
    #Micro Average Values
    def Micro_recall(self,oy,py):
        self.CM=self.confusion_matrix(oy,py)
        total_CM_sum=sum(sum(self.CM,[]))
        micro_confusion=[[0,0],[0,0]]
        for i in range(len(self.CM)):
            micro_confusion[0][0]+=self.CM[i][i]
            micro_confusion[1][1]+=total_CM_sum-self.CM[i][i]
        micro_confusion[0][1]=total_CM_sum-micro_confusion[0][0]
        micro_confusion[1][0]=total_CM_sum-micro_confusion[0][0]
        return micro_confusion[0][0]/(micro_confusion[0][0]+micro_confusion[0][1])
    def Micro_precision(self,oy,py):
        self.CM=self.confusion_matrix(oy,py)
        return self.Micro_recall(oy,py)
    def Micro_F1_score(self,oy,py):
        return 2*((self.Micro_precision(oy, py)*self.Micro_recall(oy, py))/(self.Micro_precision(oy, py)+self.Micro_recall(oy, py)))
    # ROC Curve
    def ROC_Curve(self,pprob,y_test):
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.plot([0,1],[0,1])
        for col in range(len(pprob[0])):
            thres_y_test = (y_test == col)*1
            prob1 = pprob[:, col]
            tpr = 0
            fpr = 0
            threshold = [0, 0.1, 0.2,0.3, 0.4,0.5, 0.6,0.7, 0.8,0.9, 1]
            TPR = []
            FPR = []
            for i in threshold:
                thres = (prob1 >= i)
                thres = (thres*1)
                tp = 0
                fn = 0
                fp = 0
                tn = 0
                for i in range(len(thres)):
                    if(thres_y_test[i] == thres[i]):
                        if(thres_y_test[i] == 0):
                            tn += 1
                        else:
                            tp += 1
                    else:
                        if(thres_y_test[i] == 0 and thres[i] == 1):
                            fp += 1
                        elif(thres_y_test[i] == 1 and thres[i] == 0):
                            fn += 1
                
                tpr = tp/(tp+fn)
                fpr = fp/(tn+fp)
                # print(col,tpr,fpr)
                TPR.append(tpr)
                FPR.append(fpr)
            plt.plot(FPR, TPR, marker='o',label=col)
        plt.legend()
        plt.show()