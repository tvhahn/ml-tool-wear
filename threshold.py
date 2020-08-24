import numpy as np
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import random
import seaborn as sns

import tensorflow as tf

# from tensorflow import keras

from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    auc,
)
from sklearn.utils import shuffle

# see if we can control randomness in results
from numpy.random import seed

seed(1)
tf.random.set_seed(2)


class SelectThreshold:
    def __init__(
        self,
        model,
        X_train,
        y_train,
        X_train_slim,
        X_val,
        y_val,
        X_val_slim,
        class_to_remove,
        class_names,
        model_name,
        date_time,
    ):

        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_train_slim = X_train_slim
        self.X_val = X_val
        self.y_val = y_val
        self.X_val_slim = X_val_slim
        self.class_to_remove = class_to_remove
        self.class_names = class_names
        self.model_name = model_name
        self.date_time = date_time

    def mse(self, X_val, recon_val):
        """Calculate MSE for images in X_val and recon_val"""
        # need to calculate mean across the rows, and then across the columns
        try:
            # if this works, then you will be getting the mean across all the signals
            return np.mean(np.mean(np.square(X_val - recon_val), axis=1), axis=1)
        except:
            # if the above does not work, then it is assumed that you are only looking
            # for the mean of one signal -- therefore the below should work
            return np.mean(np.square(X_val - recon_val), axis=1)

    def rmse(self, X_val, recon_val):
        """Calculate RMSE for images in X_val and recon_val"""
        return np.sqrt(self.mse(X_val, recon_val))

    def euclidean_distance(self, X_val, recon_val):
        dist = np.linalg.norm(X_val - recon_val, axis=(1, 2))
        return dist

    # function that creates a pandas dataframe with the RMSE value, and the associated class
    def create_df_reconstruction(self, y_data, reconstruction_error_val, threshold_val):
        df = pd.DataFrame(data=reconstruction_error_val, columns=["metric"])

        class_names_list = list(zip(self.class_names, range(len(self.class_names))))

        y_names = []
        for i in y_data:
            y_names.append(str(i) + ", " + class_names_list[i][0])

        # append the class values
        df["class"] = y_data
        df["class_names"] = y_names

        # label anomolous (outlier) data as -1, inliers as 1
        # -1 (outlier) is POSITIVE class
        #  1 (inlier) is NEGATIVE class
        new_y_data = []
        for i in y_data:
            if i in self.class_to_remove:
                new_y_data.append(-1)
            else:
                new_y_data.append(1)

        df["true_class"] = new_y_data

        # add prediction based on threshold
        df["prediction"] = np.where(df["metric"] >= threshold_val, -1, 1)

        return df

    def threshold_grid_search(
        self,
        y_data,
        lower_bound,
        upper_bound,
        reconstruction_error_val,
        grid_iterations=10,
    ):
        """Simple grid search for finding the best threshold"""

        roc_scores = {}
        tprs = []  # true positive rates
        fprs = []  # false positive rates
        precisions = []
        recalls = []
        grid_search_count = 0
        for i in np.arange(
            lower_bound,
            upper_bound,
            (np.abs(upper_bound - lower_bound) / grid_iterations),
        ):
            #             if grid_search_count%50 == 0:
            #                 print('grid search iteration: ', grid_search_count)

            threshold_val = i
            df = self.create_df_reconstruction(
                y_data, reconstruction_error_val, threshold_val
            )
            roc_val = roc_auc_score(df["true_class"], df["prediction"])

            # fpr, tpr, thresholds = roc_curve(df['true_class'], df['prediction'], pos_label=-1)
            # pr_auc = auc(fpr, tpr)
            roc_scores[i] = roc_val
            grid_search_count += 1

            # calculate precision and recall
            # True Positive
            tp = len(df[(df["true_class"] == -1) & (df["prediction"] == -1)])
            # False Positive -- predict anomaly (-1), when it is actually normal (1)
            fp = len(df[(df["true_class"] == 1) & (df["prediction"] == -1)])
            # True Negative
            tn = len(df[(df["true_class"] == 1) & (df["prediction"] == 1)])
            # False Negative
            fn = len(df[(df["true_class"] == -1) & (df["prediction"] == 1)])

            # print('threshold val', i)
            # print('tp:',tp,'fp:', fp, 'tn:', tn, 'fn:',fn)

            try:

                # precision/recall
                pre_score = tp / (tp + fp)
                re_score = tp / (tp + fn)
                # tpr/fpr
                tpr = tp / (tp + fn)
                fpr = fp / (fp + tn)

                precisions.append(pre_score)
                recalls.append(re_score)
                tprs.append(tpr)
                fprs.append(fpr)

            except ZeroDivisionError as err:
                pass
                # print('Handling run-time error:', err)

        # return best roc_score and the threshold used to set it
        threshold_val = max(zip(roc_scores.values(), roc_scores.keys()))
        best_threshold = threshold_val[1]
        best_roc_score = threshold_val[0]
        # print('Best threshold:', '{:.5}'.format(best_threshold),'\tROC score: {:.2%}'.format(best_roc_score))

        return best_threshold, best_roc_score, precisions, recalls, tprs, fprs

    def box_plot(self, df, best_threshold, best_roc_score, metric):
        fig, ax = plt.subplots(figsize=(12, 5))
        df.boxplot(column=["metric"], by="class_names", ax=ax).axhline(
            y=best_threshold, c="red", alpha=0.7
        )
        plt.title("Boxplots of {} for X_valid, by Class".format(metric))
        plt.suptitle("")
        plt.show()

        print("\nConfusion Matrix:")
        print(confusion_matrix(df["true_class"], df["prediction"]))

    # function to test the different reconstruction methods (mse, rmse, euclidean)
    # do a grid search looking for the best threshold, and then outputting the results
    def compare_error_method(
        self,
        show_results=True,
        grid_iterations=10,
        model_results=None,
        model_result_cols=[],
        search_iterations=2,
        one_signal_only=False,
        signal_index=None,
    ):
        """Function to test the different reconstruction methods (mse, rmse, euclidean) 

        Parameters
        ===========
        model : tensorflow model
            autoencoder model that was trained on the "slim" data set.
            Will be used to build reconstructions

        X_val : ndarray
            tensor of the X validation set

        class_to_remove : ndarray
            numpy array of the classes to remove from the X_val and y_val data
        """

        col = [
            "model_name",
            "method",
            "best_threshold",
            "roc_train_score",
            "roc_valid_score",
            "pr_auc_train_score",
            "pr_auc_val_score",
            "date_time",
        ]
        result_table = pd.DataFrame(columns=col)
        # print(model_results)
        # print(type(model_results))
        # print(model_result_cols)



        for search_iter in range(search_iterations):
            print('search_iter:',search_iter)
            # build the reconstructions on the X_val_slim dataset, and the X_val dataset
            # recon_train_slim = self.model.predict(self.X_train_slim,batch_size=32)
            recon_train = self.model.predict(self.X_train, batch_size=64, verbose=1,)
            recon_val = self.model.predict(self.X_val, batch_size=64, verbose=1,)

            # run through each of the reconstruction error methods, perform a little grid search
            # to find the optimum value

            # _______MSE_______#
            # calculate MSE reconstruction error
            # mse_recon_train_slim = self.mse(self.X_train_slim, recon_train_slim) # for slim dataset

            # if we are doing the calculation for one signal only, then:
            if one_signal_only == True:
                mse_recon_train = self.mse(
                    self.X_train[:,:,signal_index], recon_train[:,:,signal_index]
                )  # for complete train dataset
                mse_recon_val = self.mse(
                    self.X_val[:,:,signal_index], recon_val[:,:,signal_index]
                )  # for complete validation dataset

            else:

                mse_recon_train = self.mse(
                    self.X_train, recon_train
                )  # for complete train dataset
                mse_recon_val = self.mse(
                    self.X_val, recon_val
                )  # for complete validation dataset



            # calculate pr-auc and roc-auc for train data set
            lower_bound = np.min(mse_recon_train)
            upper_bound = np.max(mse_recon_train)
            (
                best_threshold,
                _,
                precisions,
                recalls,
                tprs,
                fprs,
            ) = self.threshold_grid_search(
                self.y_train, lower_bound, upper_bound, mse_recon_train, grid_iterations
            )

            pr_auc_score_train = auc(recalls, precisions)
            roc_auc_score_train = auc(fprs, tprs)

            # calculate pr-auc and roc-auc for train data set
            lower_bound = np.min(mse_recon_val)
            upper_bound = np.max(mse_recon_val)
            _, _, precisions, recalls, tprs, fprs = self.threshold_grid_search(
                self.y_val, lower_bound, upper_bound, mse_recon_val, grid_iterations
            )

            pr_auc_score_val = auc(recalls, precisions)
            roc_auc_score_val = auc(fprs, tprs)

            # print('ROC_train: {:.2%}'.format(roc_auc_score_train),'ROC_val: {:.2%}'.format(roc_auc_score_val), '\nPR_auc_train: {:.2%}'.format(pr_auc_score_train),'PR_auc_val: {:.2%}'.format(pr_auc_score_val))

            # calculate pr-auc score
            # pr_auc_score_train = auc(recalls, precisions)

            # check the results on the validation set
            # df_val = self.create_df_reconstruction(self.y_val, mse_recon_val, best_threshold)
            # roc_df_val = roc_auc_score(df_val['true_class'], df_val['prediction'])
            # print('\tROC score validation: {:.2%}'.format(roc_df_val))

            # fpr, tpr, thresholds = roc_curve(df_val['true_class'], df_val['prediction'], pos_label=-1)
            # pr_auc_score_val = auc(fpr, tpr)
            col = [
                "model_name",
                "method",
                "best_threshold",
                "roc_train_score",
                "roc_valid_score",
                "pr_auc_train_score",
                "pr_auc_val_score",
                "date_time",
            ]
            result_table = result_table.append(
                pd.DataFrame(
                    [
                        [
                            self.model_name,
                            "mse",
                            best_threshold,
                            roc_auc_score_train,
                            roc_auc_score_val,
                            pr_auc_score_train,
                            pr_auc_score_val,
                            self.date_time,
                        ]
                    ],
                    columns=col,
                ),
                sort=False,
            )

            # else:
            #     result_table = result_table.append(pd.DataFrame([[self.model_name,'mse',
            #                                                     best_threshold,
            #                                                     roc_auc_score_train,roc_auc_score_val,pr_auc_score_train,pr_auc_score_val,self.date_time]+model_results[0]],
            #                                                     columns=col),sort=False)

        if model_results == None:
            result_table = result_table.groupby(
                by=["model_name", "method", "date_time"], as_index=False
            ).mean()

        else:
            result_table = result_table.groupby(
                by=["model_name", "method", "date_time"], as_index=False
            ).mean()
            result_table = pd.concat(
                [result_table, pd.DataFrame(model_results, columns=model_result_cols)],
                axis=1,
                sort=False,
            )


        if one_signal_only == True:
            result_table['signal_index'] = signal_index
            return result_table
        else:
            return result_table
