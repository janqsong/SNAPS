import torch
import numpy as np
import pandas as pd

import torch.nn.functional as F

from gnn_cp.cp.graph_cp import GraphCP
import gnn_cp.cp.transformations as cp_t
import gnn_cp.cp.graph_transformations as cp_gt
from gnn_cp.data.data_manager import GraphDataManager

class CPManager(object):
    def __init__(self, dataset, coverage_val, tune_fraction, calib_fraction, notune_calib_fraction,
                 dataset_name, edge_index, test_idx, adj_knn):
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.num_classes = dataset.y.max().cpu().numpy() + 1
        self.label_mask = F.one_hot(dataset.y).bool()
        self.edge_index = edge_index
        self.num_nodes = dataset.x.shape[0]
        self.coverage_val = coverage_val
        self.tune_fraction = tune_fraction
        self.calib_fraction = calib_fraction
        self.notune_calib_fraction = notune_calib_fraction
        self.n_iters = 100
        
        self.test_idx = test_idx
        
        self.adj_knn = adj_knn
        
        self.embeddings = None
        
        
        self.metrics_dict = {
            "empi_coverage": lambda pred_set, true_mask: GraphCP.coverage(pred_set, true_mask),
            "average_set_size": lambda pred_set, true_mask: GraphCP.average_set_size(pred_set),
            "singleton_hit": lambda pred_set, true_mask: self.singleton_hit(pred_set, true_mask)
        }
        
        self.get_cp_results_fun = {
            "APS": self.get_aps_result,
            "RAPS": self.get_raps_result,
            "DAPS": self.get_daps_result,
            "SNAPS": self.get_snaps_result,
            "TSNAPS": self.get_temp_snaps_result,
        }
        
        self.base_scores = {
            "APS": None
        }
    
    def singleton_hit(self, pred_set, true_mask):
        one_sized_pred = (pred_set.sum(axis=1) == 1)
        result = pred_set[true_mask][one_sized_pred].sum().item() / pred_set.shape[0]
        return result
    
    def get_all_cp_results(self, cp_keys, embeddings, tune_idx, test_idx, instance_test_idx):
        res_combined_mean = []
        res_combined_std = []
        for cp_key in cp_keys:
            keys = cp_key.split('-')
            if len(keys) == 2:
                baseline_scores = self.base_scores[keys[1]]
                if baseline_scores is None:
                    self.get_cp_results_fun[keys[1]](embeddings, tune_idx, test_idx, instance_test_idx)
            else:
                baseline_scores = embeddings
            
            results = self.get_cp_results_fun[keys[0]](baseline_scores, tune_idx, test_idx, instance_test_idx)
            res_combined_mean.append(results.mean().drop("attempt"))
            res_combined_std.append(results.std().drop("attempt"))
        res_combined = pd.concat([
            pd.concat(res_combined_mean, axis=1, keys=cp_keys).T,
            pd.concat(res_combined_std, axis=1, keys=cp_keys).T
        ], axis=1, keys=['mean', 'std'])

        return res_combined

    def get_aps_result(self, embeddings, tune_idx, test_idx, instance_test_idx):
        aps_cp = GraphCP(transformation_sequence=[cp_t.APSTransformation(softmax=True)])
        aps_scores = aps_cp.get_scores_from_logits(embeddings)
        aps_result = self.fair_shuffle_test_multiple_metrics(aps_scores[instance_test_idx], self.label_mask[instance_test_idx], 
                                        coverage_guarantee=self.coverage_val, calib_fraction=self.notune_calib_fraction)
        
        # print("aps", torch.max(aps_scores), torch.min(aps_scores))
        
        self.base_scores['APS'] = aps_scores
        return aps_result
    
    def get_raps_result(self, embeddings, tune_idx, test_idx, instance_test_idx):
        _, raps_params = self.find_all_raps_params(embeddings, tune_idx)
        best_k_reg = raps_params[0]
        best_penalty = raps_params[1]
        
        raps_cp = GraphCP(transformation_sequence=[cp_t.RAPSTransformation(softmax=True, k_reg=best_k_reg, penalty=best_penalty)])
        raps_scores = raps_cp.get_scores_from_logits(embeddings)
        raps_result = GraphCP([], coverage_guarantee=self.coverage_val).shuffle_test_multiple_metrics(
            raps_scores[test_idx], self.label_mask[test_idx], metrics_dict=self.metrics_dict, calib_fraction=self.calib_fraction, n_iters=self.n_iters)
        
        self.base_scores['RAPS'] = raps_scores
        return raps_result
    
    def get_daps_result(self, baseline_scores, tune_idx, test_idx, instance_test_idx):
        _, daps_params = self.find_all_daps_params(baseline_scores, tune_idx, self.edge_index)
        best_lambda_val = daps_params
        
        daps_scores = cp_gt.VertexMPTransformation(neigh_coef=best_lambda_val, edge_index=self.edge_index, n_vertices=self.num_nodes).pipe_transform(baseline_scores)
        daps_result = GraphCP([], coverage_guarantee=self.coverage_val).shuffle_test_multiple_metrics(
            daps_scores[test_idx], self.label_mask[test_idx], metrics_dict=self.metrics_dict, calib_fraction=self.calib_fraction, n_iters=self.n_iters)
        
        # print("daps", torch.max(daps_scores), torch.min(daps_scores))
        
        return daps_result
    
    def get_snaps_result(self, baseline_scores, tune_idx, test_idx, instance_test_idx):
        snaps_cp = cp_gt.SNAPSTransformation(edge_index=self.edge_index, n_vertices=self.num_nodes, adj_knn=self.adj_knn)
        snaps_scores = snaps_cp.pipe_transform(baseline_scores)
        
        snaps_result = self.fair_shuffle_test_multiple_metrics(snaps_scores[instance_test_idx], self.label_mask[instance_test_idx], 
                                    coverage_guarantee=self.coverage_val, calib_fraction=self.notune_calib_fraction)
        # print("snaps", torch.max(snaps_scores), torch.min(snaps_scores))
        self.base_scores['SNAPS'] = snaps_scores
        return snaps_result
    
    def get_temp_snaps_result(self, baseline_scores, tune_idx, test_idx, instance_test_idx):
        _, snaps_params = self.find_all_snaps_params(baseline_scores, tune_idx)
        best_edge_val, best_feature_val = snaps_params
        # print("best_edge_val", best_edge_val, best_feature_val)
        
        snaps_scores = cp_gt.TEMPSNAPSTransformation(edge_index=self.edge_index, n_vertices=self.num_nodes, adj_knn=self.adj_knn, edge_val=best_edge_val, feature_val=best_feature_val).pipe_transform(baseline_scores)
        snaps_result = GraphCP([], coverage_guarantee=self.coverage_val).shuffle_test_multiple_metrics(
            snaps_scores[test_idx], self.label_mask[test_idx], metrics_dict=self.metrics_dict, calib_fraction=self.calib_fraction, n_iters=self.n_iters)
        
        # self.base_scores['SNAPS'] = snaps_scores
        
        return snaps_result
    
    def find_all_raps_params(self, embeddings, tune_idx, n_iterations=1):
        k_regs = np.arange(0, self.num_classes, 1).astype(int)
        penalties = np.array([0.001, 0.01, 0.1, 0.2, 0.5, 1.0, 1.5])
        
        iteration_results = []
        overal_regular_results = []
        for _ in range(n_iterations):
            baseline_scores = cp_t.APSTransformation(softmax=True).pipe_transform(embeddings)
            base_cp = GraphCP([], coverage_guarantee=self.coverage_val)
            base_cp.calibrate_from_scores(baseline_scores[tune_idx], self.label_mask[tune_idx])
            baseline_pred_sets = base_cp.predict_from_scores(baseline_scores[tune_idx])
            
            overal_regular_results = [{"k_reg": 0, "penalty": 0, "average_set_size": base_cp.average_set_size(baseline_pred_sets)}]
            for k_reg in k_regs:
                for penalty in penalties:
                    raps_cp = GraphCP(transformation_sequence=[cp_t.RAPSTransformation(softmax=True, k_reg=k_reg, penalty=penalty)])
                    raps_scores = raps_cp.get_scores_from_logits(embeddings)
                    cp = GraphCP([], coverage_guarantee=self.coverage_val)

                    cp.calibrate_from_scores(raps_scores[tune_idx], self.label_mask[tune_idx])
                    pred_set = cp.predict_from_scores(raps_scores[tune_idx])

                    overal_regular_results.append({
                        "k_reg": k_reg, "penalty": penalty,
                        "average_set_size": cp.average_set_size(pred_set)
                    })
                    # print("k_reg", k_reg, "penalty", penalty)
            overal_regular_results = pd.DataFrame(overal_regular_results)
            baseline_res = overal_regular_results.loc[(overal_regular_results["k_reg"] == 0) & (overal_regular_results["penalty"] == 0)]["average_set_size"].values[0]
            overal_regular_results["enhancement"] = overal_regular_results["average_set_size"] - baseline_res
            iteration_results.append(overal_regular_results[["enhancement"]])
        iteration_results = pd.concat(iteration_results, axis=1)
        iteration_mean = iteration_results.mean(axis=1)
        best_param_sets = overal_regular_results.loc[iteration_mean.idxmin()]
        best_params = (best_param_sets["k_reg"], best_param_sets["penalty"])
        return overal_regular_results, best_params
    
    def find_all_daps_params(self, baseline_scores, tune_idx, edge_index, n_iterations=1):
        lambda_vals = np.arange(0.05, 1, 0.05).round(3)

        iteration_results = []
        for _ in range(n_iterations):
            base_cp = GraphCP([], coverage_guarantee=self.coverage_val)
            base_cp.calibrate_from_scores(baseline_scores[tune_idx], self.label_mask[tune_idx])
            baseline_pred_set = base_cp.predict_from_scores(baseline_scores[tune_idx])

            overall_mixing_results = [{"lambda": 0, "average_set_size": base_cp.average_set_size(baseline_pred_set)}]
            for lambda_v in lambda_vals:
                mixing_scores = cp_gt.VertexMPTransformation(neigh_coef=lambda_v, edge_index=edge_index, n_vertices=self.num_nodes).pipe_transform(baseline_scores)
                cp = GraphCP([], coverage_guarantee=self.coverage_val)

                cp.calibrate_from_scores(mixing_scores[tune_idx], self.label_mask[tune_idx])
                pred_set = cp.predict_from_scores(mixing_scores[tune_idx])
                overall_mixing_results.append({
                    "lambda": lambda_v,
                    "average_set_size": cp.average_set_size(pred_set)
                })
                # print("lambda_v", lambda_v)

            overall_mixing_results = pd.DataFrame(overall_mixing_results)
            baseline_res = overall_mixing_results.loc[(overall_mixing_results["lambda"] == 0)]["average_set_size"].values[0]
            overall_mixing_results["enhancement"] = overall_mixing_results["average_set_size"] - baseline_res
            iteration_results.append(overall_mixing_results[["enhancement"]])
            
        iteration_results = pd.concat(iteration_results, axis=1)
        iteration_mean = iteration_results.mean(axis=1)
        best_param_sets = overall_mixing_results.loc[iteration_mean.idxmin()]
        best_params = best_param_sets["lambda"]
        return overall_mixing_results, best_params
    
    def find_all_snaps_params(self, baseline_scores, tune_idx):
        edge_vals = np.arange(0, 1, 0.05).round(3)
        
        base_cp = GraphCP([], coverage_guarantee=self.coverage_val)
        base_cp.calibrate_from_scores(baseline_scores[tune_idx], self.label_mask[tune_idx])
        baseline_pred_set = base_cp.predict_from_scores(baseline_scores[tune_idx])
        
        overall_mixing_results = [{"edge_val": 0, "feature_val": 0, "average_set_size": base_cp.average_set_size(baseline_pred_set)}]
        
        for edge_val in edge_vals:
            for feature_val in np.arange(0, 1-edge_val, 0.05):
                snaps_scores = cp_gt.TEMPSNAPSTransformation(edge_index=self.edge_index, n_vertices=self.num_nodes, adj_knn=self.adj_knn, edge_val=edge_val, feature_val=feature_val).pipe_transform(baseline_scores)
                cp = GraphCP([], coverage_guarantee=self.coverage_val)
                
                cp.calibrate_from_scores(snaps_scores[tune_idx], self.label_mask[tune_idx])
                pred_set = cp.predict_from_scores(snaps_scores[tune_idx])
                overall_mixing_results.append({
                    "edge_val": edge_val,
                    "feature_val": feature_val,
                    "average_set_size": cp.average_set_size(pred_set)
                })
                # print("edge_val: {}, feature_val: {}".format(edge_val, feature_val))
        overall_mixing_results = pd.DataFrame(overall_mixing_results)
        baseline_res = overall_mixing_results.loc[(overall_mixing_results["edge_val"] == 0) & (overall_mixing_results["feature_val"] == 0)]["average_set_size"].values[0]
        overall_mixing_results["enhancement"] = overall_mixing_results["average_set_size"] - baseline_res
        
        
        best_param_sets = overall_mixing_results.loc[overall_mixing_results["enhancement"].idxmin()]
        # print(best_param_sets)
        best_params = (best_param_sets["edge_val"], best_param_sets["feature_val"])
        return overall_mixing_results, best_params
    
    def fair_shuffle_test_multiple_metrics(self, scores, y_true_mask, 
                coverage_guarantee=0.92, calib_fraction=0.5, n_iters=100,
                tune_scores=None, tune_mask=None):
        result_df = []
        for iter_idx in range(n_iters):
            iteration_series = pd.Series({"attempt": iter_idx})
            calib_scores_sub, eval_scores, calib_ymask_sub, eval_ymask = GraphDataManager.train_test_split(
                scores, y_true_mask, training_fraction=calib_fraction, return_idx=False)
            calib_scores = calib_scores_sub if tune_scores is None else torch.concat([calib_scores_sub, tune_scores])
            calib_ymask = calib_ymask_sub if tune_mask is None else torch.concat([calib_ymask_sub, tune_mask])
            
            scores_quantile = GraphCP(transformation_sequence=[], coverage_guarantee=coverage_guarantee).calibrate_from_scores(calib_scores, calib_ymask)
            # TODO: 按照论文中这里难道不应该是≥吗
            # pred_set = eval_scores > scores_quantile
            # pred_set = eval_scores >= scores_quantile
            pred_set = eval_scores <= scores_quantile
            for metric_name, metric_func in self.metrics_dict.items():
                result_val = metric_func(pred_set, eval_ymask)
                iteration_series[metric_name] = result_val
            result_df.append(iteration_series)
        result_df = pd.DataFrame(result_df)
        return result_df
    
    # def plot_motivation_figure(self, embeddings, tune_idx, test_idx):
    #     aps_cp = GraphCP(transformation_sequence=[cp_t.APSTransformation(softmax=True)])
    #     aps_scores = aps_cp.get_scores_from_logits(embeddings)
        
    #     snaps_cp = cp_gt.SNAPSTransformation(edge_index=self.edge_index, n_vertices=self.num_nodes, dataset_name=self.dataset_name)
    #     snaps_scores = snaps_cp.pipe_transform(aps_scores)
        
    #     calib_idx_sub, eval_idx, _, _ = GraphDataManager.train_test_split(
    #         test_idx, self.label_mask[test_idx], training_fraction=self.calib_fraction, return_idx=False)
    #     calib_idx = torch.concat([calib_idx_sub, tune_idx])
        
    #     calib_aps_scores = aps_scores[calib_idx]
    #     calib_snaps_scores = snaps_scores[calib_idx]
    #     calib_ymask = self.label_mask[calib_idx]
        
    #     eval_aps_scores = aps_scores[eval_idx]
    #     eval_snaps_scores = snaps_scores[eval_idx]
    #     eval_ymask = self.label_mask[eval_idx]
        
    #     aps_score_quantile = GraphCP(transformation_sequence=[], coverage_guarantee=self.coverage_val).calibrate_from_scores(calib_aps_scores, calib_ymask)
    #     snaps_score_quantile = GraphCP(transformation_sequence=[], coverage_guarantee=self.coverage_val).calibrate_from_scores(calib_snaps_scores, calib_ymask)
        
    #     aps_pred_set = eval_aps_scores <= aps_score_quantile
    #     snaps_pred_set = eval_snaps_scores <= snaps_score_quantile
        
    #     print(aps_score_quantile, snaps_score_quantile)
    #     for i in range(aps_pred_set.shape[0]):
    #         if aps_pred_set[i, :].sum() > 2 and torch.equal(snaps_pred_set[i, :], eval_ymask[i, :]):
    #             print(eval_idx[i], aps_scores[eval_idx[i]], self.dataset.y[eval_idx[i]])
                
    #             neighbor_idx = self.edge_index[1, self.edge_index[0, :] == eval_idx[i]]
    #             neighbor_label_scores = aps_scores[neighbor_idx]
    #             neighbor_label_mean = torch.mean(neighbor_label_scores, dim=0)
    #             print(neighbor_idx.shape, neighbor_label_mean)
                
    #             same_label_idx = torch.where(self.dataset.y == self.dataset.y[eval_idx[i]])[0]
    #             same_label_scores = aps_scores[same_label_idx]
    #             same_label_mean = torch.mean(same_label_scores, dim=0)
    #             print(same_label_mean)
                
    #             print(aps_pred_set[i, :])
    #             print(snaps_pred_set[i, :])
    #             print(eval_ymask[i, :])
    #             print(eval_aps_scores[i, :])
    #             print(eval_snaps_scores[i, :])
    #             print("=======================================================")
    #     exit(0)
        
    #     calib_scores_sub, eval_scores, calib_ymask_sub, eval_ymask = GraphDataManager.train_test_split(
    #             scores, y_true_mask, training_fraction=self.calib_fraction, return_idx=False)
        
        