import os
import re

import matplotlib.pyplot as plt

from rampds.new_utils.utils import FileUtils


# TODO: add these in a class / UPPERCASE if use them as constants
expe_meta_data_file = "experiment_metadata.json"
scores_file = "scores.csv"
scores_plot_file = "openfe_scores_plot.png"


def score_is_better(score, comparison_score, objective_direction):
    """ Check if the current score is better than the best score based on the objective direction.

    Args:
        score (float): The current score to evaluate.
        comparison_score (float): The score to compare against.
        objective_direction (str): The direction of the objective, either 'minimize' or 'maximize'.

    Returns:
        _type_: _description_
    """
    if objective_direction == "minimize":
        return score < comparison_score
    else:
        return score > comparison_score


class OpenFEUtils:
    """
    Utility class for OpenFE related operations.
    """

    @staticmethod
    def get_experiment_type(min_cand_feat, data_blocks, feature_boost, selection_method):
        """
        Constructs a string representing the experiment type based on parameters.

        Parameters:
            min_cand_feat (int): Minimum candidate features.
            data_blocks (int): Number of data blocks.
            feature_boost (str): Feature boost method.
            selection_method (str): Feature selection method.

        Returns:
            str: Formatted experiment type string.
        """
        return f"{min_cand_feat//1000}k_mcf_{data_blocks}_db_fb_{feature_boost}_{selection_method}"
        # return f"{min_cand_feat//1000}k_min_cand_feat_{data_blocks}_data_blocks_feature_boost_{feature_boost}_{selection_method}"
    
    @staticmethod
    def plot_and_save_scores(n_feat, scores, original_score, score_name, data_name, objective_direction, results_dir):
        """
        Plots the scores and saves the visualization to a file.

        Parameters:
            n_feat (list): Number of selected features.
            scores (list): Mean scores corresponding to the selected features.
            original_score (float): Original score for comparison.
            score_name (str): Name of the score metric.
            data_name (str): Name of the dataset.
            objective_direction (str): Objective direction (e.g., maximize or minimize).
            results_dir (str): Directory to save the plot.
        """
        plt.figure(figsize=(12, 8))
        plt.plot(n_feat, scores, marker='o', label='OpenFE Scores', color='blue', linewidth=2)
        plt.axhline(y=original_score, color='red', linestyle='--', label='Original Score', linewidth=2)

        # Add labels and title with improved formatting
        plt.xlabel('Number of Selected Features', fontsize=14)
        plt.ylabel(f'Mean Score ({score_name}) ', fontsize=14)
        plt.title(f'Performance of OpenFE on `{data_name}` Challenge (Score: {score_name}, Objective: {objective_direction})', fontsize=15)
        # Add legend and grid for better readability
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)

        # Save the improved plot
        plot_path = os.path.join(results_dir, scores_plot_file)
        plt.savefig(plot_path, dpi=300)
        print(f"OpenFE Plot saved to {plot_path}")

    @staticmethod
    def load_results(experiment_result_dir, data_name):
        scores_file_path = os.path.join(experiment_result_dir, data_name, scores_file)
        experiment_metadata_file = os.path.join(experiment_result_dir, data_name, expe_meta_data_file)
        results_df = FileUtils.load_csv(scores_file_path)
        experiment_metadata = FileUtils.load_json(experiment_metadata_file)

        return results_df, experiment_metadata 
    
    @staticmethod	
    def plot_comparison_scores(
        self,
        n_feat_lst, 
        scores_list, 
        score_labels_list, 
        original_score, 
        score_name, 
        objective_direction
    ):
        """
        Plot comparison of different score configurations.
        
        Parameters:
            n_feat_lst (list): Lists of feature counts for each configuration
            scores_list (list): Lists of scores for each configuration
            score_labels_list (list): Labels for each configuration
            original_score (float): Original score for comparison
            score_name (str): Name of the score metric
            objective_direction (str): "maximize" or "minimize"
        """
        if self.best_n_selected_features is None or self.best_score is None:
            print("Warning: best_n_selected_features or best_score not set. Call get_best_n_selected_features() first.")
            
        plt.figure(figsize=(12, 6))
        for scores, label, n_feat in zip(scores_list, score_labels_list, n_feat_lst):
            plt.plot(n_feat, scores, marker='o', label=label, linewidth=2)
        plt.axhline(y=original_score, color='red', linestyle='--', label='Original Score', linewidth=2)
        
        if self.best_n_selected_features is not None and self.best_score is not None:
            plt.scatter(self.best_n_selected_features, self.best_score, color='gold', s=300, marker='*', 
                        label='Best Result', edgecolors='black', linewidths=1.5, zorder=10)
            
            # Add an annotation next to the star
            plt.annotate(f'Best: {self.best_score:.2f}', 
                xy=(self.best_n_selected_features, self.best_score), 
                xytext=(self.best_n_selected_features+2, self.best_score),
                fontsize=12,
                weight='bold',
                arrowprops=dict(arrowstyle='->')
            )
        
        plt.xlabel('Number of Selected Features', fontsize=12)
        plt.ylabel(f'Mean Score ({score_name}) ', fontsize=12)
        plt.title(f'Performance of OpenFE on `{self.data_name}` Challenge (Score: {score_name}, Objective: {objective_direction})', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()

    @staticmethod
    def get_best_n_selected_features(results_df, objective_direction, original_score, n_digits_round=5):
        """
        Get the best number of selected features based on the mean score.
        """
        # TODO: fix these hardcoded names ... 
        if n_digits_round is None:
            results_df['rounded_mean_score'] = results_df['mean_score']
        else:
            results_df['rounded_mean_score'] = results_df['mean_score'].round(n_digits_round)
        
        if objective_direction == "maximize":
            best_row = results_df[results_df['rounded_mean_score'] == results_df['rounded_mean_score'].max()].nsmallest(1, 'n_selected_features')
        else:
            best_row = results_df[results_df['rounded_mean_score'] == results_df['rounded_mean_score'].min()].nsmallest(1, 'n_selected_features')
        
        best_score = best_row['mean_score'].iloc[0]
        best_score_rounded = best_row['rounded_mean_score'].iloc[0]

        improvement = score_is_better(best_score, original_score, objective_direction)

        if improvement:
            best_n_selected_features = best_row['n_selected_features'].iloc[0]
        else:
            best_n_selected_features = 0

        print(f"\nImprovement over original score: {improvement}")
        print(f"Best Score: {best_score}")
        print(f"Best Score Rounded: {best_score_rounded}")
        print(f"Best n_selected_features: {best_n_selected_features}")

        return best_n_selected_features, best_score
    
    @staticmethod
    def parse_OpenFE_feature_name(name: str) -> str:
        """
        Parses and modifies an OpenFE feature name string into a standardized format.

        Examples:
            - "Func(var1, var2)" -> "func__var1__var2"
            - "(var1*var2)" -> "var1__mul__var2"
            - "GroupByThenMean(var1, var2)" -> "gp_mean__var1__var2"
            - "CombineThenSum(var1, var2)" -> "cmb_sum__var1__var2"
            - "Func(var1)" -> "func__var1"
        """
        name = name.strip()
        
        # Operator mappings
        op_symbol_map = {'*': 'mul', '/': 'div', '+': 'add', '-': 'sub'}
        
        # Case 1: Infix operation like (var1*var2)
        # Matches content inside parentheses with a numerical operator
        infix_match = re.match(r'\(([^,()]+?)\s*([+\-*/])\s*([^,()]+?)\)', name)
        if infix_match:
            var1, op, var2 = infix_match.groups()
            op_name = op_symbol_map.get(op, op)
            return f"{var1.strip()}__{op_name}__{var2.strip()}"

        # Case 2: Function call like Func(var1, var2) or Func(var1)
        # Matches a function name and content within parentheses
        func_match = re.match(r'([A-Za-z_][A-Za-z0-9_]*)\((.*)\)', name)
        if func_match:
            func_name, content = func_match.groups()
            
            # Simplify GroupByThen... names
            if 'GroupByThen' in func_name:
                func_name = func_name.replace('GroupByThen', 'gp_').lower()
            if 'CombineThen' in func_name:
                func_name = func_name.replace('CombineThen', 'cmb_').lower()
            
            # Split variables by comma
            variables = [v.strip() for v in content.split(',')]
            
            return f"{func_name}__{'__'.join(variables)}"
            
        # Fallback for any other pattern
        return re.sub(r'[^a-zA-Z0-9_]+', '_', name).strip('_')

    @staticmethod
    def rename_OpenFE_columns(columns: list[str]) -> list[str]:
        """
        Applies the simplified parsing to a list of column names.
        """
        return [OpenFEUtils.parse_OpenFE_feature_name(c) for c in columns]