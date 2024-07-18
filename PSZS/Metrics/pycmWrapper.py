import math
import os
from typing import Any, Dict, List, Optional, Sequence
from numbers import Number
import numpy as np

from PSZS.Utils.io import create_or_append_excel

import pycm
from pycm.pycm_param import PYCM_VERSION, DOCUMENT_ADR, PARAMS_LINK, BENCHMARK_LIST, BENCHMARK_COLOR, DEFAULT_BACKGROUND_COLOR, RECOMMEND_BACKGROUND_COLOR, PARAMS_DESCRIPTION, CAPITALIZE_FILTER
from pycm.pycm_output import html_init, html_end, html_table_color
from pycm.pycm_util import class_filter, rounder, sort_char_num

OVERALL_STAT_LIST = ['95% CI', 'ACC Macro', 'Conditional Entropy', 'Cross Entropy', 'F1 Macro', 'FNR Macro', 
                     'FNR Micro', 'FPR Macro', 'FPR Micro', 'Hamming Loss', 'Joint Entropy', 'NIR', 'NPV Macro', 
                     'NPV Micro', 'Overall ACC', 'PPV Macro', 'Standard Error', 'TNR Macro', 'TNR Micro', 
                     'TPR Macro', 'Zero-one Loss']
OVERALL_SUMMARY_STAT_LIST = ['ACC Macro', 'Cross Entropy', 'F1 Macro', 'FNR Macro', 
                            'FPR Macro', 'NPV Macro', 'Overall ACC', 'PPV Macro', 'TNR Macro', 
                            'TPR Macro', 'Zero-one Loss']

CLASS_STAT_LIST = ['ACC', 'AGF', 'AUC', 'AUCI', 'Diff', 'DiffS', 'DP', 'DPI', 'ERR', 'F0.5', 'F1', 'F2', 'FDR', 'FN', 'FNR', 'FOR', 'FP', 'FPR', 'NLR', 'NLRI', 'NumP', 'NumA', 'N', 'P', 'PLR', 'PLRI', 'POP', 'TN', 'TNR', 'TP', 'TPR']
CLASS_SUMMARY_STAT_LIST = ['ACC', 'AUC', 'AUCI', 'Diff', 'DiffS', 'DP', 'DPI', 'ERR', 'F1', 'NumP', 'NumA', 'N', 'P', 'POP', 'TN', 'TNR', 'TP', 'TPR']

RECOMMEND_LIST : List[str] = ['ACC', 'ERR', 'Diff', 'DiffS', 'F1', 'F1 Macro', 'FP', 'FPR', 'NumP', 'NumA', 'TPR Macro', 'TPR', 'Overall ACC', 'Zero-one Loss']

SUMMARY_LIST = ['ACC', 'F1', 'FPR', 'TPR', 'Diff', 'DiffS', 'ERR', 'NumP', 'NumA','N','P']

# Update/Add new Benchmark values
BENCHMARK_COLOR.update({'DiffS': {
                            "Poor": "DarkRed",
                            "Bad": "OrangeRed",
                            "Substantial": "Orange",
                            "Moderate": "Gold",
                            "Good": "YellowGreen",
                            "Very Good": "LawnGreen",
                            "Excellent": "Green",
                            "Perfect": "RoyalBlue",
                            },
                        })
BENCHMARK_LIST = list(BENCHMARK_COLOR.keys())

def cross_entropy_calc(P: Dict[Any, Number], 
                        TOP: Dict[Any, Number], 
                        population: Number) -> float:
            r = 0
            for i in TOP.keys():
                ref_likelihood = P[i] / population
                res_likelihood = TOP[i] / population
                if res_likelihood != 0 and ref_likelihood != 0:
                    r += ref_likelihood * math.log2(res_likelihood)
            return -r

def joint_entropy_calc(table: Dict[str|int, Dict[str|int, Number]],
                        population: Number,
                        class_names: Sequence[str|int]) -> float:
            p_prime = [table[i][j] / population for i in class_names for j in class_names]
            return -sum([p_prime[i] * math.log2(p_prime[i]) for i in range(len(p_prime)) if p_prime[i] != 0])

def cond_entropy_calc(table: Dict[str|int, Dict[str|int, Number]],
                      P: Dict[Any, Number],
                      population: Number,
                      class_names: Sequence[str|int]) -> float:
            r = 0
            for i in class_names:
                t = 0
                for j in class_names:
                    p_prime = 0
                    if P[i] != 0:
                        p_prime = table[i][j] / P[i]
                    if p_prime != 0:
                        t += p_prime * math.log2(p_prime)
                r += t * (P[i] / population)
            return -r

def macro_calc(field: dict | Sequence) -> float:
            if isinstance(field, dict):
                field = list(field.values())
            return sum(field) / len(field)
        
def micro_calc(field1: Dict[Any, Number] | Sequence[Number], 
                field2: Dict[Any, Number] | Sequence[Number]) -> float:
    if isinstance(field1, dict):
        field1 = list(field1.values())
    if isinstance(field2, dict):
        field2 = list(field2.values())
    return sum(field1) / sum(field1) + sum(field2)

def relative_diff(field1, field2) -> float:
    return abs(field1 - field2) / (field1 + field2)

def get_diff_significance(field: float) -> str:
    if field == 0:
        return "Perfect"
    elif field < 0.05:
        return "Excellent"
    elif field < 0.1:
        return "Very Good"
    elif field < 0.2:
        return "Good"
    elif field < 0.4:
        return "Moderate"
    elif field < 0.5:
        return "Substantial"
    elif field < 0.7:
        return "Bad"
    else:
        return "Poor"
               
class pycmConfMat(pycm.ConfusionMatrix):
    def __init__(self,
                 matrix: np.ndarray | list,
                 digit: int=5,
                 transpose: bool=False,
                 classes: Optional[Sequence[str]]=None,):
        super().__init__(matrix=matrix,
                         digit=digit,
                         transpose=transpose,
                         classes=classes,)
        self.recommended_list = RECOMMEND_LIST
        num_preds = {self.classes[i]:sum(matrix[:,i]) for i in range(len(matrix))}
        num_actual = {self.classes[i]:sum(matrix[i,:]) for i in range(len(matrix))}
        rel_diff = {self.classes[i]:relative_diff(num_preds[self.classes[i]], num_actual[self.classes[i]])
                    for i in range(len(matrix))}
        diff_significance = {self.classes[i]:get_diff_significance(rel_diff[self.classes[i]]) for i in range(len(matrix))}
        self.class_name_map = {i:self.classes[i] for i in range(len(self.classes))}
        self.class_stat.update({'NumP': num_preds, 'NumA': num_actual, 
                                'Diff': rel_diff, 'DiffS': diff_significance})
        PARAMS_LINK.update({'NumP': PARAMS_LINK['TOP'], 'NumA': PARAMS_LINK['P'], 'Diff': '#', 'DiffS': '#'})
        PARAMS_DESCRIPTION.update({'NumP': 'Number of predictions', 
                                   'NumA': 'Number of class samples',
                                   'Diff': 'Relative difference between number of predictions and number of actual samples',
                                   'DiffS': 'Signifance of Diff value'})
    
    def recompute_overall(self, class_names: Sequence[str]) -> None:
        relevant_population = sum([sum(list(self.table[k].values())) for k in class_names])
        # Filtered TP, TN, FP, FN for class_names
        filtered_TP : Dict[str, Number] = {k:v for k,v in self.TP.items() if k in class_names}
        filtered_TN : Dict[str, Number] = {k:v for k,v in self.TN.items() if k in class_names}
        filtered_FP : Dict[str, Number] = {k:v for k,v in self.FP.items() if k in class_names}
        filtered_FN : Dict[str, Number] = {k:v for k,v in self.FN.items() if k in class_names}
        # Test condition positive i.e. number of positive samples (support/number of occurrences of each class)
        filtered_P = {k:filtered_TP[k] + filtered_FN[k] for k in class_names}
        # Test outcome positive i.e. number of predictions on this class irrespective of true or false
        filtered_TOP = {k:filtered_TP[k] + filtered_FP[k] for k in class_names}
        # True positive ratio
        filtered_TPR : Dict[str, Number] = {k:self.TPR[k] for k in class_names}
        # True negative ratio
        filtered_TNR : Dict[str, Number] = {k:self.TNR[k] for k in class_names}
        # Negative predictive value  (True negative / True negative + False negative)
        filtered_NPV : Dict[str, Number] = {k:self.NPV[k] for k in class_names}
        # Positive predictive value (True positive / True positive + False positive)
        filtered_PPV : Dict[str, Number] = {k:self.PPV[k] for k in class_names}
        filtered_F1 : Dict[str, Number] = {k:self.F1[k] for k in class_names}
        # Random accuracy. Needs to be calcaulted w.r.t. the relevant population
        filtered_RACC = {k:(filtered_TOP[k] * filtered_P[k]) / (relevant_population**2) for k in class_names}
        # Random accuracy unbiased. Needs to be calcaulted w.r.t. the relevant population
        filtered_RACCU = {k:((filtered_TOP[k] + filtered_P[k]) / (2* relevant_population))**2 for k in class_names}
        # Overall Accuracy needs to be w.r.t. original population
        self.overall_stat["Overall ACC"] = sum(filtered_TP.values()) / relevant_population
        # Overall random Accuracy (RACC is calculated w.r.t. the relevant population)
        self.overall_stat["Overall RACC"] = sum(filtered_RACC.values())
        # Overall random Accuracy unbiase (RACCU is calculated w.r.t. the relevant population)
        self.overall_stat["Overall RACCU"] = sum(filtered_RACCU.values())
        self.overall_stat["Standard Error"] = math.sqrt((self.overall_stat["Overall ACC"] * (1 - self.overall_stat["Overall ACC"])) / relevant_population)
        critical_value = 1.96
        self.overall_stat["95% CI"] = (self.overall_stat["Overall ACC"] - self.overall_stat["Standard Error"]*critical_value, 
                                      self.overall_stat["Overall ACC"] + self.overall_stat["Standard Error"]*critical_value)
        self.overall_stat["Cross Entropy"] = cross_entropy_calc(filtered_P, filtered_TOP, relevant_population)
        self.overall_stat["Joint Entropy"] = joint_entropy_calc(self.table, relevant_population, class_names)
        self.overall_stat["Conditional Entropy"] = cond_entropy_calc(self.table, filtered_P, relevant_population, class_names)
        self.overall_stat["Hamming Loss"] = (1 / relevant_population) * (relevant_population - sum(filtered_TP.values()))
        self.overall_stat["Zero-one Loss"] = (relevant_population - sum(filtered_TP.values()))
        self.overall_stat["TPR Macro"] = macro_calc(filtered_TPR)
        self.overall_stat["TPR Micro"] = self.overall_stat["Overall ACC"]
        self.overall_stat["TNR Macro"] = macro_calc(filtered_TNR)
        self.overall_stat["TNR Micro"] = micro_calc(filtered_TN, filtered_FP)
        self.overall_stat["F1 Macro"] = macro_calc(filtered_F1)
        self.overall_stat["FNR Macro"] = 1 - self.overall_stat["TPR Macro"]
        self.overall_stat["FNR Micro"] = 1 - self.overall_stat["TPR Micro"]
        self.overall_stat["FPR Macro"] = 1 - self.overall_stat["TNR Macro"]
        self.overall_stat["FPR Micro"] = 1 - self.overall_stat["TNR Micro"]
        self.overall_stat["NPV Macro"] = macro_calc(filtered_NPV)
        self.overall_stat["NPV Micro"] = micro_calc(filtered_TN, filtered_FN)
        self.overall_stat["PPV Macro"] = macro_calc(filtered_PPV)
    
    def write_class_summary(self, 
                      file: str = 'summary.xlsx',
                      class_names: Optional[Sequence[str]]=None,):
        if class_names is not None:
            class_names = list(class_names)
            class_stat_classes = class_filter(self.classes, class_names)
            data = {i:{j:self.class_stat[i][j] for j in class_stat_classes} 
                    for i in SUMMARY_LIST}
            field_names = {k:v for k,v in self.class_name_map.items() if v in class_stat_classes}
        else:
            # Reduce dictionary construction cost as no filtering of classes is needed
            data = {i:self.class_stat[i] for i in SUMMARY_LIST}
            field_names = self.class_name_map
        create_or_append_excel(data=data, 
                               output_file=file,
                               field_names=field_names)
        
    def save_html(
            self,
            name,
            address: bool=True,
            overall_param: Optional[Sequence[str]]=OVERALL_STAT_LIST,
            class_param: Optional[Sequence[str]]=CLASS_STAT_LIST,
            class_name=None,
            color=(0, 0, 0),
            normalize=False,
            summary=False,
            shortener=True,
            cell_size:int=2,
            cell_width:Optional[int]=None,
            cell_height:Optional[int]=None,
            horizontal_overall: bool = True,
            sort_recommended: bool = True,
            overall_stats: Optional[Dict[str, Number]]=None):
        """
        Save ConfusionMatrix in HTML file.

        :param name: filename
        :type name: str
        :param address: flag for address return
        :type address: bool
        :param overall_param: overall parameters list for save, Example: ["Kappa", "Scott PI"]
        :type overall_param: list
        :param class_param: class parameters list for save, Example: ["TPR", "TNR", "AUC"]
        :type class_param: list
        :param class_name: class name (subset of classes names), Example: [1, 2, 3]
        :type class_name: list
        :param color: matrix color in RGB as (R, G, B)
        :type color: tuple
        :param normalize: save normalize matrix flag
        :type normalize: bool
        :param summary: summary mode flag
        :type summary: bool
        :param shortener: class name shortener flag
        :type shortener: bool
        :return: saving address as dict {"Status":bool, "Message":str}
        """
        try:
            class_list = class_param
            overall_list = overall_param
            if summary:
                class_param = CLASS_SUMMARY_STAT_LIST
                overall_list = OVERALL_SUMMARY_STAT_LIST
            if overall_stats is not None:
                overall_list += list(overall_stats.keys())
                for k in overall_stats.keys():
                    # RECOMMEND_LIST can be modified to include new stats
                    # as it only is used with 'is in' checks
                    if k not in RECOMMEND_LIST:
                        RECOMMEND_LIST.append(k)
                    # None Link for new stats
                    PARAMS_LINK[k] = '#'
                self.overall_stat.update(overall_stats)
            message = None
            table = self.table
            if normalize:
                table = self.normalized_table
            # Class filter requires class_names to be a list
            if class_name is not None:
                class_name = list(class_name)
            # Move up class filtering to use it for html_table as well
            class_stat_classes = class_filter(self.classes, class_name)
            
            html_file = open(name + ".html", "w", encoding="utf-8")
            html_file.write(html_init())
            # html_file.write(html_dataset_type(self.binary, self.imbalance))
            html_file.write(
                self.html_table(
                    # classes=self.classes,
                    classes=class_stat_classes,
                    table=table,
                    rgb_color=color,
                    normalize=normalize,
                    shortener=shortener,
                    cell_size=cell_size,
                    cell_width=cell_width,
                    cell_height=cell_height))
            html_file.write(
                self.html_overall_stat(
                    overall_stat=self.overall_stat,
                    digit=self.digit,
                    overall_param=overall_list,
                    recommended_list=self.recommended_list,
                    horizontal=horizontal_overall,
                    sort_recommended=sort_recommended,)
                )
            html_file.write(
                self.html_class_stat(
                    classes=class_stat_classes,
                    class_stat=self.class_stat,
                    digit=self.digit,
                    class_param=class_list,
                    recommended_list=self.recommended_list,
                    sort_recommended=sort_recommended,)
                    )
            html_file.write(html_end(PYCM_VERSION))
            html_file.close()
            if address:
                message = os.path.join(
                    os.getcwd(), name + ".html")  # pragma: no cover
            return {"Status": True, "Message": message}
        except Exception as e:
            return {"Status": False, "Message": str(e)}
        
    def html_table(
        self,
        classes,
        table,
        rgb_color,
        normalize=False,
        shortener=True,
        cell_size:int=2,
        cell_width:Optional[int]=None,
        cell_height:Optional[int]=None,
        shortened_length: int = 10):
        """
        Return the confusion matrix of the HTML report file.

        :param classes: confusion matrix classes
        :type classes: list
        :param table: input confusion matrix
        :type table: dict
        :param rgb_color: input color
        :type rgb_color: tuple
        :param normalize: save normalized matrix flag
        :type normalize: bool
        :param shortener: class name shortener flag
        :type shortener: bool
        :return: html_table as str
        """
        if cell_width is None:
            cell_width = cell_size
        if cell_height is None:
            cell_height = cell_size
        result = ""
        result += "<h2>Confusion Matrix "
        if normalize:
            result += "(Normalized)"
        result += ": </h2>\n"
        result += '<table>\n'
        result += '<tr style="text-align:center;">' + "\n"
        result += '<td>Actual</td>\n'
        result += '<td>Predict\n'
        table_size_w = str((len(classes) + 1) * cell_width) + "em"
        table_size_h= str((len(classes) + 1) * cell_height) + "em"
        result += '<table style="table-layout: fixed;border:1px solid black;border-collapse: collapse;height:{0};width:{1};">\n'\
            .format(table_size_h, table_size_w)
        result += '<tr style="text-align:center;">\n'
        result += '<td style="border:1px solid black;padding:10px;height:{0}em;width:{1}em;"></td>\n'.format(cell_height, cell_width)
        part_2 = ""
        for i in classes:
            class_name = str(i)
            # Add 2 to account for '...' and indexing is one shorter
            if len(class_name) > shortened_length+2 and shortener:
                class_name = class_name[:shortened_length] + "..."
            result += '<td style="border:1px solid black;padding:10px;height:{0}em;width:{1}em;">'.format(
                cell_height, cell_width) + class_name + '</td>\n'
            part_2 += '<tr style="text-align:center;">\n'
            part_2 += '<td style="border:1px solid black;padding:10px;height:{0}em;width:{1}em;">'.format(
                cell_height, cell_width) + class_name + '</td>\n'
            for j in classes:
                item = table[i][j]
                color = "black"
                back_color = html_table_color(table[i], item, rgb_color)
                if min(back_color) < 128:
                    color = "white"
                part_2 += '<td style="background-color:rgb({0},{1},{2});color:{3};padding:10px;height:{4}em;width:{5}em;">'.format(
                    str(back_color[0]), str(back_color[1]), str(back_color[2]), color, cell_height, cell_width) + str(item) + '</td>\n'
            part_2 += "</tr>\n"
        result += '</tr>\n'
        part_2 += "</table>\n</td>\n</tr>\n</table>\n"
        result += part_2
        return result
    
    def html_overall_stat(
        self,
        overall_stat,
        digit=5,
        overall_param=None,
        recommended_list=(),
        horizontal: bool = True,
        sort_recommended: bool = True,):
        if horizontal:
            return self._html_overall_stat_horizontal(
                overall_stat=overall_stat,
                digit=digit,
                overall_param=overall_param,
                recommended_list=recommended_list,
                sort_recommended=sort_recommended,)
        else:
            return self._html_overall_stat_vertical(
                overall_stat=overall_stat,
                digit=digit,
                overall_param=overall_param,
                recommended_list=recommended_list,
                sort_recommended=sort_recommended,)
    
    def _html_overall_stat_vertical(
        self,
        overall_stat,
        digit=5,
        overall_param=None,
        recommended_list=(),
        sort_recommended: bool = True,):
        """
        Return the overall stats of HTML report file.

        :param overall_stat: overall stats
        :type overall_stat: dict
        :param digit: scale (number of fraction digits)(default value: 5)
        :type digit: int
        :param overall_param: overall parameters list for print, Example: ["Kappa", "Scott PI"]
        :type overall_param: list
        :param recommended_list: recommended statistics list
        :type recommended_list: list or tuple
        :param alt_link: alternative link for document flag
        :type alt_link: bool
        :return: html_overall_stat as str
        """
        document_link = DOCUMENT_ADR
        result = ""
        result += "<h2>Overall Statistics : </h2>\n"
        result += '<table style="border:1px solid black;border-collapse: collapse;">\n'
        overall_stat_keys = sort_char_num(overall_stat.keys())
        
        if isinstance(overall_param, list):
            if set(overall_param) <= set(overall_stat_keys):
                overall_stat_keys = sort_char_num(overall_param)
        if len(overall_stat_keys) < 1:
            return ""
        
        if sort_recommended:
            # Reorder to have recommended items in front
            rec, oth = [], []
            for x in overall_stat_keys:
                (rec, oth)[x not in recommended_list].append(x)
            overall_stat_keys = rec + oth
        
        for i in overall_stat_keys:
            background_color = DEFAULT_BACKGROUND_COLOR
            if i in recommended_list:
                background_color = RECOMMEND_BACKGROUND_COLOR
            result += '<tr style="text-align:center;">\n'
            result += '<td style="border:1px solid black;padding:4px;text-align:left;background-color:{};"><a href="'.format(
                background_color) + document_link + PARAMS_LINK[i] + '" style="text-decoration:None;">' + str(i) + '</a></td>\n'
            if i in BENCHMARK_LIST:
                background_color = BENCHMARK_COLOR[i][overall_stat[i]]
                result += '<td style="border:1px solid black;padding:4px;background-color:{};">'.format(
                    background_color)
            else:
                result += '<td style="border:1px solid black;padding:4px;">'
            result += rounder(overall_stat[i], digit) + '</td>\n'
            result += "</tr>\n"
        result += "</table>\n"
        return result
    
    def _html_overall_stat_horizontal(
        self,
        overall_stat,
        digit=5,
        overall_param=None,
        recommended_list=(),
        sort_recommended: bool = True,):
        """
        Return the overall stats of HTML report file.

        :param overall_stat: overall stats
        :type overall_stat: dict
        :param digit: scale (number of fraction digits)(default value: 5)
        :type digit: int
        :param overall_param: overall parameters list for print, Example: ["Kappa", "Scott PI"]
        :type overall_param: list
        :param recommended_list: recommended statistics list
        :type recommended_list: list or tuple
        :param alt_link: alternative link for document flag
        :type alt_link: bool
        :return: html_overall_stat as str
        """
        document_link = DOCUMENT_ADR
        result = ""
        result += "<h2>Overall Statistics : </h2>\n"
        result += '<table style="border:1px solid black;border-collapse: collapse;">\n'
        
        overall_stat_keys = sort_char_num(overall_stat.keys())
        if isinstance(overall_param, list):
            if set(overall_param) <= set(overall_stat_keys):
                overall_stat_keys = sort_char_num(overall_param)
        if len(overall_stat_keys) < 1:
            return ""
        
        if sort_recommended:
            # Reorder to have recommended items in front
            rec, oth = [], []
            for x in overall_stat_keys:
                (rec, oth)[x not in recommended_list].append(x)
            overall_stat_keys = rec + oth
        
        # First row: Names
        result += '<tr style="text-align:center;">\n'
        for i in overall_stat_keys:
            background_color = DEFAULT_BACKGROUND_COLOR
            if i in recommended_list:
                background_color = RECOMMEND_BACKGROUND_COLOR
            result += '<td style="border:1px solid black;padding:4px;text-align:center;background-color:{};">'.format(background_color)
            result += '<a href="' + document_link + PARAMS_LINK[i] + '" style="text-decoration:None;">' + str(i) + '</a></td>\n'
        result += "</tr>\n"
        
        # Second row: Values
        result += '<tr style="text-align:center;">\n'
        for i in overall_stat_keys:
            if i in BENCHMARK_LIST:
                background_color = BENCHMARK_COLOR[i][overall_stat[i]]
                result += '<td style="border:1px solid black;padding:4px;background-color:{};">'.format(background_color)
            else:
                result += '<td style="border:1px solid black;padding:4px;">'
            result += rounder(overall_stat[i], digit) + '</td>\n'
        result += "</tr>\n"
        
        result += "</table>\n"
        return result
    
    def html_class_stat(
        self,
        classes,
        class_stat,
        digit=5,
        class_param=None,
        recommended_list=(),
        sort_recommended: bool = True,):
        """
        Return the class-based stats of HTML report file.

        :param classes: confusion matrix classes
        :type classes: list
        :param class_stat: class stat
        :type class_stat:dict
        :param digit: scale (number of fraction digits)(default value: 5)
        :type digit: int
        :param class_param: class parameters list for print, Example: ["TPR", "TNR", "AUC"]
        :type class_param: list
        :param recommended_list: recommended statistics list
        :type recommended_list: list or tuple
        :return: html_class_stat as str
        """
        document_link = DOCUMENT_ADR
        result = ""
        result += "<h2>Class Statistics : </h2>\n"
        result += '<table style="border:1px solid black;border-collapse: collapse;">\n'
        result += '<tr style="text-align:center;">\n<td>Class</td>\n'
        for i in classes:
            result += '<td style="border:1px solid black;padding:4px;border-collapse: collapse;">' + \
                str(i) + '</td>\n'
        result += '<td>Description</td>\n'
        result += '</tr>\n'
        class_stat_keys = sorted(class_stat.keys())
        if isinstance(class_param, list):
            if set(class_param) <= set(class_stat_keys):
                class_stat_keys = class_param
        if len(classes) < 1 or len(class_stat_keys) < 1:
            return ""
        
        if sort_recommended:
            # Reorder to have recommended items in front
            rec, oth = [], []
            for x in class_stat_keys:
                (rec, oth)[x not in recommended_list].append(x)
            class_stat_keys = rec + oth
        
        for i in class_stat_keys:
            background_color = DEFAULT_BACKGROUND_COLOR
            if i in recommended_list:
                background_color = RECOMMEND_BACKGROUND_COLOR
            result += '<tr style="text-align:center;border:1px solid black;border-collapse: collapse;">\n'
            result += '<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:{};"><a href="'.format(
                background_color) + document_link + PARAMS_LINK[i] + '" style="text-decoration:None;">' + str(i) + '</a></td>\n'
            for j in classes:
                if i in BENCHMARK_LIST:
                    background_color = BENCHMARK_COLOR[i][class_stat[i][j]]
                    result += '<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:{};">'.format(
                        background_color)
                else:
                    result += '<td style="border:1px solid black;padding:4px;border-collapse: collapse;">'
                result += rounder(class_stat[i][j], digit) + '</td>\n'
            params_text = PARAMS_DESCRIPTION[i]
            if i not in CAPITALIZE_FILTER:
                params_text = params_text.capitalize()
            result += '<td style="border:1px solid black;padding:4px;border-collapse: collapse;text-align:left;">' + \
                    params_text + '</td>\n'
            result += "</tr>\n"
        result += "</table>\n"
        return result