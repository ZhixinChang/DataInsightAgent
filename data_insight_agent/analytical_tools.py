import os
import warnings
from typing import List, Tuple, Literal, Annotated

import matplotlib.pyplot as plt
import pandas as pd
from optbinning import BinningProcess

warnings.filterwarnings("ignore")

# 配置中文字体（Windows用"SimHei"，Mac用"Songti SC"，Linux用"WenQuanYi Micro Hei"）
plt.rcParams["font.family"] = ["Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


def get_from_user(reason: Annotated[str, '字符串格式，需要从用户获取额外信息的原因']):
    """向用户获取必要的分析信息"""
    info = input("Agent请求获取额外信息：{}，请补充信息（输入“停止”结束任务）：".format(reason))
    if info == '停止':
        info = '<USER_TERMINATE>'

    return info


def get_from_user_chat_ui(reason: Annotated[str, '字符串格式，需要从用户获取额外信息的原因']):
    """向用户获取必要的分析信息"""
    info = reason + '<USER_TERMINATE>'

    return info


def get_summary_info(user_demand: Annotated[str, '字符串格式，用户的需求信息'],
                     result: Annotated[str, '字符串格式，交付用户需求的结论信息'],
                     result_path: Annotated[str, '字符串格式，分析结果的保存路径，只需要具体到目录，不需要文件名'], ):
    """获取结论总结所需要的信息"""

    return '用户需求：' + user_demand + '\n' + '结论：' + result + '\n' + '结论最终需要保存至路径：' + result_path


def save_summary(result_path: Annotated[str, '字符串格式，分析结果的保存路径，只需要具体到目录，不需要文件名'],
                 markdown_text: Annotated[str, '字符串格式，Markdown格式的结论总结内容']):
    """以Markdown格式保存结论总结"""
    # 保存到指定目录
    file_path = os.path.join(result_path, "summary.md")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(markdown_text)

    return f'结论总结已输出至：FILE_PATHS<{file_path}>FILE_PATHS'


def data_describe(data_path: Annotated[str, '字符串格式，数据集的路径，例如"/.../数据集名称.xlsx"']) -> str:
    """获取数据集相关信息"""
    df = pd.read_excel(data_path)
    return '数据中包含的信息如下：' + '\n'.join(
        '字段名：' + df.dtypes.index + '，数据类型：' + df.dtypes.values.astype('str'))


def correlation_analysis(data_path: Annotated[str, '字符串格式，数据集的路径，具体到文件名'],
                         result_path: Annotated[str, '字符串格式，分析结果的保存路径，只需要具体到目录，不需要文件名'],
                         x_monotonic_trend_list: Annotated[
                             List[Tuple[str, Literal[
                                 '无', '递增', '递减', '凹曲线', '凸曲线', '峰值', '谷值']]], '列表格式，包含指标x的字段名和该字段与y的单调趋势的元组'],
                         y: Annotated[str, '字符串格式，目标指标y的字段名']) -> str:
    """批量分析指标x与y之间的相关性趋势"""
    x_list = []
    binning_fit_params = {}
    for x, monotonic_trend in x_monotonic_trend_list:
        if monotonic_trend == '无':
            monotonic_trend = 'auto'
        elif monotonic_trend == '递增':
            monotonic_trend = 'ascending'
        elif monotonic_trend == '递减':
            monotonic_trend = 'descending'
        elif monotonic_trend == '凹曲线':
            monotonic_trend = 'concave'
        elif monotonic_trend == '凸曲线':
            monotonic_trend = 'convex'
        elif monotonic_trend == '峰值':
            monotonic_trend = 'peak'
        elif monotonic_trend == '谷值':
            monotonic_trend = 'valley'
        else:
            return '相关性分析工具的参数输入存在错误，无法进行分析。'

        x_list.append(x)
        binning_fit_params[x] = {}
        binning_fit_params[x]['monotonic_trend'] = monotonic_trend

    df = pd.read_excel(data_path)

    binning_process = BinningProcess(variable_names=x_list, binning_fit_params=binning_fit_params)
    binning_process.fit(df[x_list], df[y])
    result = []
    image_paths = []
    for x in x_list:
        binning_process.get_binned_variable(x).binning_table.plot(
            show_bin_labels=True, savefig=os.path.join(result_path, f"相关趋势图_{x}.png"), figsize=(12, 10)
        )
        image_paths.append(os.path.join(result_path, f"相关趋势图_{x}.png"))
        binning_table_df = binning_process.get_binned_variable(x).binning_table.build()
        corr_df = binning_table_df.loc[
            ~binning_table_df['Bin'].isin(['Special', 'Missing']) & (binning_table_df.index != 'Totals'), [
                'Mean']].reset_index()
        corr_coef = corr_df['index'].corr(corr_df['Mean'], method='pearson')
        result.append(x + '和' + y + '的相关系数为' + str(corr_coef))

    return f'相关性分析完成，各指标的相关系数：\n' + '\n'.join(
        result) + '\n' + f'相关性趋势图已输出至IMAGE_PATHS<{','.join(image_paths)}>IMAGE_PATHS'


def attribution_analysis(raw_df, totals, continuous_col, dims, binning_fit_params):
    df = raw_df.copy()
    binning_process = BinningProcess(variable_names=continuous_col, binning_fit_params=binning_fit_params)
    binning_process.fit(df[continuous_col], df[totals])
    df[continuous_col] = binning_process.transform(df[continuous_col], metric='bins')
    df['size'] = 1
    size = "size"

    sf = explain_levels(
        df=df,
        dims=dims,
        total_name=totals,
        size_name=size,
        max_depth=1,
        max_segments=50,
        cluster_values=False,
        solver="lasso",
    )
    return sf, binning_process
