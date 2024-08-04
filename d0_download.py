import baostock as bs
import pandas as pd


def download_code_hist(
    save_path="./datasets",
    code="sh.600000",
    start_date="2018-09-01",
    end_date="2024-06-30",
    freq="d",
    adjustflag="2"
):
    lg = bs.login()
    #### 获取沪深A股历史K线数据 ####
    # 详细指标参数，参见“历史行情指标参数”章节；“分钟线”参数与“日线”参数不同。“分钟线”不包含指数。
    # 分钟线指标：date,time,code,open,high,low,close,volume,amount,adjustflag
    # 周月线指标：date,code,open,high,low,close,volume,amount,adjustflag,turn,pctChg
    fields = "date,time,code,open,high,low,close,volume,adjustflag",
    if freq == "d":
        fields = "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST"
    rs = bs.query_history_k_data_plus(
        code,
        # "date,time,code,open,high,low,close,volume,adjustflag",
        # "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST",
        fields=fields,
        start_date=start_date,
        end_date=end_date,
        frequency=freq,
        adjustflag=adjustflag, # hfq
    )

    #### 打印结果集 ####
    data_list = []
    while (rs.error_code == "0") & rs.next():
        # 获取一条记录，将记录合并在一起
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)
    
    # print(result)

    #### 结果集输出到csv文件 ####
    filename = save_path + "/" + code + ".csv"
    result.to_csv(filename, index=True)
    # print(result)
    
    print(result.head())
    
    bs.logout()


download_code_hist()