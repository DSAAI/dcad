import pandas as pd
import tqdm

from data.traindata_multi_graph import MultiGraphTraceTrainDataSet

def toVae():
    data = {"traceIdHigh":[], #traceID
            "traceIdLow":[],  #traceID
            "parentSpanId":[],   #parentSpanId
            "spanId":[],        #spanId
            "startTime":[],     #
            "duration":[],
            "nanosecond":[],
            "DBhash":[],
            "status":[],
            "operationName":[],
            "serviceName":[],
            "nodeLatencyLabel":[],
            "graphLatencyLabel":[],
            "graphStructureLabel":[]
            }
    csv_file = "/home/cjl/csv/test.csv"
    df = pd.read_csv(csv_file)
    grouped = df.groupby('traceId')
    num = 0
    for traceId, group in grouped:
        traceIds = group.traceId.values




def getAnomalyData():
    data = {"traceId": [],
            "abnormal": [],
            "source": [],
            "target": [],

            "source_service_name": [],
            "target_service_name": [],
            "source_service_id": [],
            "target_service_id": [],

            "rawDuration": [],
            "isError": [],
            "workDuration": [],
            "statusCode": [],
            "duration": [],
            "proportionProcessingTime": [],
            "rawNetworklatency": [],
            "rawProcessingTime": [],
            "operationId": []
            }

    csv_file = "/home/cjl/csv/test.csv"
    df = pd.read_csv(csv_file)
    grouped = df.groupby('traceId')

    num = 0
    for traceId, group in grouped:


        traceIds = group.traceId.values
        abnormals = group.abnormal.values
        sources = group.source.values
        targets = group.target.values
        source_service_names = group.source_service_name.values
        target_service_names = group.target_service_name.values
        source_service_ids = group.source_service_id.values
        target_service_ids = group.target_service_id.values
        rawDurations = group.rawDuration.values
        isErrors = group.isError.values
        workDurations = group.workDuration.values
        statusCodes = group.statusCode.values
        durations = group.duration.values
        proportionProcessingTimes = group.proportionProcessingTime.values
        rawNetworklatencies = group.rawNetworklatency.values
        rawProcessingTimes = group.rawProcessingTime.values
        operationIds = group.operationId.values


        if(abnormals[0] == 1):# 43801  24860    24860:3285 = 7.5:1 = 75:10 = 15:2
            for i in range(0,len(traceIds)):
                data["traceId"].append(traceIds[i])
                data["abnormal"].append(abnormals[i])
                data["source"].append(sources[i])
                data["target"].append(targets[i])
                data["source_service_name"].append(source_service_names[i])
                data["target_service_name"].append(target_service_names[i])
                data["source_service_id"].append(source_service_ids[i])
                data["target_service_id"].append(target_service_ids[i])
                data["rawDuration"].append(rawDurations[i])
                data["isError"].append(isErrors[i])
                data["workDuration"].append(workDurations[i])
                data["statusCode"].append(statusCodes[i])
                data["duration"].append(durations[i])
                data["proportionProcessingTime"].append(proportionProcessingTimes[i])
                data["rawNetworklatency"].append(rawNetworklatencies[i])
                data["rawProcessingTime"].append(rawProcessingTimes[i])
                data["operationId"].append(operationIds[i])

        res_df = pd.DataFrame.from_dict(data, orient='columns', )
        res_df.to_csv("/home/cjl/csv/anomalydata.csv",index=True)

    print(num)


def getNormalData1():
    data = {"traceId": [],
            "abnormal": [],
            "source": [],
            "target": [],

            "source_service_name": [],
            "target_service_name": [],
            "source_service_id": [],
            "target_service_id": [],

            "rawDuration": [],
            "isError": [],
            "workDuration": [],
            "statusCode": [],
            "duration": [],
            "proportionProcessingTime": [],
            "rawNetworklatency": [],
            "rawProcessingTime": [],

            "operationId": []
            }

    csv_file = "/home/cjl/csv/test.csv"
    df = pd.read_csv(csv_file)
    grouped = df.groupby('traceId')

    num = 0
    for traceId, group in tqdm.tqdm(grouped):
        abnormals = group.abnormal.values
        if(abnormals[0] == 1): continue

        traceIds = group.traceId.values
        sources = group.source.values
        targets = group.target.values
        source_service_names = group.source_service_name.values
        target_service_names = group.target_service_name.values
        source_service_ids = group.source_service_id.values
        target_service_ids = group.target_service_id.values
        rawDurations = group.rawDuration.values
        isErrors = group.isError.values
        workDurations = group.workDuration.values
        statusCodes = group.statusCode.values
        durations = group.duration.values
        proportionProcessingTimes = group.proportionProcessingTime.values
        rawNetworklatencies = group.rawNetworklatency.values
        rawProcessingTimes = group.rawProcessingTime.values
        operationIds = group.operationId.values


        # 43801  24860    24860:3285 = 7.5:1 = 75:10 = 15:2
        for i in range(0,len(traceIds)):
            data["traceId"].append(traceIds[i])
            data["abnormal"].append(abnormals[i])
            data["source"].append(sources[i])
            data["target"].append(targets[i])
            data["source_service_name"].append(source_service_names[i])
            data["target_service_name"].append(target_service_names[i])
            data["source_service_id"].append(source_service_ids[i])
            data["target_service_id"].append(target_service_ids[i])
            data["rawDuration"].append(rawDurations[i])
            data["isError"].append(isErrors[i])
            data["workDuration"].append(workDurations[i])
            data["statusCode"].append(statusCodes[i])
            data["duration"].append(durations[i])
            data["proportionProcessingTime"].append(proportionProcessingTimes[i])
            data["rawNetworklatency"].append(rawNetworklatencies[i])
            data["rawProcessingTime"].append(rawProcessingTimes[i])
            data["operationId"].append(operationIds[i])

        res_df = pd.DataFrame.from_dict(data, orient='columns', )
        res_df.to_csv("/home/cjl/csv/normaldata.csv",index=True)

    print(num)



def getNormalData():
    csv_file = "/home/cjl/csv/test.csv"
    df = pd.read_csv(csv_file)
    df_normal = df[df["abnormal"] == 1]
    df_normal.to_csv("/home/cjl/csv/anomalydata.csv",index=True)


def getNormalDataNum():
    csv_file = "/home/cjl/csv/train.csv"
    df = pd.read_csv(csv_file)
    grouped = df.groupby('traceId')
    edges = 0
    num = 0
    for traceId, group in tqdm.tqdm(grouped):
        num += 1
    for i in tqdm.tqdm(df):
        edges += 1
    print(num)
def mergeNormalDataNum():
    csv_file1 = "/home/cjl/csv/train.csv"
    csv_file2 = "/home/cjl/csv/normaldata.csv"
    df1 = pd.read_csv(csv_file1)
    df2 = pd.read_csv(csv_file2)

    #df = df1[df1["traceId"].isin(df2["traceId"])]

    df = df1.append(df2)
    df.to_csv("/home/cjl/csv/allnormaldata.csv", index=True)

def randomsample():
    csv_file = "/home/cjl/csv/allnormaldata.csv"
    df = pd.read_csv(csv_file)
    res_df = df[df["traceId"] == "c682ded083b842b5a12ee878b139a9ee.44.16451004288830225"]
    print(res_df)

def buildtraindata():
    csv_file = "/home/cjl/csv/allnormaldata.csv"
    df = pd.read_csv(csv_file)
    grouped = df.groupby('traceId')
    abnormals_data = []
    num = 0
    traceIds = []
    for traceId, group in tqdm.tqdm(grouped):
        num += 1
        traceIds.append(traceId)
        if (num == 100000):
            break
    res_df = df[df["traceId"].isin(traceIds)]
    res_df.to_csv("/home/cjl/csv/randomnormaldata.csv", index=True)

def buildanomalydata():
    csv_file = "/home/cjl/csv/allnormaldata.csv"
    df = pd.read_csv(csv_file)
    grouped = df.groupby('traceId')
    abnormals_data = []
    num = 0
    traceIds = []
    for traceId, group in tqdm.tqdm(grouped):
        num += 1
        traceIds.append(traceId)
        if (num == 100000):
            break
    res_df = df[df["traceId"].isin(traceIds)]
    res_df.to_csv("/home/cjl/csv/randomnormaldata.csv", index=True)

def buildanomalydata2():
    csv_file1 = "/home/cjl/csv/randomnormaldata.csv"
    csv_file2 = "/home/cjl/csv/anomalydata.csv"
    df1 = pd.read_csv(csv_file1)
    df2 = pd.read_csv(csv_file2)
    grouped = df2.groupby('traceId')
    num = 0
    traceIds5000 = []
    traceIds3000 = []
    traceIds1000 = []
    traceIds500 = []
    traceIds100 = []

    for traceId, group in tqdm.tqdm(grouped):
        num += 1
        if(num < 5000):
            traceIds5000.append(traceId)
        elif(num == 5000):
            traceIds5000.append(traceId)
            df5000 = df2[df2["traceId"].isin(traceIds5000)]
            df5000.to_csv("/home/cjl/csv/anomalydata5.csv", index=True)
            df5000 = df5000.append(df1)
            df5000.to_csv("/home/cjl/csv/traindata5.csv", index=True)
        elif(num >5000 and num <8000):
            traceIds3000.append(traceId)
        elif(num == 8000):
            traceIds3000.append(traceId)
            df3000 = df2[df2["traceId"].isin(traceIds3000)]
            df3000.to_csv("/home/cjl/csv/anomalydata3.csv", index=True)
            df3000 = df3000.append(df1)
            df3000.to_csv("/home/cjl/csv/traindata3.csv", index=True)
        elif(num > 8000 and num < 9000):
            traceIds1000.append(traceId)
        elif(num == 9000):
            traceIds1000.append(traceId)
            df1000 = df2[df2["traceId"].isin(traceIds1000)]
            df1000.to_csv("/home/cjl/csv/anomalydata1.csv", index=True)
            df1000 = df1000.append(df1)
            df1000.to_csv("/home/cjl/csv/traindata1.csv", index=True)
        elif(num >9000 and num < 9500):
            traceIds500.append(traceId)
        elif(num == 9500):
            traceIds500.append(traceId)
            df500 = df2[df2["traceId"].isin(traceIds500)]
            df500.to_csv("/home/cjl/csv/anomalydata0.5.csv", index=True)
            df500 = df500.append(df1)
            df500.to_csv("/home/cjl/csv/traindata0.5.csv", index=True)
        elif(num >9500 and num <9600):
            traceIds100.append(traceId)
        elif(num == 9600):
            traceIds100.append(traceId)
            df100 = df2[df2["traceId"].isin(traceIds100)]
            df100.to_csv("/home/cjl/csv/anomalydata0.1.csv", index=True)
            df100 = df100.append(df1)
            df100.to_csv("/home/cjl/csv/traindata0.1.csv", index=True)
            break

def buildtestdata():
    csv_file1 = "/home/cjl/csv/allnormaldata.csv"
    csv_file2 = "/home/cjl/csv/anomalydata.csv"
    df1 = pd.read_csv(csv_file1)
    df2 = pd.read_csv(csv_file2)
    grouped1 = df1.groupby('traceId')
    grouped2 = df2.groupby('traceId')
    num1 = 0
    num2 = 0
    traceId1 = []
    traceId9500 = []
    traceId100 = []
    dfnormal = 0
    for traceId, group in tqdm.tqdm(grouped1):
        num1+=1
        if(num1 <= 100000):
            continue
        elif(num1 > 100000 and num1 <109500):
            traceId1.append(traceId)
        elif(num1 == 109500):
            traceId1.append(traceId)
            dfnormal = df1[df1["traceId"].isin(traceId1)]


    for traceId, group in tqdm.tqdm(grouped2):
        num2 +=1
        if(num2 <= 9600):
            continue
        elif(num2 > 9600  and num2 < 19100):
            traceId9500.append(traceId)
        elif(num2 == 19100):
            traceId9500.append(traceId)
            dfabnormal9500 = df2[df2["traceId"].isin(traceId9500)]
            tdata = dfnormal.append(dfabnormal9500)
            tdata.to_csv("/home/cjl/csv/testdata50.csv", index=True)

            '''
        elif(num2 > 10100 and num2 < 10200):
            traceId100.append(traceId)
        elif(num2 == 10200):
            traceId100.append(traceId)
            dfabnormal100 = df2[df2["traceId"].isin(traceId100)]
            tdata = dfnormal.append(dfabnormal100)
            tdata.to_csv("/home/cjl/csv/testdata1.csv", index=True)
            '''


if __name__ == '__main__':



    #mergeNormalDataNum()
    #getNormalDataNum()
    #buildtraindata()
    #buildanomalydata2()
    #buildtestdata()
    #getNormalDataNum()
    getNormalDataNum()
