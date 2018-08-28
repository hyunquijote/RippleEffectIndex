import sys
sys.path.insert(0, r'C:\\MyProject\\StockIndexPrediction')
import os
import tensorflow as tf
import numpy      as np
import pandas     as pd
from   pandas     import DataFrame
from   CodeDef    import *
from   DB_handler import *


# 딥러닝 수행 핸들러
class TensorFlow_handler:

    LearingOption = "STD"
    # LearingOption = "RNN_LSTM"

    # 초기화
    # inMain   : 메인폼
    def __init__(self, inMain):

        # 객체변수 설정
        self.DBH      = None   # DB 핸들러
        self.MainForm = inMain  # 메인폼

        self.PredMnList = CodeDef.TF_LEARNING_MN_LIST  # 예측분 리스트
        self.PredMnCnt  = len(self.PredMnList)

        # 결과저장 변수
        self.LernRsltFile = [None] * self.PredMnCnt
        self.LernSaveDir  = [None] * self.PredMnCnt
        self.LernSaveFile = [None] * self.PredMnCnt
        self.TCkpt        = [None] * self.PredMnCnt
        self.TMerged      = [None] * self.PredMnCnt
        self.TWriter      = [None] * self.PredMnCnt
        self.TSave        = [None] * self.PredMnCnt

        for idx in range(self.PredMnCnt):
            self.LernRsltFile[idx] = CodeDef.TF_LEARNING_RSLT_FILE + str(self.PredMnList[idx]) + ".xlsx"
            self.LernSaveDir[idx]  = CodeDef.TF_LEARNING_SAVE_DIR  + str(self.PredMnList[idx])
            self.LernSaveFile[idx] = self.LernSaveDir[idx] + CodeDef.TF_LEARNING_SAVE_FILE

        # 세션리스트
        self.Sess = [None] * self.PredMnCnt

        # DB 핸들러 초기화
        self.DBH = DB_handler()

        # 기본신경망 구성
        if (self.LearingOption == "STD"):
            print("기본신경망 구성")
            self.SetStdMdl()
        # RNN + LSTM
        elif (self.LearingOption == "RNN_LSTM"):
            print("RNN + LSTM 신경망 구성")
            self.SetRnnLstmMdl()

        return None

    # 예측
    # inData    : 입력데이터
    # inPredIdx : 예측분 리스트 인덱스
    def DoPrediction(self, inData, inPredIdx):

        PredV = 0.0

        # 기본신경망 구성
        if (self.LearingOption == "STD"):
            #print("기본신경망 예측수행(inPredIdx):",inPredIdx)
            RsltV = self.Sess[inPredIdx].run([self.TModel], feed_dict={self.T_X: inData, self.KeepDrop:1.0})
            RsltV = self.SetYScaling(RsltV)
            #print("DoPrediction 결과:", type(RsltV[0]), RsltV)
            PredV = RsltV[0][0]

        # RNN + LSTM
        # ★ 기본신경망에서 답이 안나올경우 해당방법으로 연구할것
        elif (self.LearingOption == "RNN_LSTM"):
            print("RNN + LSTM 신경망 예측수행")

        return PredV

    # 신경망 학습
    # inStrDtMn : 학습시작일자시각
    # inEndDtMn : 학습종료일자시각
    # inIdx     : 예측리스트 인덱스
    def DoLearning(self, inStrDtMn, inEndDtMn, inIdx):

        try:
            # 학습데이터 준비
            LernDataSets = self.DBH.queryLernData(inStrDtMn, inEndDtMn)

            # 모델링에 따른 입력 List 형태변경 및 수행
            if (self.LearingOption == "STD"):

                print("기본신경망 학습실행")
                # 입력데이터 재구성
                InputDataSets = self.GetInputDataSets( LernDataSets, inIdx)
                # 학습 입력값 엑셀저장(임시)
                #self.SaveLearnDataToExcel(InputDataSets, self.LernRsltFile[inIdx] + "_learn.xlsx")

                # List로 변경
                InputDataSets = InputDataSets.values  # numpy.ndarray
                self.L_data = InputDataSets[:, 1:].tolist()  # 학습입력
                self.R_data = InputDataSets[:, :1].tolist()  # 정답결과

                # 학습수행
                self.LernStdMdl(inIdx)

                # 학습 테스트 실행
                TestDataSets = self.DBH.queryLernData((CodeDef.TF_TEST_STR_DT + CodeDef.TF_TEST_STR_MN),
                                                      (CodeDef.TF_TEST_END_DT + CodeDef.TF_TEST_END_MN))

                # 입력데이터 재구성
                TestDataSets     = self.GetInputDataSets( TestDataSets, inIdx)
                # 테스트 입력값 엑셀저장(임시)
                #self.SaveLearnDataToExcel(TestDataSets, self.LernRsltFile[inIdx]+"_test.xlsx")
                TestDataSets     = TestDataSets.values
                InTestDataSets   = TestDataSets[:, 1:].tolist()
                RealRsltDataSets = TestDataSets[:, :1].tolist()

                PredRsltDataSets = self.Sess[inIdx].run([self.TModel], feed_dict={self.T_X: InTestDataSets, self.KeepDrop:1.0})
                PredRsltDataSets = self.SetYScaling(PredRsltDataSets)

                #print("예측값 엑셀 생성:", self.LernRsltFile[inIdx])
                self.SaveRsltToExcel(RealRsltDataSets, PredRsltDataSets, self.LernRsltFile[inIdx])

            elif (self.LearingOption == "RNN_LSTM"):
                print("RNN + LSTM 학습실행")
                LernDataSets = LernDataSets.values  # numpy.ndarray
                # RNN 형식의 맞춘 list 형으로 변경(배치사이즈, 단계, 입력)
                self.L_data = LernDataSets[:, 1:]  # numpy.ndarray
                self.L_data = self.L_data.reshape((len(self.R_data), 1, CodeDef.TF_LEARNING_INPUT_CNT))
                self.L_data = self.L_data.tolist()
                # 결과값 분리
                self.R_data = LernDataSets[:, :1].tolist()  # list

                # 학습수행
                self.LernRnnLstm()

        except Exception as e:
            print("학습수행 에러:", e)

        return None

    # 입력데이터셋 구성
    # 1분이 아닌 예측의 경우 데이터셋구성을 조정해야한다.
    # inDataSets : 구성대상 데이터셋(DataFrame)
    # inIdx      : 예측리스트 인덱스
    # 리턴: 조정된 DataFrame
    def GetInputDataSets(self, inDataSets, inIdx):

        RtrnDataSets = inDataSets

        # N분후 예상 학습(이미 들어온 데이터셋이 1분후 결과가 있는 데이터셋)
        if (inIdx > 0):
            PredMn    = self.PredMnList[inIdx]
            # 일별 학습종료 시각 계산
            TimeDelta = timedelta(minutes=1)
            TmpTime   = datetime(year=2000, month=1, day=1, hour=15, minute=20)
            EndTime   = int((TmpTime - (TimeDelta * PredMn)).strftime("%H%M"))

            # 1분뒤 현재가 삭제
            inDataSets.drop("y", axis=1, inplace=True)
            # PredMn 분뒤의 현재가 분리 후 인덱싱 재정의
            RtrnDataSets = RtrnDataSets[["x1"]].iloc[PredMn:, :].reset_index(drop=True)
            # 결과값으로 컬럼명 변경
            RtrnDataSets.rename(columns={"x1": "y"}, inplace=True)
            # 학습데이터 재결합
            RtrnDataSets = pd.merge(RtrnDataSets, inDataSets, how="inner", left_index=True, right_index=True)
            # 밀린만큼 일별 시간대 데이터를 삭제한다.
            RtrnDataSets.drop(RtrnDataSets[(RtrnDataSets.x2 > EndTime)].index, inplace=True)

        return RtrnDataSets

    # 결과값 스케일링 (ETF 로 맞춤)
    # inYlist : 결과값
    def SetYScaling(self, inYlist):
        Ylist = inYlist[0]
        RowCnt = len(Ylist)
        for idx in range(RowCnt):
            V = Ylist[idx][0]
            strV = str(int(round(V)))
            Prc = int(strV[-1:])

            if (Prc >= 7):
                Ylist[idx] = float(round(int(round(V)), -1))
            elif (Prc <= 3):
                Ylist[idx] = float(int(round(V)) - Prc)
            else:
                Ylist[idx] = float(strV[:-1] + "5")

        return Ylist

    # 학습결과값 엑셀저장
    # inY      : 정답 Y (type: list)
    # inPY     : 예측 Y (type: list)
    # inFile   : 파일
    def SaveRsltToExcel(self, inY, inPY, inFile):

        try:
            Y  = DataFrame(inY, columns=["Real_Y"])
            PY = DataFrame(inPY, columns=["Prediction_Y"])

            AY = pd.merge(Y, PY, how="outer", left_index=True, right_index=True)

            # print("합체:", AY.head())

            writer = pd.ExcelWriter(inFile, engine='xlsxwriter')

            AY.to_excel(writer, sheet_name='Sheet1')

            writer.close()

        except Exception as e:
            print("엑셀출력 에러:", e)

        return None

    # 학습 입력값 엑셀저장
    # inDataset: 입력 데이터셋 1행은 결과, 뒤는 입력 (type: DataFrame)
    # inFile   : 파일
    def SaveLearnDataToExcel(self, inDataset, inFile):

        try:

            writer = pd.ExcelWriter(inFile, engine='xlsxwriter')

            inDataset.to_excel(writer, sheet_name='Sheet1')

            writer.close()

        except Exception as e:
            print("학습입력값 엑셀출력 에러:", e)

        return None



    ### ------------------------ 모델링 --------------------------###

    # 가중치 초기화변수 생성 (균등분포)
    # inInCnt   : 입력행수
    # inOutCnt  : 출력력행수
    # inUniform : 균등분포사용여부
    def GetXavierInit(self, inInCnt, inOutCnt, inUniform=True):
        if inUniform:
            InitRange = tf.sqrt(6.0 / (inInCnt + inOutCnt))
            return tf.random_uniform_initializer(-InitRange, InitRange)
        else:
            Stddev = tf.sqrt(3.0 / (inInCnt + inOutCnt))
            return tf.truncated_normal_initializer(stddev=Stddev)

    ## --- 기본 모델링 수행 --##

    # 기본 모델링 구현
    def SetStdMdl(self):

        # 변수설정 (행, 열)
        self.T_X = tf.placeholder(tf.float32, [None, CodeDef.TF_LEARNING_INPUT_CNT  ], name="T_X")  # 학습입력값
        self.P_X = tf.placeholder(tf.float32, [None, CodeDef.TF_PREDICTION_INPUT_CNT], name="P_X")  # 예측요구입력값
        self.R_Y = tf.placeholder(tf.float32, [None, CodeDef.TF_LEARNING_OUTPUT_CNT ], name="R_Y")  # 결과값

        self.TStep    = tf.Variable(0, trainable=False, name="TStep")  # 학습횟수
        self.KeepDrop = tf.placeholder(tf.float32, name="KeepDrop")    # 과적합을 피하기위한 드롭아웃
        # tf.layers.batch_normalization 사용을 고려해볼것 코딩순서는 batch_norm > relu > drop 순

        # 가중치변수 초기화
        self.W1 = tf.get_variable("W1", shape=[CodeDef.TF_LEARNING_INPUT_CNT, CodeDef.TF_LAYER_1_NEURON_CNT ],initializer=self.GetXavierInit(CodeDef.TF_LEARNING_INPUT_CNT,CodeDef.TF_LAYER_1_NEURON_CNT ))
        self.W2 = tf.get_variable("W2", shape=[CodeDef.TF_LAYER_1_NEURON_CNT, CodeDef.TF_LAYER_2_NEURON_CNT ],initializer=self.GetXavierInit(CodeDef.TF_LAYER_1_NEURON_CNT,CodeDef.TF_LAYER_2_NEURON_CNT ))
        self.W3 = tf.get_variable("W3", shape=[CodeDef.TF_LAYER_2_NEURON_CNT, CodeDef.TF_LAYER_3_NEURON_CNT ],initializer=self.GetXavierInit(CodeDef.TF_LAYER_2_NEURON_CNT,CodeDef.TF_LAYER_3_NEURON_CNT ))
        self.W4 = tf.get_variable("W4", shape=[CodeDef.TF_LAYER_3_NEURON_CNT, CodeDef.TF_LEARNING_OUTPUT_CNT],initializer=self.GetXavierInit(CodeDef.TF_LAYER_3_NEURON_CNT,CodeDef.TF_LEARNING_OUTPUT_CNT))

        # 편향 초기화
        self.B1 = tf.Variable(tf.zeros([CodeDef.TF_LAYER_1_NEURON_CNT ], name="B1"))
        self.B2 = tf.Variable(tf.zeros([CodeDef.TF_LAYER_2_NEURON_CNT ], name="B2"))
        self.B3 = tf.Variable(tf.zeros([CodeDef.TF_LAYER_3_NEURON_CNT ], name="B3"))
        self.B4 = tf.Variable(tf.zeros([CodeDef.TF_LEARNING_OUTPUT_CNT], name="B4"))

        # 가중치 및 편향으로 Layer 구성, 활상화함수(relu 사용)
        # Layer 1
        with tf.name_scope("Layer_1"):
            self.Layer1 = tf.add(tf.matmul(self.T_X, self.W1), self.B1)
            self.Layer1 = tf.nn.relu(self.Layer1)
            self.Layer1 = tf.nn.dropout(self.Layer1, self.KeepDrop)
        # Layer 2
        with tf.name_scope("Layer_2"):
            self.Layer2 = tf.add(tf.matmul(self.Layer1, self.W2), self.B2)
            self.Layer2 = tf.nn.relu(self.Layer2)
            self.Layer2 = tf.nn.dropout(self.Layer2, self.KeepDrop)
        # Layer 3
        with tf.name_scope("Layer_3"):
            self.Layer3 = tf.add(tf.matmul(self.Layer2, self.W3), self.B3)
            self.Layer3 = tf.nn.relu(self.Layer3)
            self.Layer3 = tf.nn.dropout(self.Layer3, self.KeepDrop)
        # 출력층
        with tf.name_scope("Layer_OutPut"):
            self.TModel = tf.add(tf.matmul(self.Layer3, self.W4), self.B4)
            self.TModel = tf.nn.relu(self.TModel)

        # 최적화 모델
        with tf.name_scope("Optimizer"):
            # 손실값
            self.TCost = tf.reduce_mean(tf.square(self.TModel - self.R_Y))
            # 최적화 설정 : 다른 최적화 함수들도 테스트 해볼것 RMSPropOptimizer 등등
            self.TOptimizer = tf.train.AdamOptimizer(learning_rate=CodeDef.TF_LEARNING_RATE)
            self.TOption    = self.TOptimizer.minimize(self.TCost, global_step=self.TStep)
            # 손실값 추적 수집
            tf.summary.scalar("Cost", self.TCost)
            tf.summary.histogram("Weight_1", self.W1)
            tf.summary.histogram("Weight_2", self.W2)
            tf.summary.histogram("Weight_3", self.W3)
            tf.summary.histogram("Weight_4", self.W4)
            tf.summary.histogram("Bias_1", self.B1)
            tf.summary.histogram("Bias_2", self.B2)
            tf.summary.histogram("Bias_3", self.B3)
            tf.summary.histogram("Bias_4", self.B4)

        # 예측 분별로 세션 다중화
        for idx in range(self.PredMnCnt):
            # 세션
            self.Sess[idx] = tf.Session()

            # 학습저장 변수
            self.TSave[idx] = tf.train.Saver(tf.global_variables())
            self.TCkpt[idx] = tf.train.get_checkpoint_state(self.LernSaveDir[idx])

            # 체크포인트 복구
            # 기존 학습 모델이 다른경우는 파일삭제 먼저 수행하고 프로그램을 실행해야함.
            if (self.TCkpt[idx] and tf.train.checkpoint_exists(self.TCkpt[idx].model_checkpoint_path)):
                print("기존모델복구 :"+str(self.PredMnList[idx])+"분")
                self.TSave[idx].restore(self.Sess[idx], self.TCkpt[idx].model_checkpoint_path)
            else:
                print("모델최초시작 :"+str(self.PredMnList[idx])+"분")
                self.Sess[idx].run(tf.global_variables_initializer())

            # 텐서보드 사용변수
            self.TMerged[idx] = tf.summary.merge_all()
            self.TWriter[idx] = tf.summary.FileWriter(CodeDef.TF_LEARNING_LOG_DIR, self.Sess[idx].graph)

        return None

    # 기본 모델링 학습실행
    # inIdx : 예측리스트 인덱스
    def LernStdMdl(self, inIdx):

        try:
            # 기존학습데이터 초기화여부 처리
            if(self.MainForm.GetInitSave()):
                print("기존 저장 초기화수행:"+str(self.PredMnList[inIdx])+"분 예측용")
                self.Sess[inIdx].run(tf.global_variables_initializer())

            for epoch in range(CodeDef.TF_LEARNING_EPOCH):
                # 학습시작
                for step in range(CodeDef.TF_LEARNING_CNT):

                    # self.Sess.run(self.TOption, feed_dict={self.T_X: self.L_data, self.R_Y: self.R_data, self.KeepDrop: CodeDef.TF_LEARNING_DROPOUT_RATE})
                    # self.Sess.run(self.TOption, feed_dict={self.T_X: self.L_data, self.R_Y: self.R_data})

                    LernCost, _, _ = self.Sess[inIdx].run([self.TCost, self.TModel, self.TOption],
                                                          feed_dict={self.T_X: self.L_data, self.R_Y: self.R_data, self.KeepDrop: CodeDef.TF_LEARNING_DROPOUT_RATE})

                    if (step >= CodeDef.TF_LEARNING_CNT - 5):
                        print("Step,Cost: ", self.Sess[inIdx].run(self.TStep), LernCost)

                    # 텐서보드 출력을 위한 값 저장
                    Summary = self.Sess[inIdx].run(self.TMerged[inIdx],
                                                   feed_dict={self.T_X: self.L_data, self.R_Y: self.R_data, self.KeepDrop: CodeDef.TF_LEARNING_DROPOUT_RATE})
                    self.TWriter[inIdx].add_summary(Summary, global_step=self.Sess[inIdx].run(self.TStep))

            # 결과저장
            print("결과저장:", self.LernSaveFile[inIdx])
            self.TSave[inIdx].save(self.Sess[inIdx], self.LernSaveFile[inIdx], global_step=self.TStep)

            # 결과 보기
            # 드로아웃 사용 print("예측값: ", self.Sess.run(Prediction, feed_dict={self.T_X: self.L_data, self.KeepDrop: 1}))
            # 드로아웃 사용 print("실제값: ", self.Sess.run(Target    , feed_dict={self.R_Y: self.R_data, self.KeepDrop: 1}))

            # self.R_Y = self.Sess.run([self.TModel], feed_dict={self.T_X: self.L_data})
            # self.R_Y = self.SetYScaling(self.R_Y)

            # print("예측값 엑셀 셍성:",CodeDef.TF_LEARNING_RSLT_FILE )
            # self.SetToExcel(self.R_data, self.R_Y)

        except Exception as e:
            print(e)

        return None

    ## --- 기본 모델링 수행 끝 --- ##

    ## --- RNN + LSTM 모델링 수행 --- ##

    # RNN + LSTM 모델링 구현
    def SetRnnLstmMdl(self):
        # 변수설정 (행, 열)
        self.T_X = tf.placeholder(tf.float32, [None, 1, CodeDef.TF_LEARNING_INPUT_CNT])  # 학습입력값
        self.P_X = tf.placeholder(tf.float32, [None, 1, CodeDef.TF_LEARNING_INPUT_CNT])  # 예측요구입력값
        self.R_Y = tf.placeholder(tf.float32, [None, CodeDef.TF_LEARNING_OUTPUT_CNT])  # 결과값

        self.TStep = tf.Variable(0, trainable=False, name="TStep")  # 학습횟수
        # self.KeepDrop = tf.placeholder(tf.float32)  # 과적합을 피하기위한 드롭아웃
        # tf.layers.batch_normalization 사용을 고려해볼것 코딩순서는 batch_norm > relu > drop 순

        # 가중치 및 편향 설정
        self.W = tf.Variable(tf.random_normal([CodeDef.TF_LEARNING_RNN_CELS_CNT, CodeDef.TF_LEARNING_OUTPUT_CNT]))
        self.B = tf.Variable(tf.random_normal([CodeDef.TF_LEARNING_OUTPUT_CNT]))

        # RNN + LSTM 으로 구성
        with tf.name_scope("RNN_LSTM"):
            self.Cell = tf.nn.rnn_cell.BasicLSTMCell(CodeDef.TF_LEARNING_RNN_CELS_CNT)
            # 모델링 완성
            self.OutPuts, self.States = tf.nn.dynamic_rnn(self.Cell, self.T_X, dtype=tf.float32, time_major=True)
            # self.OutPuts = tf.transpose(self.OutPuts,[1,0,2])
            self.OutPuts = self.OutPuts[-1]
            self.TModel = tf.add(tf.matmul(self.OutPuts, self.W), self.B)
            # self.TModel = tf.nn.relu(self.TModel) 이건 필요가 없나?

        with tf.name_scope("Optimizer"):
            # 손실값
            self.TCost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.R_Y, logits=self.TModel))
            # 최적화 설정 : 다른 최적화 함수들도 테스트 해볼것 RMSPropOptimizer 등등
            self.TOptimizer = tf.train.AdamOptimizer(learning_rate=CodeDef.TF_LEARNING_RATE)
            self.TOption = self.TOptimizer.minimize(self.TCost, global_step=self.TStep)
            # 손실값 추적 수집
            tf.summary.scalar("Cost", self.TCost)
            tf.summary.histogram("Weight", self.W)
            tf.summary.histogram("Bias", self.B)

        # 학습저장 변수
        self.Sess = tf.Session()  # 세션
        self.TSave = tf.train.Saver(tf.global_variables())
        self.TCkpt = tf.train.get_checkpoint_state(self.LernSaveDir)
        # 체크포인트 복구
        # 기존 학습 모델이 다른경우는 파일삭제 먼저 수행하고 프로그램을 실행해야함.
        if (self.TCkpt and tf.train.checkpoint_exists(self.TCkpt.model_checkpoint_path)):
            print("기존모델복구")
            self.TSave.restore(self.Sess, self.TCkpt.model_checkpoint_path)
        else:
            print("모델최초시작")
            self.Sess.run(tf.global_variables_initializer())

        # 텐서보드 사용변수
        self.TMerged = tf.summary.merge_all()
        self.TWriter = tf.summary.FileWriter(CodeDef.TF_LEARNING_LOG_DIR, self.Sess.graph)

        return

    # RNN + LSTM 학습실행
    def LernRnnLstm(self):
        try:
            # 학습시작
            for step in range(CodeDef.TF_LEARNING_EPOCH):
                # self.Sess.run(self.TOption, feed_dict={self.T_X: self.L_data, self.R_Y: self.R_data, self.KeepDrop: CodeDef.TF_LEARNING_DROPOUT_RATE})
                self.Sess.run(self.TOption, feed_dict={self.T_X: self.L_data, self.R_Y: self.R_data})

                print("Step: ", self.Sess.run(self.TStep))
                print("Cost: ", self.Sess.run(self.TCost, feed_dict={self.T_X: self.L_data, self.R_Y: self.R_data}))

                # 텐서보드 출력을 위한 값 저장
                Summary = self.Sess.run(self.TMerged, feed_dict={self.T_X: self.L_data, self.R_Y: self.R_data})
                self.TWriter.add_summary(Summary, global_step=self.Sess.run(self.TStep))

            # 결과저장
            self.TSave.save(self.Sess, self.LernSaveFile, global_step=self.TStep)

            # 결과 보기
            Prediction = tf.argmax(self.TModel, 1)
            Target = tf.argmax(self.R_data, 1)
            # print("예측값: ", self.Sess.run(Prediction, feed_dict={self.T_X: self.L_data, self.KeepDrop: 1}))
            # print("실제값: ", self.Sess.run(Target    , feed_dict={self.R_Y: self.R_data, self.KeepDrop: 1}))
            print("예측값: ", self.Sess.run(Prediction, feed_dict={self.T_X: self.L_data}))
            print("실제값: ", self.Sess.run(Target, feed_dict={self.R_Y: self.R_data}))

            IsCorrct = tf.equal(Prediction, Target)
            Accuracy = tf.reduce_mean(tf.cast(IsCorrct, tf.float32))
            # print("정확도: %.2f" % self.Sess.run(Accuracy*100, feed_dict={self.T_X: self.L_data, self.R_Y: self.R_data, self.KeepDrop: 1}))
            print("정확도: %.2f" % self.Sess.run(Accuracy * 100,
                                              feed_dict={self.T_X: self.L_data, self.R_Y: self.R_data}))
        except Exception as e:
            print(e)

        return None

    ## --- RNN + LSTM 모델링 수행 끝 --- ##