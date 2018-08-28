from datetime import timedelta, date, datetime

class CodeDef():
    # ----------------------- 시스템 정보 ---------------------------
    # 접속서버 주소
    SERVER_REAL_GLOBAL  = "real.tradarglobal.api.com"  # 티레이더글로벌 운영 서버
    SERVER_SIMUL_TRADAR = "simul.tradar.api.com"        # 티레이더 모의투자 서버
    SERVER_DEV_TRADAR   = "dev.tradar.api.com"          # 티레이더글로벌 개발 서버
    SERVER_REAL_TRADAR  = "real.tradar.api.com"         # 티레이더글로벌 운영 서버

    # 시스템 통보코드
    NOTIFY_SYSTEM_NEED_TO_RESTART      = 26  # 모듈변경에 때른 재시작 필요
    NOTIFY_SYSTEM_LOGIN_START          = 27  # 로그인을 시작합니다.
    NOTIFY_SYSTEM_LOGIN_REQ_USERINFO   = 28  # 사용자 정보를 요청합니다.
    NOTIFY_SYSTEM_LOGIN_RCV_USERINFO   = 29  # 사용자 정보를 수신했습니다.
    NOTIFY_SYSTEM_LOGIN_FILE_DWN_START = 30  # 파일다운로드를 시작합니다.
    NOTIFY_SYSTEM_LOGIN_FILE_DWN_END   = 31  # 파일다운로드가완료되었습니다.

    # 결과코드
    RESULT_FAIL            = -1  # API실패반환코드
    RESULT_SUCCESS         = 1000  # API성공반환코드
    RESPONSE_LOGIN_FAIL    = 1  # 로그인실패코드
    RESPONSE_LOGIN_SUCCESS = 2  # 로그인성공코드

    # 에러코드
    ERROR_MODULE_NOT_FOUND           = 11  # YuantaOpenAPI모듈을찾을수없습니다.
    ERROR_FUNCTION_NOT_FOUND         = 12  # YuantaOpenAPI함수를찾을수없습니다.
    ERROR_NOT_INITIAL                = 13  # YuantaOpenAPI초기화상태가아닙니다.

    ERROR_SYSTEM_CERT_ERROR          = 20  # 인증오류입니다.
    ERROR_SYSTEM_MAX_CON             = 21  # 다중접속한도초과입니다.
    ERROR_SYSTEM_FORCE_KILL          = 22  # 강제종료되었습니다.
    ERROR_SYSTEM_EMERGENCY           = 23  # 시스템비상상황입니다.
    ERROR_SYSTEM_INFINIT_CALL        = 24  # 이상호출로접속이종료됩니다.
    ERROR_SYSTEM_SOCKET_CLOSE        = 25  # 네트웍연결이끊어졌습니다.

    ERROR_NOT_LOGINED                = 101  # 로그인상태가아닙니다.
    ERROR_ALREADY_LOGINED            = 102  # 이미로그인된상태입니다.
    ERROR_INDEX_OUT_OF_BOUNDS        = 103  # 인덱스가가용범위를넘었습니다.
    ERROR_TIMEOUT_DATA               = 104  # 타임아웃이발생하였습니다.
    ERROR_USERINFO_NOT_FOUND         = 105  # 사용자정보를찾을수없습니다.
    ERROR_ACCOUNT_NOT_FOUND          = 106  # 계좌번호를찾을수없습니다.
    ERROR_ACCOUNT_PASSWORD_INCORRECT = 107  # 계좌비밀번호를잘못입력하셨습니다.
    ERROR_TYPE_NOT_FOUND             = 108  # 요청한타입을찾을수없습니다.

    ERROR_CERT_PASSWORD_INCORRECT    = 110  # 공인인증비밀번호가일치하지않습니다.
    ERROR_CERT_NOT_FOUND             = 111  # 공인인증서를찾을수없습니다.
    ERROR_CETT_CANCEL_SELECT         = 112  # 공인인증서선택을취소했습니다.
    ERROR_NEED_TO_UPDATE             = 113  # 공인인증업데이트가필요합니다.
    ERROR_CERT_7_ERROR               = 114  # 공인인증7회오류입니다.
    ERROR_CERT_ERROR                 = 115  # 공인인증오류입니다.
    ERROR_CERT_PASSWORD_SHORTER      = 116  # 공인인증서비밀번호가최소길이보다짧습니다.
    ERROR_ID_SHORTER                 = 117  # 로그인아이디가최소길이보다짧습니다.
    ERROR_ID_PASSWORD_SHORTER        = 118  # 로그인비밀번호가최소길이보다짧습니다.

    ERROR_CERT_OLD                   = 121  # 폐기된인증서입니다.
    ERROR_CERT_TIME_OVER             = 122  # 만료된인증서입니다.
    ERROR_CERT_STOP                  = 123  # 정지된인증서입니다.
    ERROR_CERT_NOTMATCH_SN           = 124  # SN이일치하지않는인증서입니다.
    ERROR_CERT_ETC                   = 125  # 기타오류인증서입니다.
    ERROR_CERT_TIME_OUT              = 126  # 타인증기관발급인증서검증에서타임아웃이발생하였습니다.다시시도해주십시오

    ERROR_REQUEST_FAIL               = 201  # DSO요청이실패하였습니다.
    ERROR_DSO_NOT_FOUND              = 202  # DSO를찾을수없습니다.
    ERROR_BLOCK_NOT_FOUND            = 203  # 블록을찾을수없습니다..
    ERROR_FIELD_NOT_FOUND            = 204  # 필드를찾을수없습니다.
    ERROR_REQUEST_NOT_FOUND          = 205  # 요청정보를찾을수없습니다.
    ERROR_ATTR_NOT_FOUND             = 206  # 필드의속성을찾을수없습니다.
    ERROR_REGIST_FAIL                = 207  # AUTO등록이실패하였습니다.
    ERROR_AUTO_NOT_FOUND             = 208  # AUTO를찾을수없습니다.
    ERROR_KEY_NOT_FOUND              = 209  # 요청한키를찾을수없습니다.
    ERROR_VALUE_NOT_FOUND            = 210  # 요청한값을찾을수없습니다.

    # ----------------------- 종목 정보 ---------------------------
    # 시장구분코드
    MKT_TP_CD_INTERNAL             = "0"  # 국내주식
    MKT_TP_CD_GLOBAL_STOCK         = "1"  # 해외주식
    MKT_TP_CD_GLOBAL_DERIVATIVE    = "2"  # 해외선물옵션
    MKT_TP_CD_INTERNAL_STOCK       = "3"  # 국내주식
    MKT_TP_CD_INTERNAL_KOSPIFUTURE = "4"  # 국내 코스피선물
    MKT_TP_CD_INTERNAL_KOSPIOPTION = "5"  # 국내 코스피옵션
    MKT_TP_CD_GLOBAL_FUTURE        = "6"  # 해외 선물
    MKT_TP_CD_GLOBAL_OPTION        = "7"  # 해외 옵션

    # 최종 동기화 일자 기본
    INIT_STR_DT_DEFAULT = "20180810"
    INIT_STR_MN_DEFAULT = "1530"
    # 초기화종료일은 전날자정까지로 설정 ※ 추후 변경
    #INIT_END_DT_DEFAULT = (date.today() + timedelta(days=-1)).strftime("%Y%m%d")
    INIT_END_DT_DEFAULT = "20180821"
    INIT_END_MN_DEFAULT = "1530"

    # 처리대상 종목테이블위젯 컬럼
    PROC_STK_COL_MKT_TP_CD = 0  # 시장구분
    PROC_STK_COL_STK_CD    = 1  # 종목코드
    PROC_STK_COL_STR_DT    = 2  # 처리시작일
    PROC_STK_COL_END_DT    = 3  # 처리종료일
    PROC_STK_COL_LAST_DT   = 4  # 최종처리일
    PROC_STK_COL_LAST_MN   = 5  # 최종처리시분

    # DERV_MN_QTTN_INFO 테이블 컬럼
    QTTN_MN_DATA_COL_DT       = 0
    QTTN_MN_DATA_COL_MN       = 1
    QTTN_MN_DATA_COL_STK_CD   = 2
    QTTN_MN_DATA_COL_FTPRC    = 3
    QTTN_MN_DATA_COL_HGPRC    = 4
    QTTN_MN_DATA_COL_LOPRC    = 5
    QTTN_MN_DATA_COL_CLPRC    = 6
    QTTN_MN_DATA_COL_UPDN_PRC = 7
    QTTN_MN_DATA_COL_VLUM     = 8

    # ----------------------- 시세 정보 ---------------------------
    # 코스피지수 시세 컬럼
    REAL_QTTN_COL_STK_CD = 0  # 종목코드
    REAL_QTTN_COL_TIME   = 1  # 시간(HHMM)
    REAL_QTTN_COL_CRPRC  = 2  # 현재가
    REAL_QTTN_COL_FTPRC  = 3  # 시가
    REAL_QTTN_COL_HGPRC  = 4  # 고가
    REAL_QTTN_COL_LOPRC  = 5  # 저가
    REAL_QTTN_COL_END    = 6  # 종료 표시(E)
    REAL_QTTN_COL_COUNT  = 7  # 실시간시세 컬럼수
    
    # 시세 포트
    PORT_INDEX_QTTN   = 8080  # 지수수신정보(ETF KODEX 레버리지시세 수신포트)
    PORT_TF_RCV_RSLT  = 9080  # 텐서플로 결과 수신 포트
    PORT_TF_DATA      = 7080  # 텐서플로 데이터 송수신 포트

    # 지수 기준시세(현재는 KODEX 레버리지)
    INDEX_STK_CD = "122630"

    # ----------------------- 차트 정보 ---------------------------
    # 차트 시세컬럼
    CHART_Y1_COL = "last"      # last : 티레이더글로벌 실시간시세[61] 현재가
    CHART_Y2_COL = "curjuka"  # curjuka : 티레이더 실시간시세[11] 현재가

    CHART_X_LIMIT = 60  # 차트 X축 시간범위 제한(현시각으로 부터 CHART_X_LIMIT 분 후까지)

    CHART_RY_TICK_COUNT = 10.0  # 오른쪽 Y축 상하틱(초기 KODEX 레버리지 ETF 기준 10틱 위아래)
    CHART_RY_TICK       = 5     # 오른쪽 Y축 틱단위(초기 KODEX 레버리지 ETF 기준 )
    #CHART_RY_TICK       = 0.25  # 오른쪽 Y축 틱단위(테스트 해외선물 ESM18)

    CHART_LY_TICK_COUNT = 10.0  # 왼쪽 Y축 상하틱(10틱 위아래)
    #CHART_LY_TICK       = 0.01  # 왼쪽 Y축 틱단위(테스트 해외선물 CLN18)
    CHART_LY_TICK       = 5     # 왼쪽 Y축 틱단위(초기 KODEX 레버리지 ETF 기준)

    CHART_MARKERS = ["o","x","+",".","*","s","d","^","v",">","<","p","h"] # 차트 마커
    CHART_COLORS  = ["r", "b", "g", "y", "m", "c", "w", "k"]                 # 차트 색상

    # ----------------------- 딥러닝 관련 ---------------------------
    TF_PREDICTION_INPUT_CNT  = 11   # 예측 입력값 리스트 맴버수
    TF_PREDICTION_OUTPUT_CNT = 1    # 예측 출력값 리스트 맴버수

    TF_LAYER_1_NEURON_CNT    = 20   # 1번째 은닉층 뉴런수
    TF_LAYER_2_NEURON_CNT    = 15   # 2번째 은닉층 뉴런수
    TF_LAYER_3_NEURON_CNT    = 8    # 3번째 은닉층 뉴런수
    TF_LAYER_4_NEURON_CNT    = 10   # 4번째 은닉층 뉴런수

    TF_LEARNING_STR_DT = "20180720" # 학습대상 시작일
    TF_LEARNING_STR_MN = "0900"      # 학습대상 시작시각(HHMM)
    TF_LEARNING_END_DT = "20180820" # 학습대상 종료일
    TF_LEARNING_END_MN = "1519"      # 학습대상 종료시각(HHMM)

    TF_TEST_STR_DT     = "20180821"  # 테스트 대상 시작일
    TF_TEST_STR_MN     = "0900"       # 테스트 대상 시작시각(HHMM)
    TF_TEST_END_DT     = "20180821"  # 테스트 대상 종료일
    TF_TEST_END_MN     = "1519"       # 테스트 대상 종료시각(HHMM)

    TF_LEARNING_MN_LIST = [1, 5, 10, 30] # 학습 분 리스트(예측분) 첫번째는 1분 고정
    #TF_LEARNING_MN_LIST = [1]  # 학습 분 리스트(예측분) 첫번째는 1분 고정

    TF_LEARNING_INPUT_CNT    = 11    # 학습 입력값 리스트 맴버수
    TF_LEARNING_OUTPUT_CNT   = 1     # 학습 출력값 리스트 맴버수
    TF_LEARNING_DROPOUT_RATE = 0.8   # 드롭아웃률
    TF_LEARNING_RATE         = 0.001 # 학습율
    TF_LEARNING_EPOCH        = 10    # 반복학습횟수
    TF_LEARNING_CNT          = 1000  # 학습 횟수
    
    TF_LEARNING_RNN_CELS_CNT = 128   # RNN 셀 갯수
    TF_LEARNING_RNN_STEP_CNT = 379   # RNN 입력 단계수 (하루치는 분봉으로 379분 09시00분~15시19분)

    TF_LEARNING_SAVE_DIR     = ".\TrainingSave\Minute"     # 학습데이터 저장폴더
    TF_LEARNING_SAVE_FILE    = "\TrainingCheckPoint.ckpt"  # 학습데이터 저장파일
    TF_LEARNING_LOG_DIR      = ".\TrainingLog"               # 학습로그 폴더
    TF_LEARNING_RSLT_FILE    = ".\LearningResult\LernRslt" # 학습테스트 결과저장 엑셀파일

    TF_PREDICTION_SAVE_FILE  = ".\RealPrediction\RealPrediction.xlsx"  #실시간예측 결과저장 엑셀파일



    #------------------ 공통 함수 / 변수 --------------------#

    ONE_MINUTE = timedelta(minutes=1)

    # Tick 조회
    # inMktTp : 시장구분 (KOSPI.코스피 KOSDAQ.코스닥 ETF.ETF)
    # inPrice : 가격
    def GetTicks(self,inMktTp, inPrice):
        # 코스피
        if(inMktTp == "KOSPI"):
            if(inPrice < 1000 ):
                return 1
            elif(inPrice >= 1000    and inPrice < 5000):
                return 5
            elif (inPrice >= 5000   and inPrice < 10000):
                return 10
            elif (inPrice >= 10000  and inPrice < 50000):
                return 50
            elif (inPrice >= 50000  and inPrice < 100000):
                return 100
            elif (inPrice >= 100000 and inPrice < 500000):
                return 500
            else:
                return 1000
        # 코스닥
        elif (inMktTp == "KOSDAQ"):
            if (inPrice < 1000):
                return 1
            elif (inPrice >= 1000  and inPrice < 5000):
                return 5
            elif (inPrice >= 5000  and inPrice < 10000):
                return 10
            elif (inPrice >= 10000 and inPrice < 50000):
                return 50
            else:
                return 100
        # ETF
        elif(inMktTp == "ETF"):
            return 5
        else:
            print("inMktTp 에러 : ", inMktTp)
            return False


    # 해외파생 주말기간여부 (토요일 아침 6시1분부터 월요일 아침 7시0분까지 시세없음)
    # 시장마다 유동적이긴 하나 현재 글로벌에서 수신되지 않음. (CME 기준)
    # inDt 일자 : datetime 타입
    def isQttnBlnk(self, inDt):

        WDay = inDt.weekday()
        Mn   = int(inDt.strftime("%H%M"))
        
        # 토
        if (WDay == 5 and Mn > 600 and Mn <= 2359 ):
            return True
        # 일
        if (WDay == 6):
            return True
        # 월
        if (WDay == 0 and Mn > 0 and Mn <= 700):
            return True

        return False

    # 입력일자 기준 최근 월요일 07시 리턴
    # inDt 일자 : datetime 타입
    def getMon07AM(self, inDt):
        WDay = inDt.weekday()
        Mn = int(inDt.strftime("%H%M"))
        OneDay = timedelta(days=1)
        TwoDays = timedelta(days=2)

        # 토
        if (WDay == 5 and Mn > 600 and Mn <= 2359):
            print("토")
            inDt = inDt + TwoDays
            inDt = inDt.replace(hour=7, minute=0)
        # 일
        if (WDay == 6):
            print("일")
            inDt = inDt + OneDay
            inDt = inDt.replace(hour=7, minute=0)
        # 월
        if (WDay == 0 and Mn > 0 and Mn <= 700):
            print("월")
            inDt = inDt.replace(hour=7, minute=0)

        return inDt

    # 상품별 테이블 가져오기
    def GetMktTable(self, inMktTpCd):

        Table = ""
        if(inMktTpCd == "KOSPI"):
            Table = "KOSPI_MN_QTTN_INFO"
        elif(inMktTpCd in ("DERV",CodeDef.MKT_TP_CD_GLOBAL_DERIVATIVE, CodeDef.MKT_TP_CD_GLOBAL_FUTURE, CodeDef.MKT_TP_CD_GLOBAL_OPTION)):
            Table = "DERV_MN_QTTN_INFO"
        elif(inMktTpCd in ("STK",CodeDef.MKT_TP_CD_INTERNAL, CodeDef.MKT_TP_CD_GLOBAL_STOCK, CodeDef.MKT_TP_CD_INTERNAL_STOCK)):
            Table = "STK_MN_QTTN_INFO"
        else:
            print("정의 없음: ", inMktTpCd)

        return Table
