import sqlite3
import pandas as pd
import os
from CodeDef import *

class DB_handler:

    # 생성자
    def __init__(self):
        # 객체변수 선언
        self.gCursor = None
        self.gDbh    = None

        # 초기화
        DbFilePath = os.path.dirname(os.path.abspath(__file__))
        self.db = sqlite3.connect(DbFilePath+"\StockIndexPrediction.db")
        self.cursor = self.db.cursor()

        return None

    # 소멸자
    def __del__(self):
        return

    # 분봉데이터 입력
    # inMktTp   : 시장구분
    # inStkCd   : 종목코드
    # inDt      : 일자
    # inTime    : 시간
    # inFtPrc   : 시가
    # inHgPrc   : 고가
    # inLoPrc   : 저가
    # inClPrc   : 종가
    # inUpDnPrc : 등락가
    # inVlum    : 거래량
    def insertMnQttn(self, inMktTp, inStkCd, inDt, inTime, inFtPrc, inHgPrc, inLoPrc, inClPrc, inUpDnPrc, inVlum):

        datas = [(inDt, inTime, inStkCd, inDt+inTime, inFtPrc, inHgPrc, inLoPrc, inClPrc, inUpDnPrc, inVlum)]

        Table = CodeDef.GetMktTable(self,inMktTp)

        try:
            self.cursor.executemany("insert into "+Table+" "
                                    "(DT, MN, STK_CD, DT_MN, FTPRC, HGPRC, LOPRC, CLPRC, UPDN_PRC, VLUM) "
                                    "values "
                                    "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", datas)
        # PK 에러시 업뎃으로 전환
        except sqlite3.IntegrityError:
            datas = [(inFtPrc, inHgPrc, inLoPrc, inClPrc, inUpDnPrc, inVlum, inDt, inTime, inDt+inTime, inStkCd)]

            self.cursor.executemany("update "+Table+" "
                                    "set FTPRC = ? "
                                    "  , HGPRC = ? "
                                    "  , LOPRC = ? "
                                    "  , CLPRC = ? "
                                    "  , UPDN_PRC = ? "
                                    "  , VLUM = ? "
                                    "where DT = ? "
                                    "and   MN = ? "
                                    "and   DT_MN = ? "
                                    "and   STK_CD = ? ", datas)

        except Exception as e:
            print("분봉입력(insertMnQttn)에러: ",e)
            return None

        self.db.commit()

        return None

    # 분봉데이터 조회
    # inMktTp   : 시장구분
    # inStkCd   : 종목코드
    # inStrDt   : 시작일자
    # inStrTime : 시작시간
    # inEndDt   : 종료일자
    # inEndTiem : 종료시간
    def queryMnQttn(self, inMktTp, inStkCd, inStrDt, inStrTime, inEndDt, inEndTime):

        Table = CodeDef.GetMktTable(self,inMktTp)
        strSQL = "select DT, MN, STK_CD, FTPRC, HGPRC, LOPRC, CLPRC, UPDN_PRC, VLUM " \
                 "from "+Table+" " \
                 "where STK_CD = '" + inStkCd + "'" \
                 "and   DT_MN between '" + inStrDt + inStrTime + "' and '" + inEndDt + inEndTime + "' " \
                 "order by DT_MN"

        #print(strSQL)
        try:
            outStkList = pd.read_sql(strSQL, self.db, index_col=None)
        except Exception as e:
            print("분봉조회(queryMnQttn)에러:",e)
            return None

        return outStkList

    # 분봉 시세 삭제
    # inMktTp : 시장구분
    # inStkCd : 종목코드
    def deleteMnQttn(self, inMktTp, inStkCd, inStrDt, inStrTime, inEndDt, inEndTime):
        datas = [(inStkCd, (inStrDt+inStrTime), (inEndDt+inEndTime))]

        Table = CodeDef.GetMktTable(self, inMktTp)

        try:
            self.cursor.executemany("delete from "+Table+" "
                                    "where STK_CD = ? "
                                    "and   DT_MN > ? "
                                    "and   DT_MN <= ? ", datas)

        except Exception as e:
            print("분봉삭제(deleteMnQttn)에러:",e)
            return None

        self.db.commit()
        return None


    # 처리대상 종목입력
    # inMktTp : 시장구분
    # inStkCd : 종목코드
    # inStrDt : 시작일자
    # inEndDt : 종료일자
    def insertProcStk(self, inMktTp, inStkCd, inStrDt, inEndDt):
        datas = [(inMktTp, inStkCd, inStrDt, inEndDt)]
        try:
            self.cursor.executemany("insert into PROC_STK_LIST "
                                    "(MKT_TP_CD, STK_CD, STR_DT, END_DT) "
                                    "values "
                                    "(?, ?, ?, ?)", datas)
        except Exception as e:
            print("처리대상 종목입력(insertProcStk)에러:",e)
            return None

        self.db.commit()
        return None

    # 처리대상 종목삭제
    # inMktTp : 시장구분
    # inStkCd : 종목코드
    def deleteProcStk(self, inMktTp, inStkCd):
        datas = [(inMktTp, inStkCd)]
        try:
            self.cursor.executemany("delete from PROC_STK_LIST "
                                    "where MKT_TP_CD = ? "
                                    "and   STK_CD = ? ", datas)
        except Exception as e:
            print("처리대상 종목삭제(deleteProcStk)에러:",e)
            return None

        self.db.commit()
        return None


    # 처리대상 종목 리스트 조회
    def queryProcStkList(self):
        strSQL = "select a.MKT_TP_CD, a.STK_CD, a.STR_DT, a.END_DT, substr(a.LAST_DT_MN,1,8) as last_dt, substr(a.LAST_DT_MN,9,4) as last_mn " \
                 "from   PROC_STK_LIST a " \
                 "where  a.mkt_tp_cd in ('0','1','3') " \
                 "and    a.end_dt > STRFTIME('%Y%m%d', DATETIME('NOW')) " \
                 "group by a.MKT_TP_CD, a.STK_CD, a.STR_DT, a.END_DT " \
                 "union all " \
                 "select a.MKT_TP_CD, a.STK_CD, a.STR_DT, a.END_DT, substr(a.LAST_DT_MN,1,8) as last_dt, substr(a.LAST_DT_MN,9,4) as last_mn " \
                 "from   PROC_STK_LIST a " \
                 "where  a.mkt_tp_cd in ('2','6','7') " \
                 "and    a.end_dt > STRFTIME('%Y%m%d', DATETIME('NOW')) " \
                 "group  by a.MKT_TP_CD, a.STK_CD, a.STR_DT, a.END_DT " \
                 "order by 1, 2 "

        outStkList = pd.read_sql(strSQL, self.db, index_col=None)
        return outStkList


    # 처리종목 초기화종료일자 업뎃
    def updateProcStkLastDtMn(self, inMktTpCd, inStkCd, inLastDtMn):
        datas = [(inLastDtMn, inMktTpCd, inStkCd)]

        try:
            self.cursor.executemany("update PROC_STK_LIST "
                                    "set LAST_DT_MN = ? "
                                    "where MKT_TP_CD = ? "
                                    "and   STK_CD = ? ", datas)
            self.db.commit()
        except Exception as e:
            print("처리종목 초기화종료일자 업뎃(updateProcStkLastDtMn)에러:",e)
            return None

        return None

    # 지수 최종시세 일자 / 시분 조회
    def queryKospiLastDtMn(self, inStkCd):

        strSQL = "select a.STK_CD, substr(max(a.dt_mn),1,8) as last_dt, substr(max(a.dt_mn),9,4) as last_mn " \
                 "from   KOSPI_MN_QTTN_INFO a " \
                 "where  a.STK_CD = '" + inStkCd + "'" \
                 "and    a.dt > STRFTIME('%Y%m%d', DATE('now', '-30 day')) " \
                 "group  by a.STK_CD "

        try:

            outStkList = pd.read_sql(strSQL, self.db, index_col=None)

        except Exception as e:
            print("지수 최종시세 일자 / 시분 조회(queryKospiLastDtMn)에러:",e)
            return None

        return outStkList




    # ---------------------- 신경망 학습 관련 ------------------ #
    # 학습대상 데이터 조회
    # 종가 종목 순서는 queryProcStkList 결과값과 순서가 같아야함.
    # 현재순서는 1분후 기준종가, 현재기준종가, 시각(HHMM), 1번 시세종가, 2번시세종가....
    # inStrDtMn : 시작 일시
    # inEndDtMn : 종료 일시
    def queryLernData(self,inStrDtMn, inEndDtMn):

        # 처리대상 종목으로 쿼리를 구성한다.
        SqlTable  = ""
        SqlSelect = ""
        SqlCond   = ""
        StkList = self.queryProcStkList()
        RowCnt = len(StkList.index)
        for iRow in range(RowCnt):
            StkCd = StkList.iat[iRow, CodeDef.PROC_STK_COL_STK_CD]
            Alias = "q"+str(iRow+1)
            # 조회항목 순서
            SqlSelect += (","+Alias+".clprc as x"+str(iRow+3) + " ")

            # 테이블 조인 조건
            SqlTable += ("left outer join DERV_MN_QTTN_INFO "+Alias+" ")
            SqlTable += ("on "+Alias+".dt_mn = ksp.dt_mn ")

            # 조회 조건
            SqlCond += ("and "+Alias+".stk_cd = '"+StkCd+"' ")

        # SQL 완성시작
        strSQL  = "select  k1.clprc as y, ksp.clprc as x1, cast(ksp.mn as float) as x2"
        strSQL += SqlSelect
        strSQL += " from KOSPI_MN_QTTN_INFO ksp "
        strSQL += " left outer join KOSPI_MN_QTTN_INFO k1 "
        strSQL += " on k1.dt_mn = strftime('%Y%m%d%H%M',datetime(substr(KSP.DT_MN,1,4)||'-'||substr(KSP.DT_MN,5,2)||'-'||substr(KSP.DT_MN,7,2)|| ' ' ||substr(KSP.DT_MN,9,2)||':'||substr(KSP.DT_MN,11,2), '1 minutes')) "
        strSQL += SqlTable
        strSQL += "where ksp.dt_mn between '"+inStrDtMn+"' and '"+inEndDtMn+"' "
        strSQL += "and ksp.mn between '0900' and '1519' " # 장시간만
        strSQL += SqlCond
        strSQL +="order by ksp.dt_mn "

        print(strSQL)
        outLernData = None
        try:
            outLernData = pd.read_sql(strSQL, self.db, index_col=None)
        except Exception as e:
            print("학습대상 데이터 조회(queryLernData)에러:",e)
            return None

        return outLernData

    # 가장최근 시세 셋 조회
    def queryFstTFQttn(self):

        # 처리대상 종목으로 쿼리를 구성한다.
        strSQL = "select  "
        StkList = self.queryProcStkList()
        RowCnt = len(StkList.index)
        for iRow in range(RowCnt):
            StkCd = StkList.iat[iRow, CodeDef.PROC_STK_COL_STK_CD]
            Alias = "q" + str(iRow)
            strSQL += " (select clprc from DERV_MN_QTTN_INFO where stk_cd = '"+StkCd+"' and dt_mn = (select max(dt_mn) from DERV_MN_QTTN_INFO x where stk_cd = '"+StkCd+"' and dt >= strftime('%Y%m%d',date('now','-20 days')))) as "+Alias+" "

        try:
            print(strSQL)
            Qttn = pd.read_sql(strSQL, self.db, index_col=None)

        except Exception as e:
            print("가장최근 시세 셋 조회(queryFstTFQttn)에러:",e)
            return None

        FstTFQttn = []
        if(len(Qttn.index) > 0):
            FstTFQttn = list(Qttn.iloc[0])
        else:
            FstTFQttn = [0] * RowCnt

        return FstTFQttn


    # 저장된 테스트 시세 조회
    def querySavedTestQttn(self):

        # 처리대상 종목으로 쿼리를 구성한다.
        SqlTable = ""
        SqlSelect = ""
        SqlCond = ""
        StkList = self.queryProcStkList()
        RowCnt = len(StkList.index)
        for iRow in range(RowCnt):
            StkCd = StkList.iat[iRow, CodeDef.PROC_STK_COL_STK_CD]
            Alias = "q" + str(iRow + 1)
            # 조회항목 순서
            SqlSelect += ("," + Alias + ".clprc as x" + str(iRow + 3) + " ")

            # 테이블 조인 조건
            SqlTable += ("left outer join DERV_MN_QTTN_INFO " + Alias + " ")
            SqlTable += ("on " + Alias + ".dt_mn = ksp.dt_mn ")

            # 조회 조건
            SqlCond += ("and " + Alias + ".stk_cd = '" + StkCd + "' ")

        # SQL 완성시작
        strSQL = "select  ksp.clprc as x1, cast(ksp.mn as float) as x2"
        strSQL += SqlSelect
        strSQL += " from KOSPI_MN_QTTN_INFO ksp "
        strSQL += SqlTable
        strSQL += "where ksp.dt_mn between '" + (CodeDef.TF_TEST_STR_DT+CodeDef.TF_TEST_STR_MN) + "' and '" + (CodeDef.TF_TEST_END_DT+CodeDef.TF_TEST_END_MN) + "' "
        strSQL += "and ksp.mn between '0900' and '1519' "  # 장시간만
        strSQL += SqlCond
        strSQL += "order by ksp.dt_mn "

        #print(strSQL)
        outTestData = None
        try:
            outTestData = pd.read_sql(strSQL, self.db, index_col=None)
        except Exception as e:
            print("저장된 테스트 시세 조회(querySavedTestQttn)에러:", e)
            return None

        return outTestData