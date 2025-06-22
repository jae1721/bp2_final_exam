import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn

# 파일 불러오기 (모델, 스케일러) =================================================================================================================

# 경로 세팅
rf_model_path = './optimized_rf_model_coffee.pkl'
mlp_model_path = './optimized_mlp_model_customer.pth'
scaler_path = './customer_scaler.pkl'

# 모델 및 스케일러 로드
rf_model = joblib.load(rf_model_path)
scaler = joblib.load(scaler_path)

# MLP 구조 정의
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size // 2)
        self.layer3 = nn.Linear(hidden_size // 2, output_size)
        self.activation = nn.GELU()
    def forward(self, x):
        h1 = self.activation(self.layer1(x))
        h2 = self.activation(self.layer2(h1))
        final_out = self.layer3(h2)
        return final_out

# 하이퍼 파라미터 설정
customer_example = pd.read_csv('./customer_segmentation_data.csv').drop(['id','last_purchase_amount'],axis=1)
input_size = customer_example.shape[1]  # 7개
hidden_size = 300
output_size = 4

# MLP 모델 로드
mlp_model = MLP(input_size, hidden_size, output_size)
mlp_model.load_state_dict(torch.load(mlp_model_path, map_location='cpu'))
mlp_model.eval()

# 마케팅 전략 정의 ===========================================================================================================================

# 고객군 레이블링
segment2kor = {
    0: '가격 민감형 소비집단',
    1: '일반 소비집단',
    2: '고소비집단',
    3: 'VIP 소비집단'
}

# 고객 특징 레이블링
segment_feature = {
    0:'합리적 가격, 프로모션, 할인에 민감한 고객',
    1:'적당한 품질, 적정가격, 일상 방문 중심의 고객',
    2:'프리미엄 제품, 부가서비스, 새로운 경험을 추구하는 고객',
    3:'매우 높은 충성도, 개별화·특별대우를 기대하는 최상위 고객' 
}

# 마케팅 전략 레이블링
marketing_strategies = {
    0: """- 가격 할인 이벤트: 시간대별·요일별 할인 (예: 모닝·해피아워, 평일특가)\n
- 할인 쿠폰/스탬프 카드 제공: 방문 횟수 적립 후 무료음료, 모바일 쿠폰 등 즉시 사용 가능 혜택\n
- 세트메뉴·번들 프로모션: 커피+디저트 세트, 친구 동반 시 할인 등의 묶음 할인\n
- 가격비교 강조 마케팅: 경쟁 카페 대비 ‘가성비’ 비교 표기, ‘저렴하게 즐기는’ 카피 사용\n
- SNS 공유시 할인 제공: 인스타그램 등 사진 업로드 시 추가 할인, 친구 추천 할인""",
    1: """- 계절별·신메뉴 프로모션: 신제품 출시 이벤트, 계절 한정 메뉴 할인\n
- 포인트 적립 멤버십 운영: 구매액/방문당 포인트 적립, 포인트로 상품 교환\n
- 친근한 매장 서비스 강화: 단골 고객 대상 맞춤 서비스, 이름 불러주기, 간단한 사은품\n
- 일상생활 제휴 이벤트: 근처 회사/학교 제휴, 테이크아웃 할인, 런치타임 세트\n
- 카페 공간 경험 마케팅: 편안한 좌석, 와이파이·콘센트 제공 등 ‘머물기 좋은 공간’ 강조""",
    2: """- 프리미엄 원두·스페셜티 음료 제공: 고급 원두, 시즌 한정 스페셜 메뉴 론칭\n
- 유료 멤버십/프리패스 프로그램: 월정액 무제한 커피, 프라이빗 좌석 예약 등 유료 구독 서비스\n
- 테이스팅 클래스, 바리스타 체험: 커피 교육/체험 클래스 개최, 원두 시음 행사\n
- 감성 굿즈·콜라보 한정판 판매: 카페 자체 굿즈, 아티스트·브랜드와 협업 상품\n
- 고급 디저트·수제 메뉴 도입: 고품질 베이커리, 직접 만든 디저트, 건강식 메뉴 강화""",
    3: """- 개인 맞춤 혜택/생일·기념일 이벤트: 생일 축하 음료/선물, 개인 취향 분석 메뉴 추천\n
- VIP 전용 멤버십·라운지:VIP만 입장 가능한 공간, 사전 예약 좌석, 전용 컨시어지 서비스\n
- 초대형 한정 이벤트: 바리스타 초청 시음, 프라이빗 테이스팅 파티, 신메뉴 우선 체험권\n
- 최상급 리워드 및 적립혜택: 구매액별 캐시백, 연간 랭킹 시상, VIP 등급별 선물\n
- 고객의견 반영 서비스: 신메뉴 개발 설문 참여, 전용 피드백 채널 운영"""
}


# 매출 예측 탭 설계 ===================================================================================================================

st.set_page_config(page_title="카페 매출·고객 예측", layout="centered")
st.title("☕ 카페 매출 예측 & 마케팅 전략 추천 프로그램")

tab1, tab2 = st.tabs(["1. 매출 예측", "2. 마케팅 전략 추천"])

with tab1:
    st.header("💸 매출 예측")
    st.write("아래 조건을 입력하면 카페의 일별 매출을 예측합니다.")

    # 입력폼 (학습된 데이터셋 내의 범위에서 벗어나지 않도록 범위 설정)
    n_customers = st.number_input("일일 방문객 수(명)", min_value=50, value=100, max_value=500, step=1)
    avg_order = st.number_input("1인당 평균 주문 금액($)", min_value=2.5, value=10.0, max_value=10.0, step=0.5)
    operating_hours = st.number_input("운영 시간(시간)", min_value=6.0, value=8.0, max_value=18.0, step=0.5)
    n_employees = st.number_input("종업원 수(명)", min_value=2, value=5, max_value=15, step=1)
    marketing_spend = st.number_input("일일 마케팅 비용($)", min_value=10.0, value=50.0, max_value=500.0, step=1.0)
    foot_traffic = st.number_input("시간당 유동인구(명)", min_value=50, value=500, max_value=1000, step=1)

    # 예측 버튼 설계 및 출력값 자동 저장
    if st.button("일일 매출 예측"):
        input_data = np.array([[n_customers, avg_order, operating_hours,
                                n_employees, marketing_spend, foot_traffic]])
        prediction = rf_model.predict(input_data)
        st.success(f"오늘의 예상 매출: 약 {prediction[0]:,.2f}달러")
        st.session_state['pred_revenue'] = prediction[0]  # 2번 탭 전달용


# 마케팅 탭 설계 =====================================================================================================================

# 매출 예측 탭으로부터 전달받은 매출액을 0~100의 점수로 전환하는 함수 선언
def scale_to_100(x, original_min=305.1, original_max=4675.86):
        return (x - original_min) / (original_max - original_min) * 100

# 소비점수 범위별로 총 4가지 레이블을 할당하는 함수 선언
def segment_by_spending(spending_score):
    if spending_score < 30:
        return 0
    elif spending_score < 60:
        return 1
    elif spending_score < 90:
        return 2
    else:
        return 3

with tab2:
    st.header("🎯 마케팅 전략 추천")
    st.write("예상 매출에 알맞는 마케팅 대상 및 추천 마케팅 전략을 안내합니다.")

    # 매출 예측탭으로부터 전달받은 값이 있다면 그걸 사용하고 아니면 사용자가 직접 입력할 수 있도록 설계
    if 'pred_revenue' in st.session_state:
        prediction = st.session_state['pred_revenue']
        st.info(f"예측된 일 매출: **{prediction:,.0f} 달러**")
    else:
        prediction = st.number_input('예상 일 매출(직접 입력)', min_value=0.0, max_value= 4675.86, value=1000.0)
            

    # 전달받거나 직접 입력한 매출을 0~100의 점수로 자동 변환
    spending_score = scale_to_100(prediction, original_min=305.1, original_max=4675.86)
    st.write(f"변환된 소비점수(0~100): **{spending_score:.1f}**")

    # 세그먼트 분류
    segment_idx = segment_by_spending(spending_score)
    segment_kor = segment2kor[segment_idx]
    segment_ft = segment_feature[segment_idx]
    strategy = marketing_strategies[segment_idx]

    st.subheader(f"🧲 추천 마케팅 대상: {segment_kor}")
    st.write(f"- 특징: {segment_ft}")
    st.subheader("🗝️ 추천 마케팅 전략")
    st.write(f"{strategy}")

st.markdown("---")
st.caption("2025 Business Programming Group4")
