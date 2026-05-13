import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
from datetime import timedelta
import numpy as np
import io

# Set page config
st.set_page_config(page_title="AI Budget Optimizer", page_icon="💰", layout="wide")

# Custom CSS for styling
st.markdown("""
<style>
    .reportview-container .main .block-container{
        max-width: 1200px;
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }
    .metric-card {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
        color: white;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.title("💰 AI 기반 광고 예산 최적화 시뮬레이터")
st.markdown("과거 광고 성과 데이터를 분석하여 매체 및 광고유형별 최적의 예산 분배를 제안합니다.")

# Sidebar
with st.sidebar:
    st.header("설정 및 입력")
    
    # 1. File Upload
    uploaded_file = st.file_uploader("광고 리포트 업로드 (TSV, CSV, TXT)", type=['tsv', 'txt', 'csv'])
    
    # 2. Total Budget Input
    total_budget = st.number_input("차월 총 예산 설정 (원)", min_value=100000, max_value=10000000000, value=10000000, step=100000)
    
    # 3. Gemini API Key
    api_key = st.text_input("Gemini API Key", type="password")
    
    st.markdown("---")
    st.markdown("### 데이터 스펙 안내")
    st.info("""
    - 1열: 날짜 (Date)
    - 2열: 매체 (Platform)
    - 3열: 광고유형 (Ad Type)
    - 5열: 클릭수 (Clicks)
    - 6열: 광고비용 (Spend)
    - 8열: CPC
    - 9열: 전환수 (Conversions)
    - 11열: CPA
    """)

def preprocess_data(df):
    try:
        # Check if we have enough columns
        if len(df.columns) < 11:
            st.error("업로드된 데이터의 컬럼 수가 11개 미만입니다. 파일 형식을 확인해주세요.")
            return None

        # Rename columns based on index (0-based)
        cols = list(df.columns)
        df = df.rename(columns={
            cols[0]: 'Date',
            cols[1]: 'Platform',
            cols[2]: 'Ad Type',
            cols[4]: 'Clicks',
            cols[5]: 'Spend',
            cols[7]: 'CPC',
            cols[8]: 'Conversions',
            cols[10]: 'CPA'
        })
        
        # Keep only necessary columns
        essential_cols = ['Date', 'Platform', 'Ad Type', 'Clicks', 'Spend', 'CPC', 'Conversions', 'CPA']
        df = df[essential_cols].copy()
        
        # Convert Date to datetime
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        
        # Clean numeric columns (remove commas and convert to float)
        num_cols = ['Clicks', 'Spend', 'CPC', 'Conversions', 'CPA']
        for col in num_cols:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace(',', '', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
        return df
    except Exception as e:
        st.error(f"데이터 전처리 중 오류가 발생했습니다: {e}")
        return None

def calculate_optimal_budget(df, total_budget):
    # 1. Calculate weights based on date
    max_date = df['Date'].max()
    three_months_ago = max_date - pd.Timedelta(days=90)
    
    df['Weight'] = np.where(df['Date'] >= three_months_ago, 1.2, 1.0)
    
    # 2. Apply weights to base metrics
    df['W_Spend'] = df['Spend'] * df['Weight']
    df['W_Clicks'] = df['Clicks'] * df['Weight']
    df['W_Conversions'] = df['Conversions'] * df['Weight']
    
    # 3. Aggregate by Platform and Ad Type
    agg_df = df.groupby(['Platform', 'Ad Type']).agg({
        'Spend': 'sum', # Actual total spend
        'W_Spend': 'sum',
        'W_Clicks': 'sum',
        'W_Conversions': 'sum'
    }).reset_index()
    
    # Calculate aggregated weighted CPC and CPA
    agg_df['W_CPC'] = np.where(agg_df['W_Clicks'] > 0, agg_df['W_Spend'] / agg_df['W_Clicks'], 0)
    agg_df['W_CPA'] = np.where(agg_df['W_Conversions'] > 0, agg_df['W_Spend'] / agg_df['W_Conversions'], 0)
    
    # Combine Platform and Ad Type for combination name
    agg_df['Combination'] = agg_df['Platform'] + " - " + agg_df['Ad Type']
    
    # 4. Identify combinations with 0 spend (Test Combinations)
    # If all have spend > 0, we can't allocate test budget based on this. We'll just assume items with < 1% of total past spend as 'test'.
    total_past_spend = agg_df['Spend'].sum()
    if total_past_spend > 0:
        agg_df['Is_Test'] = agg_df['Spend'] < (total_past_spend * 0.005) # less than 0.5% is considered no/low history
    else:
        agg_df['Is_Test'] = True
        
    test_combos = agg_df[agg_df['Is_Test']]
    main_combos = agg_df[~agg_df['Is_Test']].copy()
    
    test_budget_total = 0
    test_allocation = []
    
    if len(test_combos) > 0:
        test_budget_total = total_budget * 0.05
        budget_per_test = test_budget_total / len(test_combos)
        for _, row in test_combos.iterrows():
            test_allocation.append({
                'Combination': row['Combination'],
                'Platform': row['Platform'],
                'Ad Type': row['Ad Type'],
                'Score': 0,
                'Proposed Budget': budget_per_test,
                'Category': 'Test Budget (5%)'
            })
            
    remaining_budget = total_budget - test_budget_total
    
    if len(main_combos) == 0:
        # All are test combos
        result_df = pd.DataFrame(test_allocation)
        return result_df, agg_df
        
    # 5. Scoring Model for Main Combinations
    # Normalize (Min-Max)
    def min_max_scale(series, reverse=False):
        s_min = series.min()
        s_max = series.max()
        if s_max == s_min:
            return pd.Series([0.5] * len(series), index=series.index)
        scaled = (series - s_min) / (s_max - s_min)
        if reverse:
            return 1 - scaled
        return scaled
        
    main_combos['Norm_Clicks'] = min_max_scale(main_combos['W_Clicks'])
    main_combos['Norm_Conversions'] = min_max_scale(main_combos['W_Conversions'])
    
    # For CPC and CPA, lower is better. Also filter out 0 values if they mean 'no data' to avoid skewing.
    # Actually, 0 CPA usually means no conversions, which is bad. 
    # Let's replace 0 with max value for CPC and CPA before scaling so they get the lowest score.
    max_cpc = main_combos[main_combos['W_CPC'] > 0]['W_CPC'].max()
    max_cpa = main_combos[main_combos['W_CPA'] > 0]['W_CPA'].max()
    
    main_combos['W_CPC_adj'] = main_combos['W_CPC'].replace(0, max_cpc if pd.notna(max_cpc) else 0)
    main_combos['W_CPA_adj'] = main_combos['W_CPA'].replace(0, max_cpa if pd.notna(max_cpa) else 0)
    
    main_combos['Norm_CPC'] = min_max_scale(main_combos['W_CPC_adj'], reverse=True)
    main_combos['Norm_CPA'] = min_max_scale(main_combos['W_CPA_adj'], reverse=True)
    
    # Calculate integrated score
    main_combos['Score'] = (main_combos['Norm_Clicks'] + main_combos['Norm_Conversions'] + main_combos['Norm_CPC'] + main_combos['Norm_CPA']) / 4
    
    # Ensure scores are > 0 to allocate budget
    main_combos['Score'] = main_combos['Score'].clip(lower=0.01)
    
    # 6. Allocate Budget with 40% Cap
    cap_ratio = 0.40
    max_budget_per_combo = remaining_budget * cap_ratio
    
    main_combos['Temp_Alloc'] = remaining_budget * (main_combos['Score'] / main_combos['Score'].sum())
    
    # Apply Cap repeatedly until no combo exceeds cap
    allocated = False
    while not allocated:
        over_cap_mask = main_combos['Temp_Alloc'] > max_budget_per_combo
        if not over_cap_mask.any() or len(main_combos) == 1:
            allocated = True
            break
            
        # Cap those over and redistribute the rest
        excess_budget = main_combos.loc[over_cap_mask, 'Temp_Alloc'].sum() - (over_cap_mask.sum() * max_budget_per_combo)
        main_combos.loc[over_cap_mask, 'Temp_Alloc'] = max_budget_per_combo
        
        under_cap_mask = ~over_cap_mask
        if under_cap_mask.sum() > 0:
            under_cap_score_sum = main_combos.loc[under_cap_mask, 'Score'].sum()
            main_combos.loc[under_cap_mask, 'Temp_Alloc'] += excess_budget * (main_combos.loc[under_cap_mask, 'Score'] / under_cap_score_sum)
        else:
            allocated = True # Edge case: all over cap, which shouldn't happen if len > 1 / cap_ratio

    main_combos['Proposed Budget'] = main_combos['Temp_Alloc']
    main_combos['Category'] = 'Main Budget (Based on Perf)'
    
    # 7. Combine results
    main_allocation = main_combos[['Combination', 'Platform', 'Ad Type', 'Score', 'Proposed Budget', 'Category']].to_dict('records')
    
    final_allocation = pd.DataFrame(main_allocation + test_allocation)
    final_allocation = final_allocation.sort_values(by='Proposed Budget', ascending=False).reset_index(drop=True)
    
    return final_allocation, agg_df

def generate_ai_insights(api_key, allocation_df):
    genai.configure(api_key=api_key)
    
    # Convert dataframe to JSON string for prompt
    data_str = allocation_df[['Combination', 'Score', 'Proposed Budget', 'Category']].to_json(orient='records', force_ascii=False)
    
    prompt = f"""
    당신은 15년차 시니어 미디어 플래너이자 데이터 사이언티스트입니다.
    아래는 과거 데이터를 기반으로 최적화 알고리즘(성과 4개 지표 정규화, 최근 3개월 가중치, 최대 40% Cap, 5% 테스트 예산 적용)을 통해 산출된 매체/광고유형별 차월 예산 분배 제안 결과(JSON)입니다.

    결과 데이터:
    {data_str}

    이 예산 분배 결과에 대해 광고주가 납득할 수 있도록 다음 두 가지를 작성해 주세요.
    1. 이 예산 분배안이 최적인 이유와 전략적 근거 3가지 (데이터 기반의 논리적 설명)
    2. 향후 운영 가이드 (테스트 예산 활용 방안 및 성과 모니터링 포인트 등)

    응답은 마크다운 형식으로 프로페셔널하게 작성해 주세요.
    """
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        if "404" in str(e):
            try:
                model_fallback = genai.GenerativeModel('gemini-pro')
                response_fallback = model_fallback.generate_content(prompt)
                return response_fallback.text + "\n\n*(💡 참고: 사용하신 API 키 환경에서 1.5 모델 접근이 제한되어 있어, 가장 안정적인 범용 모델인 `gemini-pro`를 통해 분석 결과를 도출했습니다.)*"
            except Exception as e2:
                return f"AI 분석 중 오류가 발생했습니다 (대체 모델 접근도 실패했습니다): {e2}"
        return f"AI 분석 중 알 수 없는 오류가 발생했습니다: {e}"

if uploaded_file is not None:
    # Read data
    try:
        file_ext = uploaded_file.name.split('.')[-1].lower()
        sep = ',' if file_ext == 'csv' else '\t'
        
        encodings = ['utf-8', 'utf-16', 'cp949', 'euc-kr']
        raw_df = None
        
        for enc in encodings:
            try:
                uploaded_file.seek(0)
                raw_df = pd.read_csv(uploaded_file, sep=sep, encoding=enc)
                if not raw_df.empty:
                    break
            except Exception:
                continue
                
        if raw_df is None or raw_df.empty:
            st.error("파일을 읽을 수 없습니다. 인코딩이 맞지 않거나 데이터가 비어있습니다.")
            df = None
        else:
            df = preprocess_data(raw_df)
    except Exception as e:
        st.error(f"파일 처리 중 오류가 발생했습니다: {str(e)}")
        df = None
        
    if df is not None:
        # Calculate optimal budget
        allocation_df, agg_df = calculate_optimal_budget(df, total_budget)
        
        # Calculate number of unique months in data to compute monthly average spend
        num_months = max(1, df['Date'].dt.to_period('M').nunique())
        
        # Merge current spend into allocation_df for comparison
        current_spend_df = agg_df[['Combination', 'Spend']].copy()
        current_spend_df['Current Spend (Monthly Avg)'] = current_spend_df['Spend'] / num_months
        
        comp_df = pd.merge(allocation_df, current_spend_df, on='Combination', how='left').fillna(0)
        
        # Tabs
        tab1, tab2 = st.tabs(["📊 과거 성과 분석", "💡 최적화 제안 및 시뮬레이션"])
        
        with tab1:
            st.subheader("매체 및 광고유형별 누적 성과")
            st.dataframe(
                agg_df[['Combination', 'Spend', 'W_Clicks', 'W_Conversions', 'W_CPC', 'W_CPA']]
                .rename(columns={'Spend': '총 지출액', 'W_Clicks':'가중 클릭수', 'W_Conversions':'가중 전환수', 'W_CPC':'가중 CPC', 'W_CPA':'가중 CPA'})
                .style.format({
                    '총 지출액': '{:,.0f}', 
                    '가중 클릭수': '{:,.0f}', 
                    '가중 전환수': '{:,.0f}', 
                    '가중 CPC': '{:,.0f}', 
                    '가중 CPA': '{:,.0f}'
                })
                .background_gradient(cmap='Blues', subset=['총 지출액', '가중 클릭수', '가중 전환수', '가중 CPC', '가중 CPA']),
                use_container_width=True
            )
            
            # Monthly trend line chart
            df['Month'] = df['Date'].dt.to_period('M').astype(str)
            df['Combination'] = df['Platform'] + " - " + df['Ad Type']
            monthly_trend = df.groupby(['Month', 'Combination'])[['Spend', 'Conversions']].sum().reset_index()
            
            col1, col2 = st.columns(2)
            with col1:
                fig1 = px.line(monthly_trend, x='Month', y='Spend', color='Combination', markers=True, title="월별 지출액 추이")
                fig1.update_layout(xaxis_title="월", yaxis_title="지출액", template="plotly_dark")
                st.plotly_chart(fig1, use_container_width=True)
            with col2:
                fig2 = px.line(monthly_trend, x='Month', y='Conversions', color='Combination', markers=True, title="월별 전환수 추이")
                fig2.update_layout(xaxis_title="월", yaxis_title="전환수", template="plotly_dark")
                st.plotly_chart(fig2, use_container_width=True)

        with tab2:
            st.subheader("최적화된 예산 배분 결과")
            
            # KPI Cards
            c1, c2, c3 = st.columns(3)
            c1.metric("총 예산", f"{total_budget:,.0f} 원")
            test_budget_amt = allocation_df[allocation_df['Category'].str.contains('Test')]['Proposed Budget'].sum()
            c2.metric("테스트 예산 (5% 내외)", f"{test_budget_amt:,.0f} 원")
            c3.metric("대상 조합 수", f"{len(allocation_df)} 개")
            
            # Pie charts for comparison
            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                fig_curr = px.pie(comp_df, values='Current Spend (Monthly Avg)', names='Combination', title='기존 예산 비중 (월 평균)', hole=0.4)
                fig_curr.update_traces(textposition='inside', textinfo='percent+label')
                fig_curr.update_layout(template="plotly_dark")
                st.plotly_chart(fig_curr, use_container_width=True)
                
            with col_chart2:
                fig_prop = px.pie(comp_df, values='Proposed Budget', names='Combination', title='제안 예산 비중 (Optimized)', hole=0.4)
                fig_prop.update_traces(textposition='inside', textinfo='percent+label')
                fig_prop.update_layout(template="plotly_dark")
                st.plotly_chart(fig_prop, use_container_width=True)
            
            st.markdown("### 예산 배분 상세 표")
            st.dataframe(
                comp_df[['Combination', 'Category', 'Score', 'Current Spend (Monthly Avg)', 'Proposed Budget']]
                .rename(columns={'Category':'분류', 'Score':'성과 점수', 'Current Spend (Monthly Avg)':'기존 지출액(월평균)', 'Proposed Budget':'제안 예산액'})
                .style.format({'성과 점수': '{:.3f}', '기존 지출액(월평균)': '{:,.0f}', '제안 예산액': '{:,.0f}'}),
                use_container_width=True
            )
            
            st.markdown("---")
            
            st.markdown("### 📈 시뮬레이션: 기존 vs 제안 예산 성과 비교")
            st.info("※ 각 매체/유형별 최근 성과(가중 CPC/CPA)가 동일하게 유지된다고 가정할 때의 예상 성과입니다.")
            
            # calculate expected metrics
            rates_df = agg_df[['Combination', 'W_CPC', 'W_CPA']]
            sim_df = pd.merge(comp_df, rates_df, on='Combination', how='left')
            
            # Handle 0 or NaNs by using overall average
            avg_cpc = agg_df[agg_df['W_CPC'] > 0]['W_CPC'].mean()
            avg_cpa = agg_df[agg_df['W_CPA'] > 0]['W_CPA'].mean()
            sim_df['W_CPC'] = sim_df['W_CPC'].replace(0, avg_cpc)
            sim_df['W_CPA'] = sim_df['W_CPA'].replace(0, avg_cpa)
            
            sim_df['Curr_Clicks'] = np.where(sim_df['W_CPC'] > 0, sim_df['Current Spend (Monthly Avg)'] / sim_df['W_CPC'], 0)
            sim_df['Curr_Conv'] = np.where(sim_df['W_CPA'] > 0, sim_df['Current Spend (Monthly Avg)'] / sim_df['W_CPA'], 0)
            sim_df['Prop_Clicks'] = np.where(sim_df['W_CPC'] > 0, sim_df['Proposed Budget'] / sim_df['W_CPC'], 0)
            sim_df['Prop_Conv'] = np.where(sim_df['W_CPA'] > 0, sim_df['Proposed Budget'] / sim_df['W_CPA'], 0)
            
            t_curr_spend = sim_df['Current Spend (Monthly Avg)'].sum()
            t_prop_spend = sim_df['Proposed Budget'].sum()
            
            t_curr_clicks = sim_df['Curr_Clicks'].sum()
            t_curr_conv = sim_df['Curr_Conv'].sum()
            t_prop_clicks = sim_df['Prop_Clicks'].sum()
            t_prop_conv = sim_df['Prop_Conv'].sum()
            
            # display metrics
            sc1, sc2, sc3, sc4 = st.columns(4)
            
            diff_clicks = t_prop_clicks - t_curr_clicks
            diff_conv = t_prop_conv - t_curr_conv
            
            t_curr_cpc = t_curr_spend / t_curr_clicks if t_curr_clicks > 0 else 0
            t_prop_cpc = t_prop_spend / t_prop_clicks if t_prop_clicks > 0 else 0
            diff_cpc = t_prop_cpc - t_curr_cpc
            
            t_curr_cpa = t_curr_spend / t_curr_conv if t_curr_conv > 0 else 0
            t_prop_cpa = t_prop_spend / t_prop_conv if t_prop_conv > 0 else 0
            diff_cpa = t_prop_cpa - t_curr_cpa
            
            sc1.metric("예상 총 클릭수", f"{t_prop_clicks:,.0f} 회", f"{diff_clicks:,.0f} 회")
            sc2.metric("예상 총 전환수", f"{t_prop_conv:,.0f} 건", f"{diff_conv:,.0f} 건")
            sc3.metric("예상 평균 CPC", f"{t_prop_cpc:,.0f} 원", f"{diff_cpc:,.0f} 원", delta_color="inverse")
            sc4.metric("예상 평균 CPA", f"{t_prop_cpa:,.0f} 원", f"{diff_cpa:,.0f} 원", delta_color="inverse")
            
            st.markdown("---")
            
            if api_key:
                st.subheader("🤖 AI 전략 분석 리포트")
                with st.spinner('Gemini AI가 전략적 근거를 분석하고 있습니다...'):
                    ai_report = generate_ai_insights(api_key, allocation_df)
                    st.markdown(ai_report)
            else:
                st.info("💡 사이드바에 Gemini API Key를 입력하시면, 매체 최적화 및 운영 가이드에 대한 상세한 AI 분석 리포트를 확인하실 수 있습니다.")
else:
    # Landing / Placeholder
    st.info("좌측 사이드바에서 광고 리포트(TSV) 파일을 업로드해주세요.")
    st.image("https://images.unsplash.com/photo-1551288049-bebda4e38f71?auto=format&fit=crop&q=80&w=1200&h=400", use_column_width=True)
