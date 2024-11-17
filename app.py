import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
def set_chinese_font():
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']  # For Windows
        plt.rcParams['axes.unicode_minus'] = False
    except:
        st.warning("无法加载中文字体，部分文字可能显示不正确")

def clean_data(df):
    """
    Clean and prepare data for analysis
    """
    try:
        # 使用dateutil解析器处理日期，设置dayfirst=True
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], dayfirst=True)
        
        # Rename first column to 'date'
        df = df.rename(columns={df.columns[0]: 'date'})
        
        # Convert all numeric columns
        for col in df.columns[1:]:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
        
        # Sort by date
        df = df.sort_values('date')
        
        # Forward fill missing values
        df = df.fillna(method='ffill')
        
        # Reset index
        df = df.reset_index(drop=True)
        
        return df
    except Exception as e:
        st.error(f"数据清洗错误: {str(e)}")
        return None

def calculate_returns(df):
    """
    Calculate returns for all numeric columns
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    returns = df[numeric_cols].pct_change()
    returns['date'] = df['date']
    return returns

def calculate_contributions(df, target_col, factor_cols):
    """
    Calculate factor contributions using standardized coefficients
    """
    try:
        # Calculate returns
        returns = calculate_returns(df)
        
        # Prepare data for analysis
        y = returns[target_col].values
        X = returns[factor_cols].values
        
        # Remove rows with NaN
        mask = ~np.isnan(y) & ~np.any(np.isnan(X), axis=1)
        X = X[mask]
        y = y[mask]
        
        if len(y) == 0:
            raise ValueError("没有足够的有效数据进行分析")
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        y_scaled = scaler.fit_transform(y.reshape(-1, 1))
        
        # Calculate correlations
        correlations = np.corrcoef(X_scaled.T, y_scaled.T)[-1][:-1]
        
        # Calculate contributions
        contributions = pd.DataFrame(
            X_scaled * correlations.reshape(1, -1),
            columns=factor_cols,
            index=returns.index[mask]
        )
        
        # Scale back
        scale_factor = y.std() / contributions.sum(axis=1).std()
        contributions = contributions * scale_factor
        
        # Add cumulative contributions
        cum_contributions = contributions.cumsum()
        
        # Fill NaN values with 0 for the first row
        cum_contributions = cum_contributions.fillna(0)
        
        return cum_contributions
    
    except Exception as e:
        st.error(f"计算贡献度时出错: {str(e)}")
        return None

def create_bloomberg_style_chart(df, contributions, target_col):
    """
    Create Bloomberg terminal style chart
    """
    fig = make_subplots(
        rows=2, 
        cols=1,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.05,
        subplot_titles=(f'因子分解分析 - {target_col}', '价格走势')
    )

    # Add stacked area chart for cumulative contributions
    # Add stacked area chart for cumulative contributions
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']  # 10个颜色
    for col, color in zip(contributions.columns, colors):
        fig.add_trace(
            go.Scatter(
                name=col,
                x=df['date'],
                y=contributions[col],
                mode='lines',
                stackgroup='one',
                fillcolor=color,
                line=dict(width=0.5),
            ),
            row=1, 
            col=1
        )

    # Add line for actual values
    fig.add_trace(
        go.Scatter(
            name='Overall change',
            x=df['date'],
            y=df[target_col],
            line=dict(color='white', width=1)
        ),
        row=2,
        col=1
    )

    # Update layout to match Bloomberg terminal style
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='black',
        plot_bgcolor='black',
        font=dict(
            family='Arial',
            size=12,
            color='white'
        ),
        showlegend=True,
        height=800,
        margin=dict(t=50, l=50, r=50, b=50),
        legend=dict(
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor='white',
            borderwidth=1
        ),
        title=dict(
            text='因子分解分析',
            font=dict(size=24)
        )
    )

    # Update axes
    fig.update_xaxes(
        showgrid=True,
        gridwidth=0.5,
        gridcolor='rgba(128,128,128,0.2)',
        linecolor='gray',
        linewidth=1
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=0.5,
        gridcolor='rgba(128,128,128,0.2)',
        linecolor='gray',
        linewidth=1
    )

    return fig

def create_bloomberg_bar_chart(df, contributions, target_col, start_date, end_date):
    
    """
    Create Bloomberg terminal style chart with stacked bar chart and line
    """
    # Filter data for selected date range
    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
    df_filtered = df[mask].copy()
    contributions_filtered = contributions[mask].copy()
    
    # Create figure
    fig = make_subplots(
        rows=2, 
        cols=1,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.05,
        subplot_titles=(f'因子分解分析 - {target_col}', '价格走势')
    )

    # Colors matching Bloomberg's style
    # colors = ['#4DAF4A',  # Green for Monetary Policy
    #          '#377EB8',   # Blue for Macro Shocks
    #          '#FF7F00',   # Orange for US Policy
    #          '#E41A1C']   # Red for Global Risk
    # colors = plt.cm.tab20(np.linspace(0, 1, len(contributions.columns))).tolist()
    # Colors matching Bloomberg's style
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']  # 10个颜色

    # Add stacked bar chart for contributions (filtered data)
    for idx, (col, color) in enumerate(zip(contributions.columns, colors)):
        fig.add_trace(
            go.Bar(
                name=col,
                x=df_filtered['date'],
                y=contributions_filtered[col],
                marker_color=color,
                hovertemplate="%{x}<br>%{y:.2f}<extra></extra>",
                width=24*60*60*1000,  # Set bar width to 1 day in milliseconds
            ),
            row=1, 
            col=1
        )

    # Add total line for the filtered period
    fig.add_trace(
        go.Scatter(
            name='总效应',
            x=df_filtered['date'],
            y=contributions_filtered.sum(axis=1),
            mode='lines',
            line=dict(color='white', width=1.5),
            hovertemplate="%{x}<br>%{y:.2f}<extra></extra>",
        ),
        row=1, 
        col=1
    )

    # Add price line for the entire period
    fig.add_trace(
        go.Scatter(
            name='Overall change',
            x=df['date'],
            y=df[target_col],
            mode='lines',
            line=dict(color='white', width=1),
            hovertemplate="%{x}<br>%{y:.2f}<extra></extra>",
        ),
        row=2,
        col=1
    )

    # Update layout
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='black',
        plot_bgcolor='black',
        barmode='relative',
        font=dict(
            family='Arial',
            size=12,
            color='white'
        ),
        showlegend=True,
        height=800,
        margin=dict(t=50, l=50, r=50, b=50),
        legend=dict(
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor='white',
            borderwidth=1,
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='x unified'
    )

    # Update axes
    fig.update_xaxes(
        showgrid=True,
        gridwidth=0.5,
        gridcolor='rgba(128,128,128,0.2)',
        linecolor='gray',
        linewidth=1,
        showspikes=True,
        spikecolor="white",
        spikethickness=1,
        spikedash="dot",
        row=1, col=1,
        range=[start_date, end_date]  # Set x-axis range for top subplot
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=0.5,
        gridcolor='rgba(128,128,128,0.2)',
        linecolor='gray',
        linewidth=1,
        zeroline=True,
        zerolinecolor='rgba(255,255,255,0.5)',
        zerolinewidth=1
    )

    # Update hover label
    fig.update_layout(
        hoverlabel=dict(
            bgcolor="black",
            font_size=12,
            font_family="Arial"
        )
    )

    return fig

def show_correlation_heatmap(df, factor_cols, target_col):
    """
    Display correlation heatmap
    """
    set_chinese_font()
    
    # Calculate returns
    returns = calculate_returns(df)
    
    # Calculate correlation matrix
    corr = returns[factor_cols + [target_col]].corr()
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr, 
        annot=True, 
        cmap='RdYlBu', 
        center=0,
        fmt='.2f',
        square=True
    )
    plt.title('因子相关性分析')
    st.pyplot(plt)

def main():
    st.set_page_config(layout="wide")
    st.title('经济因子分解分析')
    
    uploaded_file = st.file_uploader("上传CSV文件", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read data
            df = pd.read_csv(uploaded_file)
            
            # Display raw data before cleaning
            st.subheader('原始数据预览')
            st.dataframe(df.head())
            
            # Clean data
            df = clean_data(df)
            
            if df is not None:
                st.subheader('清洗后数据预览')
                st.dataframe(df.head())
                
                col1, col2 = st.columns([2, 1])
                
                with col2:
                    st.subheader('分析参数设置')
                    
                    # Date range selection
                    min_date = df['date'].min()
                    max_date = df['date'].max()
                    
                    # Calculate default date range (last 20 trading days)
                    default_end_date = max_date
                    default_start_date = df.iloc[-20]['date'] if len(df) > 20 else min_date
                    
                    # Date range selector
                    start_date = pd.to_datetime(st.date_input(
                        "选择开始日期",
                        value=default_start_date,
                        min_value=min_date,
                        max_value=max_date
                    ))
                    
                    end_date = pd.to_datetime(st.date_input(
                        "选择结束日期",
                        value=default_end_date,
                        min_value=min_date,
                        max_value=max_date
                    ))
                    
                    # Validate date range
                    if start_date > end_date:
                        st.error("开始日期不能晚于结束日期")
                        return
                    
                    # Calculate number of trading days
                    selected_data = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
                    trading_days = len(selected_data)
                    
                    if trading_days > 20:
                        st.error("选择的时间范围超过20个交易日，请缩短时间范围")
                        return
                    
                    st.write(f"选择的交易日数: {trading_days}")
                    st.subheader('选择分析参数')
                    
                    # Select target variable
                    target_col = st.selectbox(
                        '选择目标变量（Overall change）',
                        df.columns[1:].tolist()
                    )
                    
                    # Select factor columns
                    max_factors = min(10, len([col for col in df.columns[1:] if col != target_col]))
                    n_factors = st.slider('选择因子数量', min_value=1, max_value=max_factors, value=min(4, max_factors))
                    # 动态选择因子
                    st.write('选择分解因子:')
                    factor_cols = []
                    for i in range(n_factors):
                        factor = st.selectbox(
                            f'因子 {i+1}',
                            [col for col in df.columns[1:] if col not in factor_cols and col != target_col],
                            key=f'factor_{i}'
                        )
                        factor_cols.append(factor)
                
                if len(set(factor_cols)) == n_factors:
                    try:
                        # # Show correlation analysis
                        # st.subheader('相关性分析')
                        # show_correlation_heatmap(df, factor_cols, target_col)
                        
                        # Calculate contributions
                        contributions = calculate_contributions(df, target_col, factor_cols)
                        
                        if contributions is not None:
                            # Create and display decomposition chart
                            st.subheader('分解分析图表')
                            fig = create_bloomberg_style_chart(df, contributions, target_col)
                            st.plotly_chart(fig, use_container_width=True)

                            st.subheader('分解分析图表2')
                            fig = create_bloomberg_bar_chart(df, contributions, target_col, start_date, end_date)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Add download button
                            results_df = pd.concat([
                                df[['date', target_col]],
                                contributions
                            ], axis=1)
                            
                            st.download_button(
                                label="下载分析结果",
                                data=results_df.to_csv(index=False).encode('utf-8'),
                                file_name='decomposition_results.csv',
                                mime='text/csv',
                            )
                    except Exception as e:
                        st.error(f"分析过程出错: {str(e)}")
                        st.error("请检查所选列是否包含有效的数值数据")
                else:
                    st.warning('请为每个因子选择不同的变量')
                    
        except Exception as e:
            st.error(f"文件读取错误: {str(e)}")
            st.error("请确保CSV文件格式正确")

if __name__ == "__main__":
    main()