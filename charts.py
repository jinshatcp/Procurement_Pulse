"""Plotly figures shared by the dashboard. All plots derive from displayed records."""
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
INK='#193847';BLUE='#297fa7';TEAL='#118477';AMBER='#c78a45';GRAY='#afbdc7'

def styled(fig,height=340):
    fig.update_layout(template='plotly_white',height=height,margin=dict(l=16,r=16,t=22,b=20),
        paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Arial, sans-serif',color=INK,size=12),
        legend=dict(orientation='h',y=1.13,x=0,title_text=''),
        xaxis=dict(showgrid=False,zeroline=False),yaxis=dict(gridcolor='#e9eef1',zeroline=False),
        hoverlabel=dict(bgcolor='white',font_size=12))
    return fig

def cycle_trend(df):
    monthly=df.groupby('month')[['cycle_days','planned_days']].mean().reset_index()
    f=go.Figure()
    f.add_trace(go.Scatter(x=monthly.month,y=monthly.planned_days,name='Planned',mode='lines',line=dict(color=GRAY,dash='dash',width=2),hovertemplate='%{x}<br>Planned: %{y:.1f} days<extra></extra>'))
    f.add_trace(go.Scatter(x=monthly.month,y=monthly.cycle_days,name='Actual',mode='lines+markers',line=dict(color=BLUE,width=3),marker=dict(size=7),fill='tozeroy',fillcolor='rgba(41,127,167,.07)',hovertemplate='%{x}<br>Actual: %{y:.1f} days<extra></extra>'))
    f.update_yaxes(title='Average days',rangemode='tozero');return styled(f)

def category_spend(df):
    grouped=df.groupby('category',as_index=False)[['budget_aed','actual_aed']].sum()
    f=go.Figure()
    for k,label,color in [('budget_aed','Approved',GRAY),('actual_aed','Actual',TEAL)]:
        f.add_trace(go.Bar(x=grouped[k]/1e6,y=grouped.category,orientation='h',name=label,marker_color=color,hovertemplate='%{y}<br>AED %{x:.2f}m<extra></extra>'))
    f.update_layout(barmode='group');f.update_xaxes(title='AED millions');return styled(f,300)

def score_distribution(df,cutoff):
    f=go.Figure()
    if 'injected_anomaly' in df:
        truth=df.injected_anomaly.astype(str).str.lower().isin(['true','1'])
        for mask,name,color in [(~truth,'Normal',BLUE),(truth,'Injected anomaly',AMBER)]:
            f.add_trace(go.Histogram(x=df.loc[mask,'anomaly_score'],name=name,marker_color=color,opacity=.7,nbinsx=25))
    else:f.add_trace(go.Histogram(x=df.anomaly_score,name='Cases',marker_color=BLUE,nbinsx=25))
    f.add_vline(x=cutoff,line_dash='dash',line_color=INK,annotation_text='Alert cutoff')
    f.update_layout(barmode='overlay');f.update_xaxes(title='Anomaly score (larger = more unusual)');f.update_yaxes(title='Held-out cases');return styled(f)

def scatter(df):
    d=df.copy();d['cycle_ratio']=d.cycle_days/d.planned_days
    d['Review signal']=d.flagged.map({True:'Flagged',False:'Unflagged'})
    f=px.scatter(d,x='cycle_ratio',y='variance_pct',color='Review signal',
        hover_data=['id','supplier','change_orders'],color_discrete_map={'Flagged':AMBER,'Unflagged':BLUE},opacity=.72)
    f.add_vline(x=1.35,line_dash='dot',line_color=GRAY)
    f.add_hline(y=10,line_dash='dot',line_color=GRAY)
    f.update_xaxes(title='Actual / planned cycle ratio');f.update_yaxes(title='Spend variance (%)');return styled(f)

def matrix(metrics):
    z=[[metrics['tn'],metrics['fp']],[metrics['fn'],metrics['tp']]]
    f=go.Figure(go.Heatmap(z=z,x=['Not flagged','Flagged'],y=['Normal','Injected anomaly'],
        colorscale=[[0,'#edf4f8'],[1,'#216587']],showscale=False,text=z,texttemplate='%{text}',
        textfont=dict(size=24),hovertemplate='Truth: %{y}<br>Prediction: %{x}<br>Cases: %{z}<extra></extra>',xgap=8,ygap=8))
    f.update_yaxes(autorange='reversed',title='Synthetic truth');f.update_xaxes(title='Prediction');return styled(f,300)

def model_bars(comparison):
    f=go.Figure()
    for key,label,color in [('precision','Precision',BLUE),('recall','Recall',TEAL),('f1','F1',AMBER)]:
        f.add_trace(go.Bar(x=comparison['Model'],y=comparison[key],name=label,marker_color=color,hovertemplate='%{x}<br>%{y:.1%}<extra></extra>'))
    f.update_yaxes(tickformat='.0%',range=[0,1.08]);f.update_layout(barmode='group');return styled(f)
