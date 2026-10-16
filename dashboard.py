"""Procurement Pulse V2. Start with python -m streamlit run dashboard.py."""
from pathlib import Path
import json,html
import joblib,numpy as np,pandas as pd,streamlit as st
from core import score,evaluate,features,data_report,MODEL_NAMES
import charts
ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='Procurement Pulse · ML Workbench',page_icon='◈',layout='wide',initial_sidebar_state='expanded')
st.markdown('''<style>
@import url('');
:root {--ink:#193847;--blue:#297fa7}
.stApp{background:#f4f7f9}header[data-testid="stHeader"]{background:rgba(244,247,249,.92)}
section[data-testid="stSidebar"]{background:#142f40;color:#d7e3eb}
section[data-testid="stSidebar"] h1,section[data-testid="stSidebar"] h2,section[data-testid="stSidebar"] h3,section[data-testid="stSidebar"] label,section[data-testid="stSidebar"] p{color:#d7e3eb!important}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p{color:#99b2c0!important}
section[data-testid="stSidebar"] [data-testid="stRadio"] label{border-radius:7px;padding:7px 0}
.block-container{padding-top:2.3rem;max-width:1550px;padding-bottom:2rem}
h1{font-size:2.2rem!important;font-weight:650!important;letter-spacing:-1.2px!important;color:#193847!important}
h2{font-size:1.25rem!important;color:#193847!important}h3{font-size:1rem!important;color:#193847!important}
[data-testid="stMetric"]{background:white;border:1px solid #dfe8ed;border-radius:12px;padding:20px 23px;box-shadow:0 4px 12px #122c3903}
[data-testid="stMetricLabel"] p{color:#708694;font-size:.78rem}
[data-testid="stMetricValue"]{font-size:1.8rem;color:#193847;font-weight:600}
[data-testid="stVerticalBlockBorderWrapper"]>div{border-radius:12px!important}
[data-testid="stDataFrame"]{border-radius:8px;overflow:hidden}
.eyebrow{letter-spacing:2px;font-size:10px;color:#6e8c9d;font-weight:600;margin-bottom:8px}
.hero-note{font-size:13px;color:#728894;margin-bottom:22px}
.demo-strip{display:flex;gap:12px;align-items:center;background:#edf3e9;border:1px solid #dce8d4;color:#627c52;padding:12px 17px;border-radius:9px;font-size:12px;margin:2px 0 22px}
.demo-strip b{color:#425f34}.brand{font-size:25px;font-weight:650;color:#eff6fa;letter-spacing:-1px}.brand i{color:#80c9cd;font-style:normal}.brand-sub{font-size:10px;letter-spacing:2px;color:#93adbd;margin:5px 0 30px}.sidebar-rule{height:1px;background:#365061;margin:22px 0}
.label{font-size:10px;letter-spacing:1.2px;color:#7c96a5;margin:3px 0 10px}.signal{background:#faf1e6;border-left:3px solid #c78a45;border-radius:4px;padding:13px 16px;color:#886332;font-size:13px}.note{font-size:12px;color:#6d8490;line-height:1.8}.footer{font-size:10px;color:#8aa0ad;border-top:1px solid #dae4ea;padding-top:16px;margin-top:30px}.tag{background:#eaf3f6;color:#246b87;border-radius:5px;padding:5px 9px;font-size:10px;display:inline-block}.case-id{font-size:22px;letter-spacing:-.4px;font-weight:600;color:#193847}
button[kind="primary"]{border-radius:7px!important;background:#216985!important;border-color:#216985!important}
</style>'''.replace("@import url('');",''),unsafe_allow_html=True)

@st.cache_resource
def load_bundle(stamp):
    # Only a locally generated, trusted bundle is loaded. No model uploads.
    return joblib.load(ROOT/'outputs'/'model_bundle.joblib')

@st.cache_data
def load_data(stamp):return pd.read_csv(ROOT/'data'/'synthetic_procurement.csv')

def pct(v):return '—' if v is None or pd.isna(v) else f'{v:.1%}'
def money(v):return f'AED {v/1e6:,.2f}m' if abs(v)>=1e6 else f'AED {v:,.0f}'
def chart(fig,key):st.plotly_chart(fig,width='stretch',key=key,config={'displaylogo':False})
def caption(text):st.markdown(f'<div class="note">{html.escape(text)}</div>',unsafe_allow_html=True)
def csv_download(df,label,name,key):st.download_button(label,df.to_csv(index=False),name,mime='text/csv',key=key)
def title(eyebrow,heading,description):
    st.markdown(f'<div class="eyebrow">{eyebrow}</div>',unsafe_allow_html=True)
    st.title(heading)
    st.markdown(f'<div class="hero-note">{description}</div>',unsafe_allow_html=True)

bundle_path=ROOT/'outputs'/'model_bundle.joblib'
if not bundle_path.exists():
    title('LOCAL ML WORKBENCH','Welcome to Procurement Pulse.','Generate the demonstration data and train both models to get started.')
    st.code('python run.py all\npython -m streamlit run dashboard.py',language='bash');st.stop()
bundle=load_bundle(bundle_path.stat().st_mtime_ns)
raw=load_data((ROOT/'data'/'synthetic_procurement.csv').stat().st_mtime_ns)
if 'reviews' not in st.session_state:st.session_state.reviews={}
with st.sidebar:
    st.markdown('<div class="brand">procurement <i>pulse.</i></div><div class="brand-sub">PERFORMANCE + ASSURANCE</div>',unsafe_allow_html=True)
    view=st.radio('Workspace',['Executive overview','Review queue','Model lab','Case lab','Data quality','Impact simulator'],label_visibility='collapsed',key='view')
    st.markdown('<div class="sidebar-rule"></div>',unsafe_allow_html=True)
    st.caption('ANALYTICAL CONTROLS')
    model_name=st.selectbox('Detector',list(MODEL_NAMES),format_func=lambda n:MODEL_NAMES[n],key='detector')
    percentile=st.slider('Calibration percentile',80,99,int(bundle['percentile']),disabled=model_name=='rules',key='sensitivity',help='Higher percentile means fewer flags. Reference distribution: June calibration cases only.')
    period=st.selectbox('Completion cohort',['All periods']+sorted(raw.month.unique()),key='period')
    system=st.selectbox('Source attribution',['All systems']+sorted(raw.system.unique()),key='system')
    category=st.selectbox('Category',['All categories']+sorted(raw.category.unique()),key='category')
    st.caption('Global filters apply to overview, reviews, model lab and data quality. Case lab and impact are independent.')
    st.markdown('<div class="sidebar-rule"></div>',unsafe_allow_html=True)
    st.caption('● SIMULATION ENVIRONMENT\n\n1,200 synthetic cases · seed 8920\n\nNo live connections or operational actions.')
scored=score(raw,bundle,model_name,percentile)
mask=(True if period=='All periods' else scored.month==period)&(True if system=='All systems' else scored.system==system)&(True if category=='All categories' else scored.category==category)
selected=scored.copy() if isinstance(mask,bool) else scored.loc[mask].copy()
flags=selected[selected.flagged].sort_values('anomaly_score',ascending=False)
headings={
'Executive overview':('YOUR PROCUREMENT, IN FOCUS','A clearer view. Better oversight.','Monitor the portfolio, spot unusual cases and put context before decisions.'),
'Review queue':('FROM SIGNAL TO HUMAN REVIEW','Focus attention where it matters.','Inspect explainable context signals and record a review note for this session.'),
'Model lab':('A BENCHMARK YOU CAN EXAMINE','Understand the trade-off.','Two unsupervised detectors, a transparent baseline and a genuinely held-out period.'),
'Case lab':('EXPLORE WITHOUT CHANGING RECORDS','What does an unusual case look like?','Score an individual example or a CSV with the already-trained detector.'),
'Data quality':('START WITH TRUSTWORTHY INPUTS','Know the data behind the signal.','Inspect missing values, source attribution, KPI definitions and the simulation design.'),
'Impact simulator':('ASSUMPTIONS, MADE EXPLICIT','Build a business case you can explain.','Estimate reporting capacity released and review cost without inventing avoided losses.')}
title(*headings[view])
st.markdown('<div class="demo-strip"><b>SYNTHETIC DEMONSTRATION</b><span>Independent academic prototype · no real contracts, supplier identities or system integrations.</span></div>',unsafe_allow_html=True)
if selected.empty and view not in ['Case lab','Impact simulator']:
    st.info('No matching cases. Adjust the sidebar filters.');st.stop()

if view=='Executive overview':
    c=st.columns(4)
    c[0].metric('Actual procurement spend',money(selected.actual_aed.sum()),help='Sum of actual spend for completed cases in scope.')
    gap=selected.cycle_days.mean()-selected.planned_days.mean()
    c[1].metric('Average cycle time',f'{selected.cycle_days.mean():.1f} days',f'{gap:+.1f} days vs plan',delta_color='inverse')
    known=selected.late_days.dropna()
    c[2].metric('On-time delivery',pct(float((known==0).mean())) if len(known) else '—',help='Zero late days / cases with known late days. Missing values excluded.')
    c[3].metric('Cases for review',len(flags),f'{len(flags)/len(selected):.1%} of cases',delta_color='off')
    st.write('')
    l,r=st.columns([1.65,1])
    with l,st.container(border=True):
        st.subheader('Cycle time at a glance');caption('Actual and planned monthly averages · same completion cohort')
        chart(charts.cycle_trend(selected),'cycle')
    with r,st.container(border=True):
        st.subheader('Priority signals');caption(f'{MODEL_NAMES[model_name]} · ranked by anomaly score')
        if flags.empty:st.success('No alerts at the current cutoff.')
        for _,case in flags.head(3).iterrows():
            st.markdown(f'**{case.id}** · {case.supplier}')
            st.caption(case.signals)
            st.markdown(f'<span class="tag">{case.cycle_days} cycle days</span> <span class="tag">{case.variance_pct:+.1f}% spend variance</span>',unsafe_allow_html=True)
            st.divider()
        st.button('Open review queue →',on_click=lambda:st.session_state.update(view='Review queue'),key='open_queue')
    l,r=st.columns([1.1,1])
    with l,st.container(border=True):
        st.subheader('Approved and actual spend');caption('Amounts shown in AED millions; spend is not a measure of savings.')
        chart(charts.category_spend(selected),'spend')
    with r,st.container(border=True):
        st.subheader('Portfolio context');chart(charts.scatter(selected),'scatter')
        caption('Dotted lines are context-rule limits, not the detector’s decision boundary.')
    with st.container(border=True):
        st.subheader('Supplier comparison')
        agg=selected.groupby('supplier').agg(Cases=('id','count'),Actual_spend_AED=('actual_aed','sum'),Avg_cycle_days=('cycle_days','mean'),Flags=('flagged','sum'))
        known=selected[selected.late_days.notna()].copy();known['on_time']=known.late_days==0
        agg['On_time_rate']=known.groupby('supplier').on_time.mean()
        st.dataframe(agg.sort_values(['Flags','Actual_spend_AED'],ascending=False),width='stretch',
            column_config={'Actual_spend_AED':st.column_config.NumberColumn('Actual spend (AED)',format='%.0f'),
                           'Avg_cycle_days':st.column_config.NumberColumn('Avg. cycle days',format='%.1f'),
                           'On_time_rate':st.column_config.ProgressColumn('On-time rate',min_value=0,max_value=1,format='percent')})
        caption('Case mix and sample sizes vary. Flags do not establish supplier quality or justify procurement awards.')
    csv_download(selected,'Download selected case scores','synthetic_selected_scores.csv','overview_download')
    caption('Training and calibration rows are diagnostic views. Predictive metrics are reported only for the later held-out test period.')

elif view=='Review queue':
    reviews=st.session_state.reviews
    completed=sum(r.id in reviews for _,r in flags.iterrows())
    c=st.columns(3);c[0].metric('Flagged cases',len(flags));c[1].metric('Awaiting review',len(flags)-completed);c[2].metric('Spend on flagged cases',money(flags.actual_aed.sum()))
    caption('Flagged spend is the total spend attached to alerts, not financial loss or recoverable savings.')
    q=st.text_input('Search case, supplier or context signal',placeholder='e.g. PR-26-0100 or delivery delay',key='queue_search')
    state=st.selectbox('Review status',['All flagged cases','Awaiting review','Reviewed'],key='review_state')
    visible=flags.copy();visible['Review status']=visible.id.map(lambda k:'Reviewed' if k in reviews else 'Awaiting review')
    if q:visible=visible[visible[['id','supplier','signals']].astype(str).agg(' '.join,axis=1).str.contains(q,case=False,regex=False)]
    if state!='All flagged cases':visible=visible[visible['Review status']==state]
    st.dataframe(visible[['id','month','supplier','category','cycle_days','variance_pct','risk_percentile','Review status','signals']],hide_index=True,width='stretch',
        column_config={'risk_percentile':st.column_config.NumberColumn('Calibration rank',format='%.1f'),'variance_pct':st.column_config.NumberColumn('Spend variance (%)',format='%.1f')})
    if not visible.empty:
        case_id=st.selectbox('Inspect a case',visible.id.tolist(),key='inspect_id')
        row=visible.set_index('id').loc[case_id]
        l,r=st.columns([1.1,1])
        with l,st.container(border=True):
            st.markdown(f'<div class="case-id">{html.escape(case_id)}</div>',unsafe_allow_html=True)
            st.caption(f'{row.supplier} · {row.category} · {row.month}')
            st.markdown(f'<div class="signal">{html.escape(row.signals)}</div>',unsafe_allow_html=True)
            detail=pd.DataFrame({'Measure':['Cycle actual / planned','Actual / approved spend','Spend variance','Change orders','Delivery lateness','Cohort'],
                'Value':[f'{row.cycle_days} / {row.planned_days} days',f'{money(row.actual_aed)} / {money(row.budget_aed)}',f'{row.variance_pct:+.1f}%',str(row.change_orders),'Missing' if pd.isna(row.late_days) else f'{row.late_days:.0f} days',row.cohort]})
            st.dataframe(detail,hide_index=True,width='stretch')
            if pd.isna(row.late_days):st.warning('Lateness is missing. Model used the training median; confirm the source data.')
        with r,st.container(border=True):
            st.subheader('Record a review')
            note=st.text_area('Reviewer note',value=reviews.get(case_id,{}).get('note',''),key='review_note_'+case_id,placeholder='What did you inspect? What needs follow-up?')
            a,b=st.columns(2)
            if a.button('Mark reviewed',type='primary',key='mark_reviewed'):
                reviews[case_id]={'note':note,'status':'Reviewed'};st.success('Saved for this session.');st.rerun()
            if b.button('Reopen case',key='reopen_case'):
                reviews.pop(case_id,None);st.rerun()
            caption('Reviews live in this Streamlit session only. Download them to keep a copy. No operational records are changed.')
    else:st.info('No matching flagged cases. Clear the search or adjust the threshold.')
    st.download_button('Download review notes',json.dumps(reviews,indent=2),'synthetic_review_notes.json','application/json',key='reviews_download')
    csv_download(visible,'Download review queue','synthetic_review_queue.csv','queue_download')

elif view=='Model lab':
    st.info('Fit: Jan–May (670 cases) → calibrate: June (134 cases) → held-out test: Jul–Sep (396 cases). Labels never tune models or cutoffs.')
    test=selected[selected.month>bundle['calibration_end']]
    if test.empty:st.warning('This selection has no held-out test records. Choose July–September or All periods.')
    else:
        m=evaluate(test)
        c=st.columns(4)
        for col,key,label in zip(c,['precision','recall','f1','alerts'],['Precision','Recall','F1 score','Held-out alerts']):col.metric(label,str(m[key]) if key=='alerts' else pct(m[key]))
        l,r=st.columns([1,1.5])
        with l,st.container(border=True):
            st.subheader('Confusion matrix');chart(charts.matrix(m),'matrix')
            caption(f"Detected {m['tp']} of {m['tp']+m['fn']} injected anomalies; {m['fp']} false alerts. Evaluated on {m['cases']} selected test cases.")
        with r,st.container(border=True):
            st.subheader('Where the cutoff falls')
            if model_name=='rules':
                st.write('Rules flag any case with cycle ratio > 1.35, spend variance > 10%, change orders ≥ 3 or imputed lateness > 10 days.')
                caption('Rules use fixed limits, with no percentile cutoff or continuous risk rank.')
            else:
                cutoff=float(np.percentile(bundle['calibration_scores'][model_name],percentile))
                chart(charts.score_distribution(test,cutoff),'distribution')
        comparisons=[]
        for name in MODEL_NAMES:
            scored_model=score(raw.loc[test.index],bundle,name,percentile)
            comparisons.append({'Model':MODEL_NAMES[name],**evaluate(scored_model)})
        comp=pd.DataFrame(comparisons)
        with st.container(border=True):
            st.subheader('Compare the detectors');chart(charts.model_bars(comp),'model_bars')
            st.dataframe(comp[['Model','precision','recall','f1','alerts','tp','fp','fn','tn']],hide_index=True,width='stretch',
                column_config={k:st.column_config.NumberColumn(k.title(),format='%.3f') for k in ['precision','recall','f1']})
            csv_download(comp,'Download comparison','synthetic_model_comparison.csv','comparison_download')
        caption('Both ML detectors use the same calibration percentile. Rules have fixed limits, so alert counts differ. No operational winner is selected. Repeatedly changing the slider against test labels is exploratory and invalidates the test as an untouched final evaluation.')
    with st.expander('Model card & limitations',expanded=False):
        st.markdown('''**Isolation Forest:** 250 trees, random seed 8920; no feature scaling. Larger negative score_samples means more unusual.\n\n**One-Class SVM:** RBF kernel, gamma=scale, nu=0.08; training-fitted RobustScaler. Nu is a model setting, not an expected alert rate.\n\n**Preprocessing:** training-only median imputation for missing delivery lateness. Features: cycle ratio, spend variance %, amendment count, late days. Raw missing values stay missing for operational KPIs.\n\n**Thresholds:** fixed 90th percentile of the separate June score distribution by default. No synthetic truth labels tune the cutoff. Test period starts in July. Calibration data can contain injected anomalies.\n\n**Explainability:** visible context rules are not SHAP values, learned feature importance, probabilities or causal explanations.\n\n**Validity:** injected synthetic patterns are easier than many real procurement issues. No claim of real-world precision, savings or deployment readiness is supported. Validate with approved data, expert labels and stable workload limits before use.''')

elif view=='Case lab':
    st.caption('Uses the selected detector and sensitivity. Portfolio filters do not affect this lab. This scores examples without retraining.')
    manual,upload=st.tabs(['Build a case','Score a CSV'])
    with manual:
        with st.form('case_form'):
            a,b,c=st.columns(3)
            planned=a.number_input('Planned cycle (days)',min_value=1,value=40,step=1)
            actual=a.number_input('Actual cycle (days)',min_value=1,value=65,step=1)
            budget=b.number_input('Approved spend (AED)',min_value=1.0,value=500000.0,step=10000.0)
            spend=b.number_input('Actual spend (AED)',min_value=0.0,value=575000.0,step=10000.0)
            changes=c.number_input('Change orders',min_value=0,value=4,step=1)
            late=c.number_input('Delivery lateness (days)',min_value=0,value=12,step=1)
            missing=st.checkbox('Delivery lateness is unknown')
            run_case=st.form_submit_button('Score this synthetic case',type='primary')
        if run_case:
            example=pd.DataFrame([dict(id='WHATIF-001',month='2026-10',supplier='Example supplier',category='General supplies',system='Manual example',planned_days=planned,cycle_days=actual,budget_aed=budget,actual_aed=spend,variance_pct=(spend/budget-1)*100,change_orders=changes,late_days=np.nan if missing else late)])
            result=score(example,bundle,model_name,percentile);st.session_state.case_result=result
            st.session_state.case_settings=(model_name,percentile)
        if 'case_result' in st.session_state:
            result=st.session_state.case_result;case=result.iloc[0]
            st.subheader('Flagged for human review' if case.flagged else 'Below the current alert cutoff')
            used_model,used_p=st.session_state.case_settings
            st.caption(f'Scored using {MODEL_NAMES[used_model]}'+(f' at the {used_p}th calibration percentile.' if used_model!='rules' else '.'))
            st.markdown(f'<div class="signal">{html.escape(case.signals)}</div>',unsafe_allow_html=True)
            if used_model!='rules':st.metric('Rank against calibration cases',f'{case.risk_percentile:.1f}th percentile')
            csv_download(result,'Download case result','synthetic_case_result.csv','case_download')
            caption('This is a statistical result, not an approval, rejection or procurement recommendation. Resubmit the form after changing the detector or threshold.')
    with upload:
        st.write('Upload a matching CSV to score using the fitted model. Input schema is checked before scoring; this never retrains the model.')
        st.download_button('Download an unlabeled example CSV',(ROOT/'data'/'new_cases_example.csv').read_bytes(),'new_cases_example.csv','text/csv',key='template')
        uploaded=st.file_uploader('CSV file (up to 50,000 rows)',type=['csv'])
        if uploaded is not None:
            try:
                uploaded_df=pd.read_csv(uploaded)
                result=score(uploaded_df,bundle,model_name,percentile)
                st.success(f'Scored {len(result):,} rows. Flagged {int(result.flagged.sum()):,} cases.')
                st.dataframe(result,hide_index=True,width='stretch')
                csv_download(result,'Download scores','uploaded_case_scores.csv','upload_scores')
            except (ValueError,KeyError,TypeError,pd.errors.ParserError) as exc:st.error(f'Cannot score this file: {exc}')
        caption('Inputs are processed in the running Python process and are not intentionally written to disk or sent to external services. Use approved data only; this prototype has no production access controls. Labels are optional and never used for scoring.')

elif view=='Data quality':
    q=data_report(selected)
    c=st.columns(4);c[0].metric('Cases in scope',q['rows']);c[1].metric('Delivery completeness',pct(q['delivery_completeness']));c[2].metric('Missing lateness values',q['missing_late_days']);c[3].metric('Duplicate case IDs',q['duplicate_ids'])
    l,r=st.columns([1.4,1])
    with l,st.container(border=True):
        st.subheader('Raw data, visible gaps');st.dataframe(selected.drop(columns=['imputed_late_days']),hide_index=True,width='stretch')
        caption('Lateness remains blank where it is missing. Only the model uses an imputed value. No rows are dropped to improve metrics.')
    with r,st.container(border=True):
        st.subheader('Readiness checks')
        for name,desc in [('Unique IDs','Duplicate or missing case identifiers are rejected.'),('Valid time cohorts','Month must be YYYY-MM.'),('Reconciled spend','Variance must match actual / approved spend.'),('Valid numeric ranges','Positive cycle durations and budgets; nonnegative late days and amendments.'),('Explicit missingness','Missing lateness is supported; other required analytic fields must be resolved.')]:
            st.markdown('**✓ '+name+'**');st.caption(desc)
    with st.expander('Simulation design & KPI definitions',expanded=True):
        st.markdown('''**Data:** 1,200 completed cases, Jan–Sep 2026; 30 fictitious suppliers; 4 generic categories. Source attribution cycles among TMT, TMS and Oracle. It does not simulate joins or validate actual system integration.\n\n**Generator:** seed 8920. Planned cycles 25–60 days; actual cycles normally distributed around plan minus four days; spend variance centered on zero with a 3 percentage-point standard deviation; change orders drawn from a Poisson distribution with mean 0.5. About 8% of cases receive injected unusual behavior. Delivery lateness is removed randomly from about 3.5% of cases.\n\n**On-time delivery:** zero late days / records with known late days. **Cycle time:** unweighted average of actual request-to-completion days. **Spend:** sum of actual amounts. **Spend on flagged cases:** total spend attached to alerts, not loss. **Anomaly rank:** percentile relative to calibration cases, not a risk probability.\n\n**Project alignment:** source readiness, performance trends, anomaly monitoring and human review match the supplied procurement dashboard proposal. The report and presentation briefs request a visual prototype, suitable metrics, model interpretation and business impact. This package demonstrates those mechanics; synthetic data do not replace a requirement for real secondary evidence. Follow the assessment’s restrictions on AI-assisted submitted writing.''')
    csv_download(selected,'Download data in scope','synthetic_data_quality_selection.csv','quality_download')

elif view=='Impact simulator':
    st.caption('Independent scenario; not affected by portfolio filters. Assumptions are editable and not observed organizational values.')
    l,r=st.columns([1,1.3])
    with l,st.container(border=True):
        st.subheader('Your scenario assumptions')
        hours=st.slider('Manual reporting hours / month',40,400,160,10)
        rate=st.slider('Loaded staff cost / hour (AED)',50,500,180,10)
        reduction=st.slider('Reporting effort reduction (%)',0,80,35,5)/100
        review_minutes=st.slider('Review time / alert (minutes)',0,120,15,5)
        monthly_cases=st.number_input('Completed cases / month',min_value=1,value=133,step=1)
        setup=st.slider('One-time setup cost (AED)',0,150000,45000,5000)
        monthly_cost=st.slider('Recurring cost / month (AED)',0,10000,1500,250)
        caption('Review workload uses the fixed default held-out alert rate for the chosen detector; it is not a forecast of operational alerts.')
    test_full=score(raw[raw.month>bundle['calibration_end']],bundle,model_name,bundle['percentile'])
    alert_rate=float(test_full.flagged.mean())
    released=hours*12*reduction;review_hours=monthly_cases*alert_rate*12*review_minutes/60
    gross=released*rate;review_cost=review_hours*rate;year_cost=setup+monthly_cost*12+review_cost;net=gross-year_cost
    contribution=(gross-monthly_cost*12-review_cost)/12
    with r,st.container(border=True):
        st.subheader('Year-one impact, including review work')
        st.metric('Net benefit',money(net))
        c=st.columns(3);c[0].metric('Year-one ROI',pct(net/year_cost) if year_cost else '—');c[1].metric('Hours released',f'{released:.0f}');c[2].metric('Payback',('0 months' if setup==0 else f'{setup/contribution:.1f} months' if contribution>0 else 'No payback'))
        waterfall=pd.DataFrame({'Item':['Reporting capacity value','Alert review cost','Recurring platform cost','One-time setup'],'AED':[gross,review_cost,monthly_cost*12,setup]})
        st.dataframe(waterfall,hide_index=True,width='stretch',column_config={'AED':st.column_config.NumberColumn(format='%.0f')})
        st.write(f'Estimated alert-review time: **{review_hours:.1f} hours/year** at a **{alert_rate:.1%}** synthetic alert rate.')
        caption('Released capacity is not guaranteed cash savings. No avoided contract loss or causal procurement benefit is assumed. A model can have a negative business case.')
        st.code('Gross value = hours/month × 12 × reduction × staff rate\nReview cost = cases/month × alert rate × 12 × review minutes/60 × staff rate\nNet benefit = gross value − review cost − recurring cost − setup\nROI = net benefit / all year-one costs\nPayback = setup / monthly net recurring benefit',language=None)
        csv_download(pd.DataFrame([dict(reporting_hours_per_month=hours,staff_rate_aed=rate,reduction_pct=reduction*100,review_minutes=review_minutes,monthly_cases=monthly_cases,synthetic_alert_rate=alert_rate,setup_aed=setup,recurring_monthly_aed=monthly_cost,gross_value_aed=gross,review_cost_aed=review_cost,net_year_one_aed=net)]),'Download assumptions & result','illustrative_business_case.csv','roi_download')
    with st.expander('A practical path to a pilot'):
        st.markdown('1. Approve scope, KPI definitions, data ownership and security classification.\n2. Map source fields and test joins, duplication and completeness using approved records.\n3. Obtain expert labels, compare baseline rules and set an acceptable alert workload.\n4. Pilot with human reviewers; measure actual reporting time and alert usefulness.\n5. Review drift, access controls, audit requirements and value before expansion.')

st.markdown('<div class="footer">PROCUREMENT PULSE V2 · Synthetic academic workbench · All findings require real-data validation · No automated procurement actions</div>',unsafe_allow_html=True)
