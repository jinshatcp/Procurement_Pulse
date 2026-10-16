"""Run: python tests/test_app.py (requires requirements.txt and generated outputs)."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from streamlit.testing.v1 import AppTest
root=Path(__file__).resolve().parents[1]
a=AppTest.from_file(str(root/'dashboard.py'),default_timeout=40).run()
def check(label):
    assert not a.exception,f'{label}: {a.exception}'
    print('PASS:',label)
check('Executive overview')
assert len(a.get('plotly_chart'))==3
for view in ['Review queue','Model lab','Case lab','Data quality','Impact simulator']:
    a.radio(key='view').set_value(view).run();check(view)
a.radio(key='view').set_value('Model lab').run()
a.selectbox(key='detector').set_value('one_class_svm').run();check('Model switching')
a.slider(key='sensitivity').set_value(95).run();check('Sensitivity adjustment')
a.selectbox(key='period').set_value('2026-01').run();check('Training-only cohort')
assert any('no held-out' in x.value for x in a.warning)
a.selectbox(key='period').set_value('All periods').run()
a.selectbox(key='system').set_value('TMT').run();check('Source filtering')
a.selectbox(key='system').set_value('All systems').run()
a.selectbox(key='detector').set_value('rules').run();check('Rules baseline')
a.radio(key='view').set_value('Case lab').run()
a.button(key='FormSubmitter:case_form-Score this synthetic case').click().run();check('Manual case scoring')
assert len(a.session_state['case_result'])==1
a.selectbox(key='detector').set_value('isolation_forest').run()
a.radio(key='view').set_value('Review queue').run()
a.button(key='mark_reviewed').click().run();check('Mark reviewed')
assert len(a.session_state['reviews'])==1
a.button(key='reopen_case').click().run();check('Reopen case')
assert len(a.session_state['reviews'])==0
print('All Streamlit interaction checks passed. This is not a browser screenshot test.')
