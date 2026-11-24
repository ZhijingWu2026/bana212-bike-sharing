"""Bike-sharing demand explorer. Loads saved models; never trains inside the app."""
from pathlib import Path
from datetime import date
import json
import joblib
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='Bike-sharing demand | BANA 212', page_icon='🚲', layout='wide')

@st.cache_resource
def load_models():
    # Only load the trusted model file shipped with this project.
    return joblib.load(ROOT / 'models.joblib')

@st.cache_data
def load_data():
    df = pd.read_csv(ROOT / 'hour.csv')
    df['dteday'] = pd.to_datetime(df['dteday'])
    df['timestamp'] = df['dteday'] + pd.to_timedelta(df['hr'], unit='h')
    results = pd.read_csv(ROOT / 'test_results.csv')
    predictions = pd.read_csv(ROOT / 'test_predictions.csv', parse_dates=['timestamp'])
    metadata = json.loads((ROOT / 'metadata.json').read_text())
    return df, results, predictions, metadata

def predict(model_name, row, bundle):
    if model_name == 'Historical Average':
        means = bundle['average']
        match = means[(means['workingday'] == int(row['workingday'].iloc[0])) &
                      (means['hr'] == int(row['hr'].iloc[0]))]
        return float(match['cnt'].iloc[0]) if len(match) else bundle['global_mean']
    categories = bundle['encoder'].transform(row[bundle['cat_cols']])
    inputs = np.hstack([categories, row[bundle['num_cols']]])
    return max(0.0, float(bundle[model_name].predict(inputs)[0]))

try:
    bundle = load_models()
    df, results, predictions, meta = load_data()
except FileNotFoundError:
    st.error('The deployment is missing the saved data or model files. Include the complete project folder.')
    st.stop()

with st.sidebar:
    st.image(str(ROOT / 'commuter_scene.png'))
    st.markdown('### BANA 212\nCapital Bikeshare demand study')
    st.caption('Washington, D.C. · 2011–2012')
    st.caption('Conceptual illustration generated with AI.')
    st.markdown('[Dataset and citation](https://doi.org/10.24432/C5W894)')
    st.download_button('Download analysis notebook', (ROOT/'notebook.ipynb').read_bytes(),
                       file_name='BANA212_Bike_Sharing.ipynb', mime='application/x-ipynb+json')

st.title('When will bike demand rise?')
st.markdown('Explore hourly rental patterns, compare prediction methods, and try a historical demand scenario.')
st.caption('Historical study • Predictions count rental events across the system, not bicycles required at a station.')
summary, patterns, performance, interactive = st.tabs([
    'Executive summary', 'Descriptive analytics', 'Model performance', 'Interactive prediction'])

with summary:
    st.subheader('The business question')
    st.write('How well can time and weather estimate hourly bike rentals? Anticipating busy periods can help operators prepare staffing and monitor demand. This study supports planning decisions without claiming measured cost savings or station-level allocation recommendations.')
    st.subheader('Data and target')
    st.write('The UCI Bike Sharing dataset contains 17,379 recorded hours in Washington, D.C., during 2011–2012. We predict cnt, the number of rental events in each hour. The table has no empty cells or duplicate hours, but 165 hours have no record. We leave those gaps unfilled and exclude casual and registered because they sum to cnt.')
    st.subheader('Approach and findings')
    st.write('We compare a historical mean for the same hour and day type with Linear Regression and Random Forest. Earlier records support training and validation; October–December 2012 provides the test period. Random Forest lowers test RMSE by about 20.4% relative to historical averages, but still underestimates high-demand hours. Working and non-working days also have different demand profiles.')
    a,b,c = st.columns(3)
    a.metric('Recorded hours', f'{len(df):,}')
    b.metric('Random Forest test RMSE', '95.22', help='Rentals per hour. Lower is better.')
    c.metric('RMSE reduction vs baseline', f"{meta['rmse_improvement']:.1f}%")
    st.info('Practical use: distinguish working days from non-working days and monitor high-demand periods closely. Estimates should inform operational judgment rather than determine station inventory automatically.')
    with st.expander('Study limitations', expanded=True):
        st.markdown('''- One city and two historical years limit generalization to current demand.
- Weather inputs are observed values. Advance forecasts would require forecast weather and new evaluation.
- Only three severe-weather records are available.
- An earlier exploratory version examined the test period, so this is not a fresh independent test.
- We compared a small set of model settings. Rolling time validation is a possible next step.''')

with patterns:
    st.subheader('Working days have distinct commute peaks')
    hourly = df.groupby(['hr','workingday'],as_index=False)['cnt'].mean()
    hourly['Day type'] = hourly['workingday'].map({0:'Non-working day',1:'Working day'})
    chart = alt.Chart(hourly).mark_line(point=True).encode(
        x=alt.X('hr:Q',title='Hour of day',scale=alt.Scale(domain=[0,23])),
        y=alt.Y('cnt:Q',title='Mean hourly rentals'),
        color=alt.Color('Day type:N',scale=alt.Scale(range=['#D87537','#087F8C'])),
        tooltip=['Day type',alt.Tooltip('hr:Q',title='Hour'),alt.Tooltip('cnt:Q',title='Mean rentals',format='.1f')])
    st.altair_chart(chart.properties(height=310),width='stretch')
    st.caption('Working days have peaks around 08:00 and 17:00–18:00, while non-working days peak around midday and afternoon. These full-dataset descriptive averages suggest using different hourly planning profiles for the two day types.')
    st.subheader('Monthly patterns differ between years')
    monthly = df.groupby(['mnth','yr'],as_index=False)['cnt'].mean()
    monthly['Year'] = (monthly['yr']+2011).astype(str)
    st.altair_chart(alt.Chart(monthly).mark_line(point=True).encode(
        x=alt.X('mnth:O',title='Month'),y=alt.Y('cnt:Q',title='Mean hourly rentals'),
        color=alt.Color('Year:N',scale=alt.Scale(range=['#D87537','#087F8C'])),
        tooltip=['Year','mnth',alt.Tooltip('cnt:Q',format='.1f')]).properties(height=290),width='stretch')
    st.caption('Mean hourly rentals vary across months and are generally higher in 2012. Two years of data cannot establish a lasting growth trend, and these values are hourly averages rather than monthly totals.')
    st.subheader('Weather groups have different average demand')
    weather = df.groupby('weathersit')['cnt'].agg(['count','mean']).reset_index()
    weather['Weather'] = weather['weathersit'].map({1:'Clear / partly cloudy',2:'Mist / cloudy',3:'Light rain / snow',4:'Severe'})
    st.altair_chart(alt.Chart(weather).mark_bar(color='#087F8C').encode(
        x=alt.X('Weather:N',sort=weather['Weather'].tolist(),axis=alt.Axis(labelAngle=0)),
        y=alt.Y('mean:Q',title='Mean hourly rentals'),tooltip=['Weather','count',alt.Tooltip('mean:Q',format='.1f')]
    ).properties(height=280),width='stretch')
    st.caption('Less favorable weather groups generally show lower mean demand, but time and season also differ between groups. Severe weather has only three records, so its mean provides very limited evidence and does not establish a causal effect.')
    st.dataframe(weather[['Weather','count','mean']].rename(columns={'count':'Recorded hours','mean':'Mean rentals'}),hide_index=True,width='stretch')

with performance:
    st.subheader('Evaluation across time')
    st.dataframe(pd.DataFrame({'Period':['Training','Validation','Test'],
        'Dates':['Jan 2011 – Jun 2012','Jul – Sep 2012','Oct – Dec 2012'],
        'Recorded hours':[meta['training_rows'],meta['validation_rows'],meta['test_rows']]}),hide_index=True,width='stretch')
    st.write('The validation period selects between two forest depths. Final models use all pre-test records, with the encoder fitted on the original training data. The test period was previously examined during exploratory work.')
    st.subheader('Test results')
    st.dataframe(results.round(2),hide_index=True,width='stretch')
    st.caption('MAE is average absolute error in rentals per hour. RMSE gives large errors more weight; lower values are better. R² is not percentage accuracy.')
    errors = results.melt(id_vars='Method',value_vars=['MAE','RMSE'],var_name='Metric',value_name='Error')
    st.altair_chart(alt.Chart(errors).mark_bar().encode(x=alt.X('Method:N',axis=alt.Axis(labelAngle=0)),
        xOffset='Metric:N',y=alt.Y('Error:Q',title='Error in rentals per hour'),
        color=alt.Color('Metric:N',scale=alt.Scale(range=['#D87537','#087F8C'])),
        tooltip=['Method','Metric',alt.Tooltip('Error:Q',format='.2f')]).properties(height=300),width='stretch')
    st.caption('Random Forest has the lowest test MAE and RMSE among these methods, reducing RMSE by approximately 20.4% relative to the historical baseline. Linear Regression does not outperform that baseline on this test quarter, and the error reduction does not represent cost savings.')
    st.subheader('Actual and predicted rentals')
    shown = st.selectbox('Method shown in the time plot',results['Method'].tolist(),index=2,key='plot_method')
    first_week = predictions[predictions['timestamp'] < predictions['timestamp'].min()+pd.Timedelta(days=7)]
    series = first_week[['timestamp','cnt',shown]].rename(columns={'cnt':'Actual'}).melt('timestamp',var_name='Series',value_name='Rentals')
    st.altair_chart(alt.Chart(series).mark_line().encode(x=alt.X('timestamp:T',title='Time'),y='Rentals:Q',
        color=alt.Color('Series:N',scale=alt.Scale(range=['#132F46','#D87537'])),tooltip=['timestamp:T','Series',alt.Tooltip('Rentals:Q',format='.1f')]).properties(height=300),width='stretch')
    st.caption('The plot uses the first seven days of the test period, rather than selecting dates with unusually good performance. Compare peak heights as well as timing, since an overall error score can hide underprediction during busy hours.')
    st.subheader('High-demand hours need extra attention')
    peak = pd.read_csv(ROOT/'peak_results.csv')
    st.dataframe(peak.round(2),hide_index=True,width='stretch')
    st.caption('High demand means actual rentals at or above the pre-test 90th percentile of 441. Across 318 such test hours, Random Forest underpredicts by about 162 rentals per hour on average; grouping by actual demand is retrospective error analysis.')
    with st.expander('Selected settings and validation comparison'):
        st.write('Random Forest: 100 trees, unrestricted maximum depth, minimum leaf size 2, random seed 42. These settings are the selected configuration among the two depth candidates tested.')
        st.json(meta['parameters'])
        st.write('Validation RMSE by forest maximum depth:')
        st.json(meta['validation_depths'])
    st.download_button('Download test metrics',results.to_csv(index=False),'test_results.csv','text/csv')

with interactive:
    st.subheader('Try the saved models')
    st.write('Choose a model and scenario. Predictions update from saved models without retraining.')
    selected = st.selectbox('Prediction model',['Random Forest','Linear Regression','Historical Average'],key='prediction_model')
    mode = st.radio('Scenario',['Recorded demo','Change the hour','Custom scenario'],horizontal=True,key='scenario_mode')
    row = bundle['example'].copy()
    if mode == 'Recorded demo':
        st.write('Recorded test case: October 1, 2012 at 08:00. All inputs use the historical observation.')
    elif mode == 'Change the hour':
        row['hr'] = st.slider('Hypothetical hour',0,23,14,key='demo_hour')
        st.write('Only the hour differs from the recorded demo. All other model inputs stay fixed, and no actual outcome is assigned to this hypothetical case.')
    else:
        a,b = st.columns(2)
        chosen_date = a.date_input('Historical calendar date',value=date(2012,10,1),min_value=date(2011,1,1),max_value=date(2012,12,31),key='date')
        hour = b.slider('Hour of day',0,23,8,key='custom_hour')
        # Derive mutually consistent calendar variables from an observed date.
        row = df[df['dteday'].dt.date == chosen_date].iloc[[0]].copy()
        for col in ['temp','atemp','hum','windspeed']:
            row[col] = bundle['numeric_means'][col]
        row['hr'] = hour
        options = {'Clear / partly cloudy':1,'Mist / cloudy':2,'Light rain / snow':3}
        row['weathersit'] = options[st.selectbox('Weather category',list(options),key='weather')]
        row['temp'] = st.slider('Normalized temperature',0.0,1.0,float(bundle['numeric_means']['temp']),step=.01,key='temp')
        row['hum'] = st.slider('Normalized humidity',0.0,1.0,float(bundle['numeric_means']['hum']),step=.01,key='humidity')
        st.caption('Apparent temperature and wind speed stay at pre-test averages. Calendar fields follow the selected historical date. Values are in the dataset’s normalized units, not degrees Celsius. Severe weather is excluded from this interface because only three observations exist.')
        st.info('This is a hypothetical input combination, even when its date matches a recorded day. Independent temperature settings can create unusual combinations with mean apparent temperature; use the recorded demo for an internally observed case.')
    value = predict(selected,row,bundle)
    a,b = st.columns(2)
    a.metric('Predicted rental events per hour',f'{value:,.1f}')
    if mode == 'Recorded demo':
        actual = int(row['cnt'].iloc[0])
        b.metric('Actual rental events',f'{actual:,}')
        st.caption(f'Prediction minus actual: {value-actual:+.1f} rentals. A negative value means underprediction.')
    else:
        b.markdown('**Hypothetical scenario**\n\nNo verified actual outcome.')
    if selected == 'Historical Average':
        st.caption('This baseline uses only working-day status and hour. Changing weather or numerical weather inputs will not change its prediction.')
    st.caption('These are historical demand estimates, not present-day forecasts, causal effects, or required bicycle inventory.')
    with st.expander('View all model inputs'):
        st.dataframe(row[bundle['features']],hide_index=True,width='stretch')
        st.caption('yr: 0 = 2011, 1 = 2012; season: 1–4; weekday: 0 = Sunday; holiday and workingday: 0/1. Weather: 1 = clear, 2 = mist, 3 = light rain/snow.')

st.divider()
st.caption('BANA 212 · Donya Bagheri, Chia-Chi Chang, Siyi Li, Zhijing Wu · Data: UCI Bike Sharing, Fanaee-T (2013), CC BY 4.0.')
